"""
Discovery V2 — Orchestrator
The brain that ties analysis, arrangement, and quality evaluation together.
Runs the iterative loop: analyze → arrange → score → adjust → repeat.

Deploy on Modal as a single endpoint that handles the full pipeline.
"""
import modal
import json
import os
import uuid

app = modal.App("discovery-orchestrator")

volume = modal.Volume.from_name("discovery-fragments", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "libfftw3-dev")
    .pip_install(
        "numpy>=1.24.0,<2.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "madmom>=0.16.1",
        "soundfile>=0.12.0",
        "fastapi[standard]",
        "httpx>=0.27.0",
    )
)


# Import the analysis and arrangement functions
from v2.analysis import analyze_track
from v2.arranger import arrange_and_evaluate


QUALITY_THRESHOLD = 0.65  # Minimum overall score to accept
MAX_ITERATIONS = 3        # Max arrangement attempts


@app.function(image=image, timeout=600, volumes={"/fragments": volume})
def full_pipeline(audio_bytes: bytes, filename: str, user_params: dict = None):
    """
    The full Discovery pipeline:
    1. Analyze the track (beats, key, structure)
    2. Arrange with default params
    3. Evaluate quality
    4. If quality < threshold, adjust params and retry
    5. Return best result
    """
    import numpy as np
    import soundfile as sf
    import tempfile
    import subprocess
    
    session_id = str(uuid.uuid4())
    user_params = user_params or {}
    
    # ── Step 1: Save original audio to volume ────────────────────────
    session_dir = f"/fragments/{session_id}"
    os.makedirs(session_dir, exist_ok=True)
    
    ext = os.path.splitext(filename)[1] or ".mp3"
    input_path = os.path.join(session_dir, f"original{ext}")
    with open(input_path, "wb") as f:
        f.write(audio_bytes)
    
    # Convert to WAV
    wav_path = os.path.join(session_dir, "original_mix.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "2", wav_path],
        capture_output=True,
    )
    volume.commit()
    
    # ── Step 2: Analyze ──────────────────────────────────────────────
    analysis = analyze_track.remote(audio_bytes, filename)
    
    results = {
        "session_id": session_id,
        "analysis": analysis,
        "iterations": [],
    }
    
    # ── Step 3: Iterative arrangement ────────────────────────────────
    best_score = 0
    best_iteration = None
    
    # Parameter presets to try
    param_variations = [
        # Default: smooth filter sweep
        {
            "bpm": analysis.get("tempo", 120),
            "duration": 120.0,
            "filter_start": 300,
            "filter_end": 12000,
            "sidechain_depth": 0.35,
            "saturation": 1.05,
        },
        # Darker, more filtered
        {
            "bpm": analysis.get("tempo", 120),
            "duration": 120.0,
            "filter_start": 200,
            "filter_end": 6000,
            "sidechain_depth": 0.3,
            "saturation": 1.1,
        },
        # Brighter, more open
        {
            "bpm": analysis.get("tempo", 120),
            "duration": 120.0,
            "filter_start": 600,
            "filter_end": 16000,
            "sidechain_depth": 0.4,
            "saturation": 1.02,
        },
    ]
    
    # Override with user params if provided
    if user_params:
        param_variations[0].update(user_params)
    
    for i, params in enumerate(param_variations[:MAX_ITERATIONS]):
        iteration_result = arrange_and_evaluate.remote(session_id, analysis, params)
        
        score = iteration_result.get("quality_scores", {}).get("overall", 0)
        
        results["iterations"].append({
            "iteration": i + 1,
            "params": params,
            "scores": iteration_result.get("quality_scores", {}),
        })
        
        if score > best_score:
            best_score = score
            best_iteration = i + 1
        
        # If quality passes threshold, stop iterating
        if score >= QUALITY_THRESHOLD:
            break
    
    results["best_iteration"] = best_iteration
    results["best_score"] = best_score
    results["passed_threshold"] = best_score >= QUALITY_THRESHOLD
    
    # Save results
    with open(os.path.join(session_dir, "pipeline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    
    return results


@app.function(image=image, timeout=600, volumes={"/fragments": volume})
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File, Request
    from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import io

    api = FastAPI(title="Discovery Orchestrator V2")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    async def health():
        return {"status": "ok", "version": "v2-orchestrator"}

    @api.post("/pipeline")
    async def pipeline(file: UploadFile = File(...)):
        """
        Full pipeline: upload → analyze → arrange → evaluate → return best.
        """
        audio_bytes = await file.read()
        try:
            result = full_pipeline.remote(audio_bytes, file.filename or "upload.mp3")
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @api.post("/pipeline/custom")
    async def pipeline_custom(request: Request):
        """Pipeline with custom parameters."""
        # Expect multipart: file + params as form fields
        form = await request.form()
        file = form.get("file")
        params_str = form.get("params", "{}")
        
        audio_bytes = await file.read()
        try:
            params = json.loads(params_str)
        except:
            params = {}
        
        try:
            result = full_pipeline.remote(audio_bytes, file.filename or "upload.mp3", params)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @api.get("/export/{session_id}")
    async def export(session_id: str, format: str = "wav"):
        """Download the arrangement."""
        volume.reload()
        wav_path = f"/fragments/{session_id}/arrangement.wav"
        if not os.path.exists(wav_path):
            return JSONResponse(status_code=404, content={"error": "No arrangement found"})
        
        if format == "m4a":
            import subprocess
            m4a_path = wav_path.replace(".wav", ".m4a")
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac", "-b:a", "192k", m4a_path],
                capture_output=True,
            )
            return FileResponse(m4a_path, media_type="audio/mp4", filename="discovery.m4a")
        
        return FileResponse(wav_path, media_type="audio/wav", filename="discovery.wav")

    @api.get("/results/{session_id}")
    async def results(session_id: str):
        """Get pipeline results (analysis + scores)."""
        volume.reload()
        path = f"/fragments/{session_id}/pipeline_results.json"
        if not os.path.exists(path):
            return JSONResponse(status_code=404, content={"error": "No results found"})
        with open(path) as f:
            return JSONResponse(content=json.load(f))

    return api
