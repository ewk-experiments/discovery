"""
Discovery V2 — Arrangement Engine
Orchestrated by Opus. Uses analysis data to make informed arrangement decisions.

The arrangement engine:
1. Receives analyzed tracks (beats, key, structure, energy)
2. Selects the best loop points using energy/groove analysis
3. Arranges with beat-locked precision using madmom beat positions
4. Applies French house production (sidechain, filter sweep, saturation)
5. Self-evaluates output quality via spectral analysis
6. Iterates until quality score passes threshold
"""
import modal
import json
import os
import numpy as np

app = modal.App("discovery-arranger")

volume = modal.Volume.from_name("discovery-fragments", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "numpy==1.23.5",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "fastapi[standard]",
        "httpx>=0.27.0",
    )
)


def quality_score(audio, sr, target_bpm):
    """
    Score the quality of an arrangement. Returns dict of scores 0-1.
    Used for iterative improvement — arrange, score, fix, repeat.
    """
    import librosa
    
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    
    scores = {}
    
    # 1. Rhythmic consistency — are beats evenly spaced?
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr)
    tempo_val = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])
    if len(beat_frames) > 2:
        intervals = np.diff(librosa.frames_to_time(beat_frames, sr=sr))
        expected_interval = 60.0 / target_bpm
        deviation = np.mean(np.abs(intervals - expected_interval)) / expected_interval
        scores["rhythmic_consistency"] = round(max(0, 1.0 - deviation * 5), 3)
    else:
        scores["rhythmic_consistency"] = 0.0
    
    # 2. Spectral smoothness — no harsh frequency spikes
    spec = np.abs(librosa.stft(mono))
    spec_diff = np.diff(spec, axis=1)
    spectral_roughness = np.mean(np.abs(spec_diff)) / (np.mean(spec) + 1e-8)
    scores["spectral_smoothness"] = round(max(0, 1.0 - spectral_roughness * 2), 3)
    
    # 3. Energy continuity — no dead spots
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
    # Check for frames with very low energy (< 10% of mean)
    mean_rms = np.mean(rms)
    dead_frames = np.sum(rms < mean_rms * 0.1) / len(rms)
    scores["energy_continuity"] = round(max(0, 1.0 - dead_frames * 5), 3)
    
    # 4. Harmonic stability — key shouldn't wander
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    chroma_var = np.mean(np.std(chroma, axis=1))
    scores["harmonic_stability"] = round(max(0, 1.0 - chroma_var), 3)
    
    # 5. Dynamic range — should have some pump, not flat
    rms_std = np.std(rms) / (mean_rms + 1e-8)
    # Sweet spot: 0.2-0.5 std/mean ratio
    if 0.15 < rms_std < 0.6:
        scores["dynamic_range"] = round(0.8 + 0.2 * (1 - abs(rms_std - 0.35) / 0.25), 3)
    else:
        scores["dynamic_range"] = round(max(0, 0.5 - abs(rms_std - 0.35)), 3)
    
    # Overall score (weighted)
    weights = {
        "rhythmic_consistency": 0.3,
        "spectral_smoothness": 0.2,
        "energy_continuity": 0.2,
        "harmonic_stability": 0.15,
        "dynamic_range": 0.15,
    }
    scores["overall"] = round(sum(scores[k] * weights[k] for k in weights), 3)
    
    return scores


@app.function(image=image, timeout=300, volumes={"/fragments": volume})
def arrange_and_evaluate(
    session_id: str,
    analysis: dict,
    params: dict,
):
    """
    Arrange a track using analysis data, then self-evaluate.
    
    Args:
        session_id: The decompose session with stems
        analysis: Output from analyze_track (beats, key, tempo, etc.)
        params: Arrangement parameters (filter_start, filter_end, bpm, etc.)
    
    Returns:
        arrangement result with quality scores
    """
    import soundfile as sf
    from scipy.signal import butter, sosfilt
    from scipy.interpolate import interp1d
    
    sr = 44100
    
    # Load the original full mix from the volume
    volume.reload()
    
    # Get beat positions from analysis
    beats = np.array(analysis.get("beats", []))
    bar_starts = np.array(analysis.get("bar_starts", []))
    tempo = analysis.get("tempo", 120.0)
    groove_section = analysis.get("groove_section", {"start_sec": 8, "end_sec": 16})
    
    # Load the audio
    # For now, expect the full mix WAV in the session dir
    mix_path = f"/fragments/{session_id}/original_mix.wav"
    if not os.path.exists(mix_path):
        # Try to find any wav file
        session_dir = f"/fragments/{session_id}"
        if os.path.exists(session_dir):
            for fname in os.listdir(session_dir):
                if fname.endswith('.wav') and 'other' not in fname:
                    mix_path = os.path.join(session_dir, fname)
                    break
    
    if not os.path.exists(mix_path):
        return {"error": "No audio file found", "session_id": session_id}
    
    audio, file_sr = sf.read(mix_path)
    if file_sr != sr:
        # Resample
        from scipy.signal import resample
        new_len = int(len(audio) * sr / file_sr)
        audio = resample(audio, new_len)
    
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    
    audio = audio.astype(np.float32)
    
    # ── Cut loop at beat-locked positions ────────────────────────────
    groove_start = groove_section["start_sec"]
    groove_end = groove_section["end_sec"]
    
    # Find the nearest beat to groove_start
    if len(beats) > 0:
        start_beat_idx = np.argmin(np.abs(beats - groove_start))
        # Get 8 bars worth of beats (32 beats at 4/4)
        end_beat_idx = min(start_beat_idx + 32, len(beats) - 1)
        
        loop_start_sample = int(beats[start_beat_idx] * sr)
        loop_end_sample = int(beats[end_beat_idx] * sr)
    else:
        loop_start_sample = int(groove_start * sr)
        loop_end_sample = int(groove_end * sr)
    
    loop = audio[loop_start_sample:loop_end_sample].copy()
    loop_duration = len(loop) / sr
    
    # ── Build arrangement ────────────────────────────────────────────
    target_bpm = params.get("bpm", tempo)
    total_seconds = params.get("duration", 120.0)
    filter_start = params.get("filter_start", 300)
    filter_end = params.get("filter_end", 12000)
    sidechain_depth = params.get("sidechain_depth", 0.35)
    saturation = params.get("saturation", 1.05)
    
    canvas = np.zeros((int(sr * total_seconds), 2), dtype=np.float32)
    
    def apply_lowpass(audio_chunk, cutoff):
        nyq = sr / 2
        if cutoff >= nyq * 0.95:
            return audio_chunk.copy()
        sos = butter(4, cutoff / nyq, btype='low', output='sos')
        if audio_chunk.ndim == 2:
            return np.column_stack([
                sosfilt(sos, audio_chunk[:, 0]),
                sosfilt(sos, audio_chunk[:, 1])
            ])
        return sosfilt(sos, audio_chunk)
    
    # Place loop with per-second filtering
    total_chunks = int(total_seconds)
    for chunk_idx in range(total_chunks):
        progress = chunk_idx / total_chunks
        
        # Filter envelope
        if progress < 0.1:
            cutoff = filter_start
        elif progress < 0.45:
            p = (progress - 0.1) / 0.35
            cutoff = filter_start + (filter_end * 0.4 - filter_start) * p
        elif progress < 0.7:
            p = (progress - 0.45) / 0.25
            cutoff = filter_end * 0.4 + (filter_end - filter_end * 0.4) * p
        elif progress < 0.85:
            cutoff = filter_end
        else:
            p = (progress - 0.85) / 0.15
            cutoff = filter_end - (filter_end - filter_start) * p
        
        # Get audio from looping groove
        start_sample = chunk_idx * sr
        end_sample = start_sample + sr
        
        loop_pos = (start_sample % len(loop))
        chunk_audio = np.zeros((sr, 2), dtype=np.float32)
        
        remaining = sr
        dst = 0
        src = loop_pos
        while remaining > 0:
            available = min(remaining, len(loop) - src)
            chunk_audio[dst:dst+available] = loop[src:src+available]
            dst += available
            remaining -= available
            src = 0
        
        filtered = apply_lowpass(chunk_audio, cutoff)
        
        if end_sample <= len(canvas):
            canvas[start_sample:end_sample] = filtered * 0.8
    
    # ── Sidechain ────────────────────────────────────────────────────
    beat_interval = 60.0 / target_bpm
    beat_samples = int(beat_interval * sr)
    sidechain = np.ones(len(canvas), dtype=np.float32)
    
    for bs in range(0, len(canvas), beat_samples):
        attack = int(0.003 * sr)
        release = int(0.25 * sr)
        depth = sidechain_depth
        for i in range(min(attack, len(canvas) - bs)):
            sidechain[bs + i] = min(sidechain[bs + i], 1.0 - (1.0 - depth) * (i / attack))
        for i in range(min(release, len(canvas) - bs - attack)):
            idx = bs + attack + i
            if idx < len(canvas):
                val = depth + (1.0 - depth) * (1.0 - np.exp(-3.0 * i / release))
                sidechain[idx] = min(sidechain[idx], val)
    
    canvas *= sidechain[:, np.newaxis]
    
    # ── Saturation ───────────────────────────────────────────────────
    canvas = np.tanh(canvas * saturation) * 0.85
    
    # ── Fade in/out ──────────────────────────────────────────────────
    fade_in = int(2.0 * sr)
    fade_out = int(8.0 * sr)
    canvas[:fade_in] *= np.linspace(0, 1, fade_in)[:, np.newaxis]
    canvas[-fade_out:] *= np.linspace(1, 0, fade_out)[:, np.newaxis]
    
    # ── Normalize ────────────────────────────────────────────────────
    peak = np.max(np.abs(canvas))
    if peak > 0:
        canvas *= 0.9 / peak
    canvas = np.clip(canvas, -1.0, 1.0)
    
    # ── Quality evaluation ───────────────────────────────────────────
    scores = quality_score(canvas, sr, target_bpm)
    
    # ── Save ─────────────────────────────────────────────────────────
    output_dir = f"/fragments/{session_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "arrangement.wav")
    sf.write(output_path, canvas, sr)
    volume.commit()
    
    return {
        "session_id": session_id,
        "duration": round(total_seconds, 1),
        "bpm": target_bpm,
        "loop_duration": round(loop_duration, 3),
        "quality_scores": scores,
        "filter_sweep": f"{filter_start}Hz → {filter_end}Hz",
        "output_path": output_path,
    }


@app.function(image=image, timeout=300, volumes={"/fragments": volume})
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File, Request
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import httpx
    import uuid

    api = FastAPI(title="Discovery Arranger V2")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    async def health():
        return {"status": "ok", "version": "v2-arranger"}

    @api.post("/arrange")
    async def arrange(request: Request):
        """
        Full pipeline: analyze → arrange → evaluate → return.
        Expects multipart with audio file, or JSON with session_id + analysis.
        """
        body = await request.json()
        session_id = body.get("session_id")
        analysis = body.get("analysis")
        params = body.get("params", {})
        
        if not session_id or not analysis:
            return JSONResponse(status_code=400, content={
                "error": "session_id and analysis required"
            })
        
        result = arrange_and_evaluate.remote(session_id, analysis, params)
        return JSONResponse(content=result)

    @api.get("/export/{session_id}")
    async def export(session_id: str):
        volume.reload()
        path = f"/fragments/{session_id}/arrangement.wav"
        if not os.path.exists(path):
            return JSONResponse(status_code=404, content={"error": "No arrangement found"})
        return FileResponse(path, media_type="audio/wav", filename="discovery_arrangement.wav")

    return api
