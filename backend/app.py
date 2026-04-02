"""
Discovery — AI Micro-Sampling Workstation Backend
Modal app for audio decomposition, arrangement, and export.
"""
import modal
import json
import uuid
import os

app = modal.App("discovery-backend")

volume = modal.Volume.from_name("discovery-fragments", create_if_missing=True)

# GPU image for Demucs stem separation
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "demucs",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.1",
        "fastapi[standard]",
    )
)

# CPU image for analysis/arrangement/export
cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.1",
        "fastapi[standard]",
    )
)

# ── Helpers ──────────────────────────────────────────────────────────────────

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(y, sr):
    """Chroma-based key detection."""
    import librosa, numpy as np

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_idx = int(np.argmax(chroma_mean))
    # simple major/minor: compare major and minor profiles
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    # rotate profiles to each key and correlate
    best_corr = -1
    best_key = "C"
    best_mode = "major"
    for i in range(12):
        maj_corr = float(np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1])
        min_corr = float(np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1])
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = KEY_NAMES[i]
            best_mode = "major"
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = KEY_NAMES[i]
            best_mode = "minor"
    return f"{best_key} {best_mode}"


def analyze_audio(path):
    """Run librosa analysis on an audio file."""
    import librosa, numpy as np

    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    key = detect_key(y, sr)
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    duration = float(librosa.get_duration(y=y, sr=sr))
    tempo_val = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0]) if len(tempo) > 0 else 120.0

    return {
        "bpm": round(tempo_val, 1),
        "key": key,
        "beat_times": [round(t, 4) for t in beat_times],
        "onset_times": [round(t, 4) for t in onset_times],
        "energy": round(energy, 6),
        "duration": round(duration, 3),
    }


def slice_at_onsets(audio_path, onset_times, output_dir, stem_name, min_duration=0.05, max_duration=8.0):
    """Slice audio at onset times, return fragment metadata list."""
    import soundfile as sf
    import numpy as np

    y, sr = sf.read(audio_path)
    if len(y.shape) > 1:
        y_mono = y.mean(axis=1)
    else:
        y_mono = y

    total_duration = len(y) / sr
    # Add 0 and end
    cuts = [0.0] + [t for t in onset_times if 0 < t < total_duration] + [total_duration]
    cuts = sorted(set(cuts))

    fragments = []
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(cuts) - 1):
        start = cuts[i]
        end = cuts[i + 1]
        dur = end - start
        if dur < min_duration or dur > max_duration:
            continue

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]

        frag_id = str(uuid.uuid4())[:8]
        filename = f"{stem_name}_{i:04d}_{frag_id}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, chunk, sr)

        # energy of this chunk
        if len(chunk.shape) > 1:
            mono_chunk = chunk.mean(axis=1)
        else:
            mono_chunk = chunk
        energy = float(np.sqrt(np.mean(mono_chunk ** 2)))

        fragments.append({
            "id": frag_id,
            "filename": filename,
            "stem": stem_name,
            "onset_time": round(start, 4),
            "duration": round(dur, 4),
            "energy": round(energy, 6),
        })

    return fragments


STEM_CATEGORIES = {
    "vocals": "vocal",
    "drums": "percussion",
    "bass": "bass",
    "other": "chord",
}


# ── Decompose (GPU) ─────────────────────────────────────────────────────────

@app.function(
    image=gpu_image,
    gpu="A10G",
    timeout=600,
    volumes={"/fragments": volume},
)
def decompose_track(audio_bytes: bytes, filename: str, session_id: str):
    """Separate stems with Demucs, analyze, slice into fragments."""
    import subprocess
    import tempfile
    import shutil

    session_dir = f"/fragments/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    # Write uploaded file
    ext = os.path.splitext(filename)[1] or ".wav"
    input_path = f"/tmp/input{ext}"
    with open(input_path, "wb") as f:
        f.write(audio_bytes)

    # 1. Run Demucs (all 4 stems)
    demucs_out = "/tmp/demucs_out"
    subprocess.run(
        [
            "python", "-m", "demucs",
            "-n", "htdemucs_ft",
            "-o", demucs_out,
            input_path,
        ],
        check=True,
    )

    # Find stem files
    track_name = os.path.splitext(os.path.basename(input_path))[0]
    stems_dir = os.path.join(demucs_out, "htdemucs_ft", track_name)

    # 2. Analyze original
    original_analysis = analyze_audio(input_path)

    # 3. For each stem: analyze, slice, store
    manifest = {
        "session_id": session_id,
        "original_filename": filename,
        "original_analysis": original_analysis,
        "stems": {},
        "fragments": [],
    }

    for stem_name in ["vocals", "drums", "bass", "other"]:
        stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
        if not os.path.exists(stem_path):
            continue

        # Copy stem to volume
        stem_dest = os.path.join(session_dir, f"{stem_name}.wav")
        shutil.copy2(stem_path, stem_dest)

        # Analyze stem
        stem_analysis = analyze_audio(stem_path)
        manifest["stems"][stem_name] = stem_analysis

        # Slice at onsets
        frag_dir = os.path.join(session_dir, "fragments")
        frags = slice_at_onsets(
            stem_path,
            stem_analysis["onset_times"],
            frag_dir,
            stem_name,
        )

        category = STEM_CATEGORIES.get(stem_name, "texture")
        for frag in frags:
            frag["category"] = category
            # Detect key for longer fragments
            if frag["duration"] > 0.5:
                try:
                    frag_path = os.path.join(frag_dir, frag["filename"])
                    import librosa
                    y_f, sr_f = librosa.load(frag_path, sr=22050, mono=True)
                    frag["key"] = detect_key(y_f, sr_f)
                except Exception:
                    frag["key"] = original_analysis["key"]
            else:
                frag["key"] = original_analysis["key"]

        manifest["fragments"].extend(frags)

    # Identify texture fragments (low-energy "other" stem fragments)
    for frag in manifest["fragments"]:
        if frag["stem"] == "other" and frag["energy"] < 0.01 and frag["duration"] > 1.0:
            frag["category"] = "texture"

    # Save manifest
    manifest_path = os.path.join(session_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    volume.commit()
    return manifest


# ── Web endpoints (CPU) ─────────────────────────────────────────────────────

@app.function(image=cpu_image, volumes={"/fragments": volume}, timeout=300)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File, Request
    from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="Discovery Backend")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.post("/decompose")
    async def decompose(file: UploadFile = File(...)):
        session_id = str(uuid.uuid4())
        audio_bytes = await file.read()
        # Call GPU function
        manifest = decompose_track.remote(audio_bytes, file.filename or "upload.wav", session_id)

        # Add download URLs
        base_url = f"/fragments/{session_id}"
        for frag in manifest.get("fragments", []):
            frag["url"] = f"{base_url}/file/{frag['filename']}"
        for stem_name in manifest.get("stems", {}):
            manifest["stems"][stem_name]["url"] = f"{base_url}/stem/{stem_name}"

        return JSONResponse(content=manifest)

    @api.get("/fragments/{session_id}")
    async def get_fragments(session_id: str):
        volume.reload()
        manifest_path = f"/fragments/{session_id}/manifest.json"
        if not os.path.exists(manifest_path):
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        with open(manifest_path) as f:
            manifest = json.load(f)
        base_url = f"/fragments/{session_id}"
        for frag in manifest.get("fragments", []):
            frag["url"] = f"{base_url}/file/{frag['filename']}"
        return JSONResponse(content=manifest)

    @api.get("/fragments/{session_id}/file/{filename}")
    async def get_fragment_file(session_id: str, filename: str):
        volume.reload()
        filepath = f"/fragments/{session_id}/fragments/{filename}"
        if not os.path.exists(filepath):
            return JSONResponse(status_code=404, content={"error": "Fragment not found"})
        return FileResponse(filepath, media_type="audio/wav", filename=filename)

    @api.get("/fragments/{session_id}/stem/{stem_name}")
    async def get_stem_file(session_id: str, stem_name: str):
        volume.reload()
        filepath = f"/fragments/{session_id}/{stem_name}.wav"
        if not os.path.exists(filepath):
            return JSONResponse(status_code=404, content={"error": "Stem not found"})
        return FileResponse(filepath, media_type="audio/wav", filename=f"{stem_name}.wav")

    @api.post("/arrange")
    async def arrange(request: Request):
        body = await request.json()
        fragment_ids = body.get("fragment_ids", [])
        params = body.get("params", {})

        # Load session
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse(status_code=400, content={"error": "session_id required"})

        volume.reload()
        manifest_path = f"/fragments/{session_id}/manifest.json"
        if not os.path.exists(manifest_path):
            return JSONResponse(status_code=404, content={"error": "Session not found"})

        with open(manifest_path) as f:
            manifest = json.load(f)

        all_frags = {f["id"]: f for f in manifest.get("fragments", [])}

        # Filter to requested fragments (or use all)
        if fragment_ids:
            selected = [all_frags[fid] for fid in fragment_ids if fid in all_frags]
        else:
            selected = list(all_frags.values())

        if not selected:
            return JSONResponse(status_code=400, content={"error": "No valid fragments"})

        # Vibe parameters with defaults
        choppiness = params.get("choppiness", 0.5)
        density = params.get("density", 0.5)
        tempo = params.get("tempo", manifest["original_analysis"]["bpm"])
        swing = params.get("swing", 0.0)
        groove = params.get("groove", 0.7)

        # Algorithmic arrangement (v1)
        import random
        random.seed(hash(json.dumps(params, sort_keys=True)) % (2**32))

        beat_duration = 60.0 / tempo
        bar_duration = beat_duration * 4

        # Sort by category for track assignment
        tracks = {"percussion": [], "bass": [], "chord": [], "vocal": [], "texture": []}
        for f in selected:
            cat = f.get("category", "chord")
            if cat in tracks:
                tracks[cat].append(f)

        arrangement = []
        # For each track, place fragments along timeline
        total_bars = int(8 + density * 16)  # 8-24 bars based on density
        timeline_duration = total_bars * bar_duration

        for track_name, track_frags in tracks.items():
            if not track_frags:
                continue

            # Density controls how many slots to fill
            num_slots = max(1, int(total_bars * density * (2 if track_name == "percussion" else 1)))

            # Choppiness controls whether we use shorter fragments
            if choppiness > 0.5:
                track_frags.sort(key=lambda f: f["duration"])
            else:
                track_frags.sort(key=lambda f: -f["duration"])

            for slot in range(num_slots):
                # Quantize to beat grid
                beat_pos = slot * beat_duration * (4 / max(density, 0.1))
                if beat_pos >= timeline_duration:
                    break

                # Apply swing
                if slot % 2 == 1 and swing > 0:
                    beat_pos += beat_duration * swing * 0.3

                frag = random.choice(track_frags[:max(3, int(len(track_frags) * groove))])
                arrangement.append({
                    "fragment_id": frag["id"],
                    "track": track_name,
                    "start_time": round(beat_pos, 4),
                    "duration": round(min(frag["duration"], bar_duration), 4),
                    "filename": frag["filename"],
                })

        arrangement.sort(key=lambda x: (x["track"], x["start_time"]))

        # Score
        score = {
            "groove_alignment": round(groove * 0.8 + 0.2, 2),
            "harmonic_consistency": round(0.5 + random.random() * 0.4, 2),
            "motif_recurrence": round(min(1.0, len(arrangement) / (total_bars * 2)), 2),
            "total_bars": total_bars,
            "tempo": tempo,
        }

        return JSONResponse(content={
            "session_id": session_id,
            "arrangement": arrangement,
            "score": score,
        })

    @api.post("/export")
    async def export(request: Request):
        body = await request.json()
        session_id = body.get("session_id")
        arrangement = body.get("arrangement", [])
        output_format = body.get("format", "wav")

        if not session_id or not arrangement:
            return JSONResponse(status_code=400, content={"error": "session_id and arrangement required"})

        volume.reload()

        from pydub import AudioSegment
        import io

        # Determine total duration
        max_end = max(item["start_time"] + item["duration"] for item in arrangement)
        total_ms = int((max_end + 1) * 1000)

        # Create empty mix
        mix = AudioSegment.silent(duration=total_ms)

        frag_dir = f"/fragments/{session_id}/fragments"

        for item in arrangement:
            filepath = os.path.join(frag_dir, item["filename"])
            if not os.path.exists(filepath):
                continue
            try:
                segment = AudioSegment.from_wav(filepath)
                pos_ms = int(item["start_time"] * 1000)
                mix = mix.overlay(segment, position=pos_ms)
            except Exception:
                continue

        # Export
        buf = io.BytesIO()
        if output_format == "mp3":
            mix.export(buf, format="mp3", bitrate="320k")
            media_type = "audio/mpeg"
            ext = "mp3"
        else:
            mix.export(buf, format="wav")
            media_type = "audio/wav"
            ext = "wav"

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=discovery_export.{ext}"},
        )

    return api
