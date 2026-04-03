"""
Discovery V2 — Audio Analysis Pipeline
Deployed on Modal with Python 3.10 for madmom/essentia compatibility.

Endpoints:
  /analyze — Full audio analysis (beats, key, structure, segments)
  /health — Health check
"""
import modal
import json
import os
import io
import numpy as np

app = modal.App("discovery-analysis")

volume = modal.Volume.from_name("discovery-fragments", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "libfftw3-dev", "libyaml-dev", "git")
    .pip_install("cython", "numpy==1.23.5")
    .pip_install("madmom==0.16.1")
    .run_commands(
        # Fix madmom's Python 3.10+ compatibility: collections.MutableSequence moved to collections.abc
        "grep -rl 'from collections import MutableSequence' /usr/local/lib/python3.10/site-packages/madmom/ | xargs -r sed -i 's/from collections import MutableSequence/from collections.abc import MutableSequence/g'",
        "grep -rl 'collections.MutableSequence' /usr/local/lib/python3.10/site-packages/madmom/ | xargs -r sed -i 's/collections.MutableSequence/collections.abc.MutableSequence/g'",
        # Verify fix
        "python -c 'import madmom; print(\"madmom OK\")'",
    )
    .pip_install(
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "fastapi[standard]",
        "httpx>=0.27.0",
    )
)

# Essentia has its own binary wheels — try separately
image_with_essentia = image.pip_install("essentia>=2.1b6", force_build=False)


@app.function(image=image, timeout=120)
def analyze_track(audio_bytes: bytes, filename: str):
    """
    Full audio analysis using madmom + librosa.
    Returns: beats, downbeats, tempo, key, segments, energy profile.
    """
    import tempfile
    import subprocess
    import librosa
    import madmom
    from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

    # Write to temp file
    ext = os.path.splitext(filename)[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(audio_bytes)
    tmp.close()

    # Convert to wav if needed
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp.name, "-ar", "44100", "-ac", "1", wav_path],
        capture_output=True,
    )

    results = {}

    # ── 1. Beat tracking (madmom RNN — state of the art) ─────────────
    try:
        beat_proc = RNNBeatProcessor()
        beat_act = beat_proc(wav_path)
        beat_tracker = BeatTrackingProcessor(fps=100)
        beats = beat_tracker(beat_act)
        results["beats"] = [round(float(b), 4) for b in beats]
        
        if len(beats) > 1:
            intervals = np.diff(beats)
            results["tempo"] = round(60.0 / float(np.median(intervals)), 2)
            results["tempo_stability"] = round(1.0 - float(np.std(intervals) / np.median(intervals)), 4)
        else:
            results["tempo"] = 120.0
            results["tempo_stability"] = 0.0
    except Exception as e:
        results["beats"] = []
        results["tempo"] = 120.0
        results["tempo_stability"] = 0.0
        results["beat_error"] = str(e)

    # ── 2. Downbeat tracking (madmom RNN) ────────────────────────────
    try:
        db_proc = RNNDownBeatProcessor()
        db_act = db_proc(wav_path)
        db_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[4, 3], fps=100)
        downbeats_raw = db_tracker(db_act)
        # downbeats_raw is Nx2: [time, beat_position]
        results["downbeats"] = [
            {"time": round(float(row[0]), 4), "position": int(row[1])}
            for row in downbeats_raw
        ]
        # Extract just the bar starts (position == 1)
        results["bar_starts"] = [
            round(float(row[0]), 4)
            for row in downbeats_raw if int(row[1]) == 1
        ]
    except Exception as e:
        results["downbeats"] = []
        results["bar_starts"] = []
        results["downbeat_error"] = str(e)

    # ── 3. Key detection (madmom CNN) ────────────────────────────────
    try:
        key_proc = CNNKeyRecognitionProcessor()
        key_act = key_proc(wav_path)
        key_label = key_prediction_to_label(key_act)
        results["key"] = key_label
    except Exception as e:
        # Fallback to librosa
        try:
            y, sr = librosa.load(wav_path, sr=22050, mono=True)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            # Simple major/minor detection
            major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
            minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
            best_corr = -1
            best_key = "C major"
            for i in range(12):
                maj_corr = float(np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1])
                min_corr = float(np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1])
                if maj_corr > best_corr:
                    best_corr = maj_corr
                    best_key = f"{keys[i]} major"
                if min_corr > best_corr:
                    best_corr = min_corr
                    best_key = f"{keys[i]} minor"
            results["key"] = best_key
        except:
            results["key"] = "unknown"
        results["key_method"] = "librosa_fallback"

    # ── 4. Energy profile (for finding groovy sections) ──────────────
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        
        # RMS energy per beat
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        
        # Segment into 1-second chunks for energy profile
        chunk_frames = int(sr / 512)  # frames per second
        energy_profile = []
        brightness_profile = []
        for i in range(0, len(rms) - chunk_frames, chunk_frames):
            energy_profile.append(round(float(np.mean(rms[i:i+chunk_frames])), 6))
            brightness_profile.append(round(float(np.mean(centroid[i:i+chunk_frames])), 2))
        
        results["energy_profile"] = energy_profile
        results["brightness_profile"] = brightness_profile
        results["duration"] = round(float(librosa.get_duration(y=y, sr=sr)), 3)
        
        # Find the "grooviest" section — highest sustained energy
        if len(energy_profile) > 8:
            # Sliding window of 8 seconds, find highest average energy
            window = 8
            best_start = 0
            best_energy = 0
            for i in range(len(energy_profile) - window):
                avg_e = np.mean(energy_profile[i:i+window])
                if avg_e > best_energy:
                    best_energy = avg_e
                    best_start = i
            results["groove_section"] = {
                "start_sec": best_start,
                "end_sec": best_start + window,
                "energy": round(best_energy, 6),
            }
        
    except Exception as e:
        results["energy_error"] = str(e)

    # ── 5. Structure analysis (find intro/verse/chorus boundaries) ───
    try:
        # Use spectral contrast changes to find section boundaries
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        contrast_mean = contrast.mean(axis=0)
        
        # Find significant changes in spectral character
        diff = np.abs(np.diff(contrast_mean))
        threshold = np.mean(diff) + 2 * np.std(diff)
        boundaries_frames = np.where(diff > threshold)[0]
        boundary_times = librosa.frames_to_time(boundaries_frames, sr=sr, hop_length=512)
        
        # Filter to only keep boundaries > 4 seconds apart
        filtered = [boundary_times[0]] if len(boundary_times) > 0 else []
        for t in boundary_times[1:]:
            if t - filtered[-1] > 4.0:
                filtered.append(t)
        
        results["section_boundaries"] = [round(float(t), 3) for t in filtered]
    except Exception as e:
        results["structure_error"] = str(e)

    # Cleanup
    os.unlink(tmp.name)
    os.unlink(wav_path)

    return results


@app.function(image=image, timeout=300)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="Discovery Analysis V2")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    async def health():
        return {"status": "ok", "version": "v2-analysis"}

    @api.post("/analyze")
    async def analyze(file: UploadFile = File(...)):
        audio_bytes = await file.read()
        try:
            result = analyze_track.remote(audio_bytes, file.filename or "upload.mp3")
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return api
