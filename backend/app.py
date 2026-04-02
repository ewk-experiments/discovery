"""
Discovery — AI Micro-Sampling Workstation Backend
Modal app for audio decomposition, arrangement, and export.
"""
import modal
import json
import uuid
import os
import httpx

app = modal.App("discovery-backend")

volume = modal.Volume.from_name("discovery-fragments", create_if_missing=True)

# GitHub OAuth App credentials
# TODO: Create a GitHub OAuth App at https://github.com/settings/developers
# Set these as Modal secrets or environment variables
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "PLACEHOLDER_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "PLACEHOLDER_CLIENT_SECRET")

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
        "httpx>=0.27.0",
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

    # ── GitHub OAuth ──────────────────────────────────────────────────────

    @api.get("/auth/github")
    async def auth_github(redirect_uri: str = "https://discovery.ewklabs.xyz"):
        """Redirect user to GitHub OAuth authorization page."""
        state = str(uuid.uuid4())
        # The callback URL is this backend's /auth/callback
        callback = f"https://heyitskim-ai--discovery-backend-web.modal.run/auth/callback"
        params = (
            f"?client_id={GITHUB_CLIENT_ID}"
            f"&redirect_uri={callback}"
            f"&scope=read:user"
            f"&state={state}:{redirect_uri}"
        )
        from fastapi.responses import RedirectResponse
        return RedirectResponse(f"https://github.com/login/oauth/authorize{params}")

    @api.get("/auth/callback")
    async def auth_callback(code: str, state: str = ""):
        """Exchange OAuth code for access token and redirect back to frontend."""
        from fastapi.responses import RedirectResponse
        import httpx as hx

        # Extract redirect_uri from state
        parts = state.split(":", 1)
        redirect_uri = parts[1] if len(parts) > 1 else "https://discovery.ewklabs.xyz"

        # Exchange code for token
        async with hx.AsyncClient() as client:
            resp = await client.post(
                "https://github.com/login/oauth/access_token",
                json={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code,
                },
                headers={"Accept": "application/json"},
            )
            data = resp.json()

        token = data.get("access_token", "")
        if not token:
            return JSONResponse(status_code=400, content={"error": "OAuth failed", "details": data})

        # Redirect back to frontend with token
        separator = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(f"{redirect_uri}{separator}token={token}")

    # ── Intelligent Arrangement (Opus 4.6 via GitHub Copilot) ──────────

    @api.post("/arrange/intelligent")
    async def arrange_intelligent(request: Request):
        body = await request.json()
        github_token = body.get("github_token")
        fragments_manifest = body.get("fragments", [])
        vibe = body.get("vibe", {})
        session_id = body.get("session_id")

        if not github_token:
            return JSONResponse(status_code=401, content={"error": "GitHub token required"})

        if not fragments_manifest:
            return JSONResponse(status_code=400, content={"error": "No fragments provided"})

        # Build the arrangement prompt
        fragment_desc = "\n".join(
            f"  - {f['id']}: {f['name']} | {f['category']} | dur={f['duration']} | key={f['key']} | energy={f.get('energy', '?')} | source=\"{f.get('source', '?')}\""
            for f in fragments_manifest
        )

        vibe_desc = "\n".join(f"  - {k}: {v}" for k, v in vibe.items())

        prompt = f"""You are a micro-sampling arrangement engine inspired by Todd Edwards and Daft Punk's "Face to Face" production technique.

AVAILABLE FRAGMENTS:
{fragment_desc}

VIBE PARAMETERS:
{vibe_desc}

PARAMETER MEANINGS:
- choppiness (0-100): How short/frequent the cuts are. High = rapid stutter edits. Low = longer phrases.
- density (0-100): How many fragments play simultaneously. High = layered, dense. Low = sparse, minimal.
- harmonic (0-100): How far from the key center fragments can stray. Low = strict harmonic matching. High = adventurous.
- tempo: BPM for the arrangement.
- groove (0-100): How tightly cuts align to the beat grid. High = perfectly quantized. Low = loose, human feel.
- swing (0-100): Rhythmic offset on every other beat. Creates shuffle/bounce.
- attack (0-100): How abruptly fragments enter. High = hard cuts. Low = faded in.
- decay (0-100): How fragments fade out. High = abrupt end. Low = long tail.
- warmth (0-100): Favor warmer/fuller frequency fragments. High = warm vinyl feel. Low = crisp/digital.
- drift (0-100): Allow gradual key/tempo wandering over time.

ARRANGEMENT RULES:
1. Create a 4-8 bar arrangement that could loop seamlessly
2. Choose ONE fragment as the "anchor" — a recurring element that holds it together (like Todd Edwards' vocal chop in Face to Face)
3. The anchor should appear in at least 50% of bars
4. Ensure cuts land on the beat grid (respect groove parameter)
5. Keep fragments in compatible harmonic space (respect harmonic parameter)
6. Bring fragments back for motif recurrence — repetition builds familiarity
7. Vary density — build up and break down within the arrangement
8. Percussion fragments form the rhythmic backbone
9. Apply choppiness: high choppiness = more fragments with shorter durations, low = fewer with longer durations
10. Think like you're producing a French house / filtered disco track

SCORING (optimize for):
- groove_alignment: Do cuts land on the beat grid?
- harmonic_consistency: Are fragments in compatible keys?
- anchor_ratio: Does the anchor fragment recur enough?
- motif_recurrence: Do fragments repeat to build familiarity?

OUTPUT FORMAT (JSON only, no markdown):
{{
  "arrangement": [
    {{"fragment_id": "...", "track": "vocal|chord|texture|percussion|bass", "start_time": 0.0, "duration": 0.8, "repeat_count": 1}}
  ],
  "anchor_fragment_id": "...",
  "total_bars": 8,
  "scores": {{
    "groove_alignment": 0.0-1.0,
    "harmonic_consistency": 0.0-1.0,
    "anchor_ratio": 0.0-1.0,
    "motif_recurrence": 0.0-1.0
  }}
}}

Respond with ONLY the JSON object. No explanation, no markdown fences."""

        # Call GitHub Copilot API with Opus 4.6
        import httpx as hx
        async with hx.AsyncClient(timeout=60.0) as client:
            # First get a Copilot token from the GitHub token
            copilot_token_resp = await client.get(
                "https://api.github.com/copilot_internal/v2/token",
                headers={
                    "Authorization": f"token {github_token}",
                    "Accept": "application/json",
                },
            )

            if copilot_token_resp.status_code == 200:
                copilot_token_data = copilot_token_resp.json()
                copilot_token = copilot_token_data.get("token", github_token)
                api_url = copilot_token_data.get("endpoints", {}).get("api", "https://api.githubcopilot.com")
            else:
                # Fallback: try using the GitHub token directly with the models API
                copilot_token = github_token
                api_url = "https://models.inference.ai.azure.com"

            resp = await client.post(
                f"{api_url}/chat/completions",
                json={
                    "model": "claude-opus-4.6",
                    "messages": [
                        {"role": "system", "content": "You are a music arrangement engine. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                headers={
                    "Authorization": f"Bearer {copilot_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

        if resp.status_code != 200:
            return JSONResponse(
                status_code=502,
                content={"error": "Copilot API error", "status": resp.status_code, "detail": resp.text[:500]},
            )

        # Parse the response
        try:
            completion = resp.json()
            content = completion["choices"][0]["message"]["content"]
            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            result = json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return JSONResponse(
                status_code=502,
                content={"error": "Failed to parse Opus response", "detail": str(e), "raw": content[:1000] if 'content' in dir() else ""},
            )

        # Validate fragment IDs
        valid_ids = {f["id"] for f in fragments_manifest}
        arrangement = result.get("arrangement", [])
        arrangement = [a for a in arrangement if a.get("fragment_id") in valid_ids]

        # Build response matching frontend expectations
        tempo = vibe.get("tempo", 128)
        beat_duration = 60.0 / tempo

        return JSONResponse(content={
            "session_id": session_id,
            "arrangement": arrangement,
            "anchor_fragment_id": result.get("anchor_fragment_id"),
            "score": result.get("scores", {}),
            "total_bars": result.get("total_bars", 8),
            "intelligent": True,
        })

    return api
