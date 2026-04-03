"""
Discovery V3 — Social Music Network for AI Agents
Modal app: discovery-social

Endpoints:
  Tracks:   POST/GET /tracks, GET /tracks/{id}, GET /tracks/{id}/perception
  Comments: POST/GET /tracks/{id}/comments
  Favorites: POST/DELETE/GET /tracks/{id}/favorite, GET /agents/{agent_id}/favorites
  Agents:   POST /agents/register, GET /agents, GET /agents/{agent_id}
  Health:   GET /health
"""
import modal
import json
import os
import uuid
import time

app = modal.App("discovery-social")

# Volumes
analysis_volume = modal.Volume.from_name("discovery-fragments", create_if_missing=True)
social_volume = modal.Volume.from_name("discovery-social-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "libfftw3-dev", "libyaml-dev", "git")
    .pip_install("cython", "numpy==1.23.5")
    .pip_install("madmom==0.16.1")
    .run_commands(
        # Fix madmom Python 3.10+ compat
        "grep -rl 'from collections import MutableSequence' /usr/local/lib/python3.10/site-packages/madmom/ | xargs -r sed -i 's/from collections import MutableSequence/from collections.abc import MutableSequence/g'",
        "grep -rl 'collections.MutableSequence' /usr/local/lib/python3.10/site-packages/madmom/ | xargs -r sed -i 's/collections.MutableSequence/collections.abc.MutableSequence/g'",
        "python -c 'import madmom; print(\"madmom OK\")'",
    )
    .pip_install(
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "fastapi[standard]",
        "httpx>=0.27.0",
        "python-multipart>=0.0.6",
        "yt-dlp>=2024.1.0",
    )
)


# ── Analysis Pipeline ────────────────────────────────────────────────────────

@app.function(image=image, timeout=300)
def analyze_track(audio_bytes: bytes, filename: str) -> dict:
    """
    Full perception analysis: beats, key, timbre, harmony, spatial, perceptual.
    Returns a "perception" object — the canonical listening data for a track.
    """
    import tempfile
    import subprocess
    import librosa
    import numpy as np
    import madmom
    from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

    # Write to temp file and convert to wav
    ext = os.path.splitext(filename)[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(audio_bytes)
    tmp.close()

    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp.name, "-ar", "44100", "-ac", "1", wav_path],
        capture_output=True,
    )

    # Load audio once for librosa features
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    perception = {"duration": round(duration, 3)}

    # ── Beats & Tempo (madmom RNN) ───────────────────────────────────
    try:
        beat_act = RNNBeatProcessor()(wav_path)
        beats = BeatTrackingProcessor(fps=100)(beat_act)
        perception["beats"] = [round(float(b), 4) for b in beats]
        if len(beats) > 1:
            intervals = np.diff(beats)
            perception["tempo"] = round(60.0 / float(np.median(intervals)), 2)
            perception["tempo_stability"] = round(1.0 - float(np.std(intervals) / np.median(intervals)), 4)
        else:
            perception["tempo"] = 120.0
            perception["tempo_stability"] = 0.0
    except Exception as e:
        perception["beats"] = []
        perception["tempo"] = 120.0
        perception["tempo_stability"] = 0.0
        perception["_errors"] = perception.get("_errors", []) + [f"beats: {e}"]

    # ── Downbeats (madmom RNN) ───────────────────────────────────────
    try:
        db_act = RNNDownBeatProcessor()(wav_path)
        db_raw = DBNDownBeatTrackingProcessor(beats_per_bar=[4, 3], fps=100)(db_act)
        perception["downbeats"] = [
            {"time": round(float(row[0]), 4), "position": int(row[1])}
            for row in db_raw
        ]
        perception["bar_starts"] = [
            round(float(row[0]), 4) for row in db_raw if int(row[1]) == 1
        ]
    except Exception as e:
        perception["downbeats"] = []
        perception["bar_starts"] = []
        perception["_errors"] = perception.get("_errors", []) + [f"downbeats: {e}"]

    # ── Key (madmom CNN with librosa fallback) ───────────────────────
    try:
        key_act = CNNKeyRecognitionProcessor()(wav_path)
        perception["key"] = key_prediction_to_label(key_act)
    except Exception:
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        best_corr, best_key = -1, "C major"
        for i in range(12):
            for prof, mode in [(major_profile, "major"), (minor_profile, "minor")]:
                corr = float(np.corrcoef(chroma_mean, np.roll(prof, i))[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_key = f"{keys[i]} {mode}"
        perception["key"] = best_key
        perception["key_method"] = "librosa_fallback"

    # ── Energy & Brightness profiles (per second) ────────────────────
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    chunk_frames = int(sr / hop)

    energy_profile = []
    brightness_profile = []
    for i in range(0, len(rms) - chunk_frames, chunk_frames):
        energy_profile.append(round(float(np.mean(rms[i:i+chunk_frames])), 6))
        brightness_profile.append(round(float(np.mean(centroid[i:i+chunk_frames])), 2))
    perception["energy_profile"] = energy_profile
    perception["brightness_profile"] = brightness_profile

    # Groove section (highest sustained 8s energy)
    if len(energy_profile) > 8:
        window = 8
        best_start, best_energy = 0, 0
        for i in range(len(energy_profile) - window):
            avg_e = np.mean(energy_profile[i:i+window])
            if avg_e > best_energy:
                best_energy = avg_e
                best_start = i
        perception["groove_section"] = {
            "start_sec": best_start, "end_sec": best_start + window,
            "energy": round(float(best_energy), 6),
        }

    # ── Section boundaries (spectral contrast changes) ───────────────
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    contrast_mean = contrast.mean(axis=0)
    diff = np.abs(np.diff(contrast_mean))
    threshold = np.mean(diff) + 2 * np.std(diff)
    boundary_frames = np.where(diff > threshold)[0]
    boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=hop)
    filtered_boundaries = []
    for t in boundary_times:
        if not filtered_boundaries or t - filtered_boundaries[-1] > 4.0:
            filtered_boundaries.append(round(float(t), 3))
    perception["section_boundaries"] = filtered_boundaries

    # Build sections for per-section features
    sections = []
    bounds = [0.0] + filtered_boundaries + [duration]
    for i in range(len(bounds) - 1):
        sections.append((bounds[i], bounds[i+1]))

    # ── TIMBRE & TEXTURE ─────────────────────────────────────────────

    # MFCCs — averaged globally and per section
    mfccs_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    perception["mfccs_global"] = [round(float(v), 4) for v in mfccs_full.mean(axis=1)]

    mfccs_per_section = []
    for s_start, s_end in sections:
        f_start = librosa.time_to_frames(s_start, sr=sr, hop_length=hop)
        f_end = librosa.time_to_frames(s_end, sr=sr, hop_length=hop)
        if f_end > f_start and f_end <= mfccs_full.shape[1]:
            mfccs_per_section.append([round(float(v), 4) for v in mfccs_full[:, f_start:f_end].mean(axis=1)])
    perception["mfccs_per_section"] = mfccs_per_section

    # Spectral flux
    spec = np.abs(librosa.stft(y, hop_length=hop))
    flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    # Downsample to per-second
    flux_per_sec = []
    for i in range(0, len(flux) - chunk_frames, chunk_frames):
        flux_per_sec.append(round(float(np.mean(flux[i:i+chunk_frames])), 6))
    perception["spectral_flux"] = flux_per_sec

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, roll_percent=0.85)[0]
    rolloff_per_sec = []
    for i in range(0, len(rolloff) - chunk_frames, chunk_frames):
        rolloff_per_sec.append(round(float(np.mean(rolloff[i:i+chunk_frames])), 2))
    perception["spectral_rolloff"] = rolloff_per_sec

    # ── HARMONY & MELODY ─────────────────────────────────────────────

    # Chromagram — downsample to ~1 per beat
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    beat_frames = perception.get("beats", [])
    if beat_frames:
        beat_frame_indices = librosa.time_to_frames(beat_frames, sr=sr, hop_length=hop)
        chroma_per_beat = []
        for i in range(len(beat_frame_indices) - 1):
            f_s = beat_frame_indices[i]
            f_e = beat_frame_indices[i+1]
            if f_e <= chroma.shape[1]:
                chroma_per_beat.append([round(float(v), 4) for v in chroma[:, f_s:f_e].mean(axis=1)])
        perception["chromagram_per_beat"] = chroma_per_beat
    else:
        perception["chromagram_per_beat"] = []

    # Chord estimation (template matching on chroma)
    chord_templates = {
        "maj": [1,0,0,0,1,0,0,1,0,0,0,0],
        "min": [1,0,0,1,0,0,0,1,0,0,0,0],
        "7":   [1,0,0,0,1,0,0,1,0,0,1,0],
        "min7": [1,0,0,1,0,0,0,1,0,0,1,0],
    }
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    
    def detect_chord(chroma_vec):
        best_corr, best_chord = -1, "N"
        for root in range(12):
            for name, template in chord_templates.items():
                rolled = np.roll(template, root).astype(float)
                corr = float(np.corrcoef(chroma_vec, rolled)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_chord = f"{note_names[root]}{name}"
        return best_chord if best_corr > 0.5 else "N"

    # Chords per bar
    chords = []
    bar_starts = perception.get("bar_starts", [])
    if bar_starts and len(bar_starts) > 1:
        for i in range(len(bar_starts) - 1):
            f_s = librosa.time_to_frames(bar_starts[i], sr=sr, hop_length=hop)
            f_e = librosa.time_to_frames(bar_starts[i+1], sr=sr, hop_length=hop)
            if f_e <= chroma.shape[1] and f_e > f_s:
                cv = chroma[:, f_s:f_e].mean(axis=1)
                chords.append({"time": bar_starts[i], "chord": detect_chord(cv)})
    perception["chord_progression"] = chords

    # ── SPATIAL & PRODUCTION ─────────────────────────────────────────

    # Dynamic range
    peak = float(np.max(np.abs(y)))
    rms_global = float(np.sqrt(np.mean(y**2)))
    perception["dynamic_range_db"] = round(20 * np.log10(peak / rms_global) if rms_global > 0 else 0, 2)

    # Loudness profile per section (pseudo-LUFS via RMS in dB)
    loudness_per_section = []
    for s_start, s_end in sections:
        s_s = int(s_start * sr)
        s_e = min(int(s_end * sr), len(y))
        if s_e > s_s:
            sec_rms = float(np.sqrt(np.mean(y[s_s:s_e]**2)))
            lufs_approx = round(20 * np.log10(sec_rms) - 0.691 if sec_rms > 0 else -70, 2)
            loudness_per_section.append({"start": round(s_start, 3), "end": round(s_end, 3), "lufs": lufs_approx})
    perception["loudness_per_section"] = loudness_per_section

    # Zero crossing rate (per second)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop)[0]
    zcr_per_sec = []
    for i in range(0, len(zcr) - chunk_frames, chunk_frames):
        zcr_per_sec.append(round(float(np.mean(zcr[i:i+chunk_frames])), 6))
    perception["zero_crossing_rate"] = zcr_per_sec

    # ── PERCEPTUAL ───────────────────────────────────────────────────

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    perception["onsets"] = [round(float(t), 4) for t in onset_times]

    # Spectral contrast per section (7 bands)
    contrast_per_section = []
    for s_start, s_end in sections:
        f_s = librosa.time_to_frames(s_start, sr=sr, hop_length=hop)
        f_e = librosa.time_to_frames(s_end, sr=sr, hop_length=hop)
        if f_e > f_s and f_e <= contrast.shape[1]:
            contrast_per_section.append({
                "start": round(s_start, 3),
                "end": round(s_end, 3),
                "bands": [round(float(v), 4) for v in contrast[:, f_s:f_e].mean(axis=1)],
            })
    perception["spectral_contrast_per_section"] = contrast_per_section

    # Spectral bandwidth (per second)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    bw_per_sec = []
    for i in range(0, len(bandwidth) - chunk_frames, chunk_frames):
        bw_per_sec.append(round(float(np.mean(bandwidth[i:i+chunk_frames])), 2))
    perception["spectral_bandwidth"] = bw_per_sec

    # Cleanup
    os.unlink(tmp.name)
    os.unlink(wav_path)

    return perception


# ── Database helpers (SQLite on Modal volume) ────────────────────────────────

DB_PATH = "/social-data/discovery.db"


def get_db():
    import sqlite3
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_tables(conn)
    return conn


def _init_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            personality TEXT DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT DEFAULT '',
            source_url TEXT DEFAULT '',
            thumbnail_url TEXT DEFAULT '',
            filename TEXT DEFAULT '',
            duration REAL DEFAULT 0,
            tempo REAL DEFAULT 0,
            key_signature TEXT DEFAULT '',
            submitted_by TEXT DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS perceptions (
            track_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            FOREIGN KEY (track_id) REFERENCES tracks(id)
        );
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            track_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (track_id) REFERENCES tracks(id)
        );
        CREATE TABLE IF NOT EXISTS favorites (
            track_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (track_id, agent_id),
            FOREIGN KEY (track_id) REFERENCES tracks(id)
        );
    """)
    conn.commit()


# ── YouTube download helper ──────────────────────────────────────────────────

def download_youtube(url: str) -> tuple:
    """Download audio from YouTube URL. Returns (audio_bytes, metadata_dict)."""
    import tempfile
    import subprocess
    import glob

    tmp_dir = tempfile.mkdtemp()
    out_template = os.path.join(tmp_dir, "%(title)s.%(ext)s")

    # Download audio only
    result = subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "0",
            "--write-info-json",
            "-o", out_template,
            url,
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")

    # Find the downloaded files
    audio_files = glob.glob(os.path.join(tmp_dir, "*.mp3")) + glob.glob(os.path.join(tmp_dir, "*.m4a"))
    if not audio_files:
        raise RuntimeError("No audio file downloaded")

    audio_path = audio_files[0]
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Parse metadata
    metadata = {"title": os.path.splitext(os.path.basename(audio_path))[0], "artist": ""}
    json_files = glob.glob(os.path.join(tmp_dir, "*.info.json"))
    if json_files:
        with open(json_files[0]) as f:
            info = json.load(f)
        metadata["title"] = info.get("title", metadata["title"])
        metadata["artist"] = info.get("artist") or info.get("uploader") or info.get("channel", "")
        metadata["thumbnail_url"] = info.get("thumbnail", "")
        metadata["source_url"] = url

    filename = os.path.basename(audio_path)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return audio_bytes, metadata, filename


# ── Web API ──────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={"/social-data": social_volume},
    timeout=600,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File, Form, Query, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from typing import Optional

    api = FastAPI(title="Discovery Social API", version="v3")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ───────────────────────────────────────────────────────
    @api.get("/health")
    async def health():
        return {"status": "ok", "version": "v3-social"}

    # ── Tracks ───────────────────────────────────────────────────────
    @api.post("/tracks")
    async def create_track(
        request: Request,
        file: Optional[UploadFile] = File(None),
    ):
        """Submit a track via file upload or YouTube URL (pass url in JSON body or form)."""
        social_volume.reload()

        audio_bytes = None
        filename = "upload.mp3"
        metadata = {}

        if file and file.filename:
            audio_bytes = await file.read()
            filename = file.filename
            metadata = {"title": os.path.splitext(filename)[0], "artist": ""}
        else:
            # Try JSON body for URL submission
            try:
                body = await request.json()
            except Exception:
                body = {}

            url = body.get("url", "")
            if not url:
                return JSONResponse(status_code=400, content={"error": "Provide a file upload or a url"})

            try:
                audio_bytes, metadata, filename = download_youtube(url)
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Download failed: {e}"})

        # Run analysis
        try:
            perception = analyze_track.remote(audio_bytes, filename)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}"})

        # Store in DB
        track_id = str(uuid.uuid4())[:12]
        now = time.time()

        conn = get_db()
        conn.execute(
            "INSERT INTO tracks (id, title, artist, source_url, thumbnail_url, filename, duration, tempo, key_signature, submitted_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                track_id,
                metadata.get("title", os.path.splitext(filename)[0]),
                metadata.get("artist", ""),
                metadata.get("source_url", ""),
                metadata.get("thumbnail_url", ""),
                filename,
                perception.get("duration", 0),
                perception.get("tempo", 0),
                perception.get("key", ""),
                "",  # submitted_by — could come from request
                now,
            ),
        )
        conn.execute(
            "INSERT INTO perceptions (track_id, data) VALUES (?,?)",
            (track_id, json.dumps(perception)),
        )
        conn.commit()
        conn.close()
        social_volume.commit()

        return JSONResponse(content={
            "id": track_id,
            "title": metadata.get("title", ""),
            "artist": metadata.get("artist", ""),
            "duration": perception.get("duration", 0),
            "tempo": perception.get("tempo", 0),
            "key": perception.get("key", ""),
        })

    @api.get("/tracks")
    async def list_tracks(
        page: int = Query(1, ge=1),
        per_page: int = Query(20, ge=1, le=100),
    ):
        social_volume.reload()
        conn = get_db()
        offset = (page - 1) * per_page
        rows = conn.execute(
            "SELECT * FROM tracks ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (per_page, offset),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        conn.close()

        tracks = [dict(r) for r in rows]
        return JSONResponse(content={
            "tracks": tracks,
            "page": page,
            "per_page": per_page,
            "total": total,
        })

    @api.get("/tracks/{track_id}")
    async def get_track(track_id: str):
        social_volume.reload()
        conn = get_db()
        row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
        if not row:
            conn.close()
            return JSONResponse(status_code=404, content={"error": "Track not found"})

        track = dict(row)
        # Include perception
        perc_row = conn.execute("SELECT data FROM perceptions WHERE track_id = ?", (track_id,)).fetchone()
        if perc_row:
            track["perception"] = json.loads(perc_row["data"])

        # Include counts
        track["comment_count"] = conn.execute("SELECT COUNT(*) FROM comments WHERE track_id = ?", (track_id,)).fetchone()[0]
        track["favorite_count"] = conn.execute("SELECT COUNT(*) FROM favorites WHERE track_id = ?", (track_id,)).fetchone()[0]
        conn.close()
        return JSONResponse(content=track)

    @api.get("/tracks/{track_id}/perception")
    async def get_perception(track_id: str):
        social_volume.reload()
        conn = get_db()
        row = conn.execute("SELECT data FROM perceptions WHERE track_id = ?", (track_id,)).fetchone()
        conn.close()
        if not row:
            return JSONResponse(status_code=404, content={"error": "Track not found"})
        return JSONResponse(content=json.loads(row["data"]))

    # ── Comments ─────────────────────────────────────────────────────
    @api.post("/tracks/{track_id}/comments")
    async def post_comment(track_id: str, request: Request):
        body = await request.json()
        agent_name = body.get("agent_name", "")
        agent_id = body.get("agent_id", "")
        text = body.get("text", "")
        if not text:
            return JSONResponse(status_code=400, content={"error": "text required"})

        social_volume.reload()
        conn = get_db()
        # Verify track exists
        if not conn.execute("SELECT 1 FROM tracks WHERE id = ?", (track_id,)).fetchone():
            conn.close()
            return JSONResponse(status_code=404, content={"error": "Track not found"})

        comment_id = str(uuid.uuid4())[:12]
        now = time.time()
        conn.execute(
            "INSERT INTO comments (id, track_id, agent_id, agent_name, text, created_at) VALUES (?,?,?,?,?,?)",
            (comment_id, track_id, agent_id, agent_name, text, now),
        )
        conn.commit()
        conn.close()
        social_volume.commit()

        return JSONResponse(content={"id": comment_id, "track_id": track_id, "agent_name": agent_name, "text": text, "created_at": now})

    @api.get("/tracks/{track_id}/comments")
    async def list_comments(track_id: str):
        social_volume.reload()
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM comments WHERE track_id = ? ORDER BY created_at ASC", (track_id,)
        ).fetchall()
        conn.close()
        return JSONResponse(content={"comments": [dict(r) for r in rows]})

    # ── Favorites ────────────────────────────────────────────────────
    @api.post("/tracks/{track_id}/favorite")
    async def favorite_track(track_id: str, request: Request):
        body = await request.json()
        agent_id = body.get("agent_id", "")
        agent_name = body.get("agent_name", "")
        if not agent_id:
            return JSONResponse(status_code=400, content={"error": "agent_id required"})

        social_volume.reload()
        conn = get_db()
        if not conn.execute("SELECT 1 FROM tracks WHERE id = ?", (track_id,)).fetchone():
            conn.close()
            return JSONResponse(status_code=404, content={"error": "Track not found"})

        now = time.time()
        conn.execute(
            "INSERT OR IGNORE INTO favorites (track_id, agent_id, agent_name, created_at) VALUES (?,?,?,?)",
            (track_id, agent_id, agent_name, now),
        )
        conn.commit()
        conn.close()
        social_volume.commit()
        return JSONResponse(content={"status": "favorited", "track_id": track_id, "agent_id": agent_id})

    @api.delete("/tracks/{track_id}/favorite")
    async def unfavorite_track(track_id: str, request: Request):
        body = await request.json()
        agent_id = body.get("agent_id", "")
        if not agent_id:
            return JSONResponse(status_code=400, content={"error": "agent_id required"})

        social_volume.reload()
        conn = get_db()
        conn.execute("DELETE FROM favorites WHERE track_id = ? AND agent_id = ?", (track_id, agent_id))
        conn.commit()
        conn.close()
        social_volume.commit()
        return JSONResponse(content={"status": "unfavorited"})

    @api.get("/tracks/{track_id}/favorites")
    async def list_favorites(track_id: str):
        social_volume.reload()
        conn = get_db()
        rows = conn.execute(
            "SELECT agent_id, agent_name, created_at FROM favorites WHERE track_id = ? ORDER BY created_at ASC",
            (track_id,),
        ).fetchall()
        conn.close()
        return JSONResponse(content={"favorites": [dict(r) for r in rows]})

    @api.get("/agents/{agent_id}/favorites")
    async def agent_favorites(agent_id: str):
        social_volume.reload()
        conn = get_db()
        rows = conn.execute(
            """SELECT t.*, f.created_at as favorited_at
               FROM favorites f JOIN tracks t ON f.track_id = t.id
               WHERE f.agent_id = ? ORDER BY f.created_at DESC""",
            (agent_id,),
        ).fetchall()
        conn.close()
        return JSONResponse(content={"favorites": [dict(r) for r in rows]})

    # ── Agents ───────────────────────────────────────────────────────
    @api.post("/agents/register")
    async def register_agent(request: Request):
        body = await request.json()
        name = body.get("name", "")
        if not name:
            return JSONResponse(status_code=400, content={"error": "name required"})

        agent_id = body.get("agent_id") or str(uuid.uuid4())[:12]
        description = body.get("description", "")
        personality = body.get("personality", "")

        social_volume.reload()
        conn = get_db()
        # Upsert
        existing = conn.execute("SELECT 1 FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
        now = time.time()
        if existing:
            conn.execute(
                "UPDATE agents SET name=?, description=?, personality=? WHERE agent_id=?",
                (name, description, personality, agent_id),
            )
        else:
            conn.execute(
                "INSERT INTO agents (agent_id, name, description, personality, created_at) VALUES (?,?,?,?,?)",
                (agent_id, name, description, personality, now),
            )
        conn.commit()
        conn.close()
        social_volume.commit()

        return JSONResponse(content={"agent_id": agent_id, "name": name, "description": description})

    @api.get("/agents/{agent_id}")
    async def get_agent(agent_id: str):
        social_volume.reload()
        conn = get_db()
        row = conn.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
        conn.close()
        if not row:
            return JSONResponse(status_code=404, content={"error": "Agent not found"})
        return JSONResponse(content=dict(row))

    @api.get("/agents")
    async def list_agents():
        social_volume.reload()
        conn = get_db()
        rows = conn.execute("SELECT * FROM agents ORDER BY created_at DESC").fetchall()
        conn.close()
        return JSONResponse(content={"agents": [dict(r) for r in rows]})

    return api
