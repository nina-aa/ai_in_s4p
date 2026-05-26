import subprocess
import time
import json
import re
import sys
import argparse
import platform
import shutil
import urllib.request as _urllib
from pathlib import Path
import _thread
from datetime import datetime, timedelta
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import threading
import queue
try:
    import whisper as _whisper_std
    _WHISPER_STD_AVAILABLE = True
except ImportError:
    _WHISPER_STD_AVAILABLE = False

try:
    from faster_whisper import WhisperModel as _WhisperModel_fw
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False

try:
    import mlx_whisper as _mlx_whisper
    _MLX_WHISPER_AVAILABLE = True
except ImportError:
    _MLX_WHISPER_AVAILABLE = False

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"
try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not installed — battery check unavailable. Run: pip install psutil")
from model_client26 import analyze_transcript, analyze_topics_and_agreements, set_model_preference, set_language

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

BASE_DIR = Path(__file__).parent.absolute()
SESSIONS_DIR = BASE_DIR / "sessions"

# Session setup — SESSION_ID is fixed at startup for the whole session lifetime
SESSION_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SESSION_DIR = SESSIONS_DIR / SESSION_ID

CHUNKS_DIR      = SESSION_DIR / "chunks"
TRANSCRIPTS_DIR = SESSION_DIR / "transcripts"
ANALYSIS_DIR    = SESSION_DIR / "analysis"
CONTEXT_DIR     = SESSION_DIR / "context"

for d in [CHUNKS_DIR, TRANSCRIPTS_DIR, ANALYSIS_DIR, CONTEXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Write pointer so app.py knows which session is active
(SESSIONS_DIR / "current_session.txt").write_text(SESSION_ID, encoding="utf-8")

# Initialize context isolation state
(CONTEXT_DIR / "context_mode.json").write_text(
    json.dumps({"mode": "main", "isolation_start_chunk": None, "main_state_backup": None},
               ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"Session: {SESSION_ID}  →  {SESSION_DIR}")

# Configuration
SAMPLE_RATE = 16000 #16000 #48000 jutikkala #
#keeping 16000 is actually the best choice for this use case. 
# It's Whisper's native sample rate (trained on 16000 Hz audio), 
# so you get slightly better transcription quality with no resampling on the Whisper side
OVERLAP = 5  # seconds of overlap between chunks
RMS_SILENCE_THRESHOLD = 0.001  # chunks below this are skipped entirely
DISK_MIN_BYTES = 3 * 3600 * 48000 * 2  # 3 hours @ 48 kHz 16-bit mono ≈ 989 MB
MIC_WATCHDOG_TIMEOUT = 5.0             # seconds without audio → assume disconnect
SYSTEM_CHECK_INTERVAL = 20 * 60        # re-check battery/disk every 20 minutes

def _parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--whisper-backend', dest='whisper_backend', default='faster',
                   choices=['standard', 'faster', 'mlx'])
    p.add_argument('--model-size', dest='model_size', default='medium',
                   choices=['small', 'medium', 'large'])
    p.add_argument('--channels', type=int, default=1, choices=[1, 2])
    p.add_argument('--language', default='fi', choices=['fi', 'en'])
    p.add_argument('--chunk-duration', dest='chunk_duration', type=int, default=300)
    p.add_argument('--ai-model', dest='ai_model', default='aaltoai')
    p.add_argument('--gaps-window', dest='gaps_window', type=int, default=None)
    p.add_argument('--end-time', dest='end_time', default=None)
    p.add_argument('--local-do-synthesis', dest='local_do_synthesis', type=int, default=1)
    p.add_argument('--local-synteesi-every-n', dest='local_synteesi_every_n', type=int, default=1)
    p.add_argument('--local-gaps-window', dest='local_gaps_window', type=int, default=1)
    return p.parse_args()

_args = _parse_args()

WHISPER_BACKEND = _args.whisper_backend
WHISPER_MODEL   = _args.model_size
N_CHANNELS      = _args.channels
LANGUAGE        = _args.language
CHUNK_DURATION  = _args.chunk_duration
_AI_PREF        = _args.ai_model
GAPS_WINDOW     = _args.gaps_window

print(f"[Config] backend={WHISPER_BACKEND}  model={WHISPER_MODEL}  ch={N_CHANNELS}  "
      f"lang={LANGUAGE}  chunk={CHUNK_DURATION}s  ai={_AI_PREF}")
(SESSION_DIR / "language.txt").write_text(LANGUAGE, encoding="utf-8")

if _AI_PREF in ("phi", "phi-mini", "gemma", "gemma12", "magistral"):
    import model_client_local26 as _lmr
    _lmr._MODEL_KEY        = _AI_PREF
    _lmr._LANGUAGE         = LANGUAGE
    _lmr._MODEL_NAME       = _lmr.MODEL_IDS[_AI_PREF]
    _lmr._USE_JSON_MODE    = _AI_PREF in ("phi", "phi-mini")
    _lmr._STRIP_THINK      = _AI_PREF in ("phi-mini", "magistral")
    _lmr._transcripts_dir  = SESSION_DIR / "transcripts"
    _lmr._analysis_dir     = SESSION_DIR / "analysis"
    _lmr._synteesi_log     = _lmr._analysis_dir / "synteesi_log.txt"
    _lmr._latest_synteesi  = ""
    _lmr._DO_SYNTHESIS     = bool(_args.local_do_synthesis)
    _lmr._SYNTEESI_EVERY_N = _args.local_synteesi_every_n
    _lmr._SYNTEESI_NUM_CTX = 8192 if _args.local_synteesi_every_n == 2 else 4096
    _lmr._GAPS_WINDOW      = _args.local_gaps_window
    _lmr._GAPS_NUM_CTX     = {1: 4096, 2: 8192, 3: 12288}[_args.local_gaps_window]
    analyze_transcript            = _lmr.analyze_transcript
    analyze_topics_and_agreements = _lmr.analyze_topics_and_agreements
    GAPS_WINDOW = None
else:
    set_model_preference(_AI_PREF)
    set_language(LANGUAGE)
    print(f"[Config] gaps_window={'last 3 chunks' if GAPS_WINDOW else 'whole subsession'}")

if WHISPER_BACKEND == "standard":
    if not _WHISPER_STD_AVAILABLE:
        print("ERROR: openai-whisper not installed. Run: pip install openai-whisper")
        sys.exit(1)
    print(f"Loading whisper '{WHISPER_MODEL}' on {DEVICE}...")
    _load_start = time.time()
    whisper_model = _whisper_std.load_model(WHISPER_MODEL, device=DEVICE)
    print(f"Whisper model loaded in {time.time() - _load_start:.1f}s")
elif WHISPER_BACKEND == "mlx":
    if not _MLX_WHISPER_AVAILABLE:
        print("ERROR: mlx-whisper not installed. Run: pip install mlx-whisper")
        sys.exit(1)
    _mlx_model_map = {
        "small":  "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large":  "mlx-community/whisper-large-v3-mlx",
    }
    whisper_model = _mlx_model_map.get(WHISPER_MODEL, "mlx-community/whisper-medium-mlx")
    print(f"MLX Whisper model: {whisper_model}  (Apple Silicon — loads on first transcription)")
else:
    if not _FASTER_WHISPER_AVAILABLE:
        print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
        sys.exit(1)
    _fw_device  = "cuda" if DEVICE == "cuda" else "cpu"
    _fw_compute = "float16" if _fw_device == "cuda" else "int8"
    print(f"Loading faster-whisper '{WHISPER_MODEL}' on {_fw_device} ({_fw_compute})...")
    _load_start = time.time()
    whisper_model = _WhisperModel_fw(WHISPER_MODEL, device=_fw_device, compute_type=_fw_compute)
    print(f"faster-whisper model loaded in {time.time() - _load_start:.1f}s")

class ContinuousRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.buffer = np.array([], dtype=np.float32)
        self.last_audio_time = datetime.now().timestamp()

    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block"""
        if status:
            print(f"Audio error: {status}")
        self.last_audio_time = datetime.now().timestamp()  # use datetime; 'time' here is sd's object
        audio = indata.mean(axis=1) if indata.ndim == 2 else indata[:, 0]
        self.audio_queue.put(audio.copy())

    def _mic_watchdog_loop(self):
        """Background thread: alert if audio stops arriving (mic disconnected)."""
        was_ok = True
        while self.is_recording:
            time.sleep(1)
            if not self.is_recording:
                break
            elapsed = datetime.now().timestamp() - self.last_audio_time
            is_ok = elapsed < MIC_WATCHDOG_TIMEOUT
            if not is_ok and was_ok:
                msg = f"No audio for {elapsed:.0f}s — microphone disconnected?"
                print(f"\n[{get_timestamp()}] ⚠ {msg}")
                _post_mic_status(ok=False, message=msg)
                was_ok = False
            elif is_ok and not was_ok:
                print(f"[{get_timestamp()}] ✓ Microphone reconnected")
                _post_mic_status(ok=True, message="")
                was_ok = True

    def start_recording(self):
        """Start continuous recording"""
        self.is_recording = True
        self.last_audio_time = datetime.now().timestamp()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=N_CHANNELS,
            callback=self.audio_callback,
            blocksize=8192
        )
        self.stream.start()
        threading.Thread(target=self._mic_watchdog_loop, daemon=True).start()
        print(f"[{get_timestamp()}] 🔴 Recording started (continuous)")

    def get_chunk(self, duration):
        """Get next chunk of audio with overlap"""
        samples_needed = int(SAMPLE_RATE * duration)

        audio_chunks = []
        current_samples = len(self.buffer)

        while current_samples < samples_needed:
            try:
                new_audio = self.audio_queue.get(timeout=1)
                audio_chunks.append(new_audio.flatten())
                current_samples += len(new_audio)
            except queue.Empty:
                continue

        if audio_chunks:
            self.buffer = np.concatenate([self.buffer] + audio_chunks)

        chunk = self.buffer[:samples_needed]
        overlap_samples = int(SAMPLE_RATE * OVERLAP)
        self.buffer = self.buffer[samples_needed - overlap_samples:]

        return chunk

    def stop(self):
        self.is_recording = False
        self.stream.stop()
        self.stream.close()

def print_audio_stats(audio_data, chunk_num):
    """Print audio energy stats to help calibrate silence detection."""
    rms  = float(np.sqrt(np.mean(audio_data ** 2)))
    peak = float(np.max(np.abs(audio_data)))

    window = int(SAMPLE_RATE * 0.1)  # 100 ms windows
    frames = [audio_data[i:i+window] for i in range(0, len(audio_data) - window, window)]
    n = len(frames) or 1
    frame_rms = [float(np.sqrt(np.mean(f**2))) for f in frames]

    pct_low  = 100 * sum(1 for r in frame_rms if r > 0.003) / n
    pct_mid  = 100 * sum(1 for r in frame_rms if r > 0.01)  / n
    pct_high = 100 * sum(1 for r in frame_rms if r > 0.04)  / n

    print(f"[{get_timestamp()}] 📊 Chunk {chunk_num:03d} — "
          f"RMS: {rms:.4f}  peak: {peak:.4f}  "
          f"active frames: {pct_low:.0f}%/>0.003  {pct_mid:.0f}%/>0.01  {pct_high:.0f}%/>0.04")
    return rms


def transcribe_chunk(audio_file):
    """Transcribe with Whisper (in-process, model loaded once at startup)"""
    output_file = TRANSCRIPTS_DIR / f"{audio_file.stem}.txt"
    print(f"[{get_timestamp()}] 📝 Transcribing {audio_file.name}...")

    start_time = time.time()
    _lang_full = "English" if LANGUAGE == "en" else "Finnish"
    if WHISPER_BACKEND == "standard":
        result = whisper_model.transcribe(str(audio_file), language=_lang_full, fp16=(DEVICE == "cuda"))
        text = result["text"].strip()
    elif WHISPER_BACKEND == "mlx":
        result = _mlx_whisper.transcribe(str(audio_file), path_or_hf_repo=whisper_model, language=LANGUAGE)
        text = result["text"].strip()
    else:
        segments, _ = whisper_model.transcribe(str(audio_file), language=LANGUAGE)
        text = " ".join(seg.text for seg in segments).strip()
    elapsed = time.time() - start_time
    print(f"[{get_timestamp()}] ✓ Transcription complete: {audio_file.name} ({elapsed:.1f}s)")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    return text

def get_full_transcript(since_chunk=None):
    """Concatenate all transcripts in chronological order.
    If since_chunk is set, only include chunk files with number >= since_chunk."""
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("chunk_*.txt"))
    full_text = []
    for file in transcript_files:
        if since_chunk is not None:
            chunk_n = int(file.stem.split("_")[1])
            if chunk_n < since_chunk:
                continue
        with open(file, 'r', encoding='utf-8') as f:
            full_text.append(f.read())
    return "\n\n".join(full_text)

def _append_to_log(log_file, entry):
    """Append an entry to a JSON array log file."""
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            log = json.load(f)
    else:
        log = []
    log.append(entry)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def get_context_mode():
    try:
        return json.loads((CONTEXT_DIR / "context_mode.json").read_text(encoding="utf-8"))
    except Exception:
        return {"mode": "main", "isolation_start_chunk": None, "main_state_backup": None}

def analyze_cumulative(chunk_num):
    """Analyze full transcript up to current point"""
    ctx = get_context_mode()
    isolated = ctx["mode"] == "isolated" and ctx["isolation_start_chunk"] is not None
    if isolated:
        since = ctx["isolation_start_chunk"]
    else:
        since = ctx.get("post_isolation_start_chunk")
    log_file = ANALYSIS_DIR / ("small_group_gaps_log.json" if isolated else "gaps_assumptions_synthesis_log.json")
    label = "pienryhmä" if isolated else "full"

    print(f"[{get_timestamp()}] 🤖 Analyzing gaps & assumptions ({label}, up to chunk {chunk_num})...")

    if isolated and since is not None and since > chunk_num:
        print(f"[{get_timestamp()}] ⏭️ Skipping gaps analysis — isolation starts at chunk {since}, current chunk {chunk_num} not yet in scope")
        return None

    base_since = since  # keep for full-transcript disk write
    if GAPS_WINDOW is not None:
        window_start = chunk_num - GAPS_WINDOW + 1
        since = max(since, window_start) if since is not None else max(1, window_start)

    analysis_transcript = get_full_transcript(since_chunk=since)

    if chunk_num % 5 == 0:
        token_est = len(analysis_transcript) // 4
        print(f"[{get_timestamp()}] 📊 Transcript: ~{token_est:,} tokens (~{len(analysis_transcript):,} chars)")

    # Always write the complete (non-windowed) transcript so chat/yhteenveto see everything
    with open(TRANSCRIPTS_DIR / "full_transcript.txt", 'w', encoding='utf-8') as f:
        f.write(get_full_transcript(since_chunk=base_since))

    start_time = time.time()
    result, model_used = analyze_transcript(analysis_transcript)
    elapsed = time.time() - start_time
    if model_used:
        print(f"[{get_timestamp()}] ✓ Gaps & assumptions analysis complete ({elapsed:.1f}s)")
    else:
        print(f"[{get_timestamp()}] ❌ Gaps & assumptions analysis failed ({elapsed:.1f}s)")

    output_data = {
        "session_id": SESSION_ID,
        "chunk_num": chunk_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": model_used,
        "analysis": result
    }

    with open(ANALYSIS_DIR / f"gaps_assumptions_synthesis_at_chunk_{chunk_num:03d}.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if model_used:
        with open(ANALYSIS_DIR / "gaps_assumptions_synthesis_latest.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    _append_to_log(log_file, output_data)

    return output_data

def analyze_topics_cumulative(chunk_num):
    """Analyze topics/agreements for full transcript and save to disk"""
    ctx = get_context_mode()
    isolated = ctx["mode"] == "isolated" and ctx["isolation_start_chunk"] is not None
    if isolated:
        since = ctx["isolation_start_chunk"]
    else:
        since = ctx.get("post_isolation_start_chunk")
    log_file = ANALYSIS_DIR / ("small_group_topics_log.json" if isolated else "topics_log.json")

    if isolated and since is not None and since > chunk_num:
        print(f"[{get_timestamp()}] ⏭️ Skipping topics analysis — isolation starts at chunk {since}, current chunk {chunk_num} not yet in scope")
        return None
    label = "pienryhmä" if isolated else "full"

    print(f"[{get_timestamp()}] 📊 Analyzing topics ({label}, up to chunk {chunk_num})...")

    full_transcript = get_full_transcript(since_chunk=since)

    start_time = time.time()
    result, model_used = analyze_topics_and_agreements(full_transcript)
    elapsed = time.time() - start_time
    if model_used:
        print(f"[{get_timestamp()}] ✓ Topics analysis complete ({elapsed:.1f}s)")
    else:
        print(f"[{get_timestamp()}] ❌ Topics analysis failed ({elapsed:.1f}s)")

    output_data = {
        "session_id": SESSION_ID,
        "chunk_num": chunk_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": model_used,
        "topics_analysis": result
    }

    with open(ANALYSIS_DIR / f"topics_at_chunk_{chunk_num:03d}.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if model_used:
        with open(ANALYSIS_DIR / "topics_latest.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    _append_to_log(log_file, output_data)

    return output_data

def process_chunk_async(chunk_num, audio_file):
    """Process chunk in background thread"""
    transcribe_chunk(audio_file)

    t1 = threading.Thread(target=analyze_cumulative, args=(chunk_num,))
    t2 = threading.Thread(target=analyze_topics_cumulative, args=(chunk_num,))
    t1.start(); t2.start()
    t1.join();  t2.join()

    print(f"[{get_timestamp()}] ✅ Chunk {chunk_num} processed\n")

def _post(path, payload):
    """POST JSON to Flask. Silent if Flask isn't running."""
    try:
        data = json.dumps(payload).encode()
        req = _urllib.Request(
            f"http://localhost:5001{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        _urllib.urlopen(req, timeout=2)
    except Exception:
        pass

def _post_network_status(ok, message, ip):
    _post("/api/network-status", {"ok": ok, "message": message, "ip": ip})

def _post_recorder_status(recording, chunk, started_at):
    _post("/api/recorder-status", {"recording": recording, "chunk": chunk, "started_at": started_at, "chunk_duration": CHUNK_DURATION})

def _post_system_status(battery_ok, battery_pct, charging, disk_ok, disk_free_gb, disk_hours):
    _post("/api/system-status", {
        "battery_ok": battery_ok,
        "battery_pct": battery_pct,
        "charging": charging,
        "disk_ok": disk_ok,
        "disk_free_gb": disk_free_gb,
        "disk_hours": disk_hours,
    })

def _post_mic_status(ok, message):
    _post("/api/mic-status", {"ok": ok, "message": message})


def check_system_status():
    """Check disk space and battery; print to terminal and POST status to UI."""
    print("=== System Status ===")
    print(f"  Whisper device : {DEVICE}")

    # Disk space on the session drive
    disk = shutil.disk_usage(str(BASE_DIR))
    disk_free_gb = disk.free / 1e9
    disk_hours   = disk.free / DISK_MIN_BYTES * 3
    disk_ok      = disk.free >= DISK_MIN_BYTES
    print(f"  Disk free      : {disk_free_gb:.1f} GB  ≈{disk_hours:.1f}h @ 48kHz  {'✓' if disk_ok else '⚠ LOW'}")

    # Battery
    battery_ok  = True
    battery_pct = None
    charging    = None
    try:
        bat = psutil.sensors_battery() if psutil else None
        if bat is not None:
            battery_pct = round(bat.percent)
            charging    = bat.power_plugged
            battery_ok  = battery_pct >= 30 or bool(charging)
            charge_str  = "⚡ Charging" if charging else "🔋 Discharging"
            warn_str    = "  ⚠ LOW" if not battery_ok else ""
            print(f"  Battery        : {battery_pct}%  {charge_str}{warn_str}")
        else:
            print("  Battery        : N/A (desktop / sensor unavailable)")
    except Exception as e:
        print(f"  Battery        : could not read ({e})")

    print("=====================\n")

    # Token count from current full transcript (if available)
    transcript_file = TRANSCRIPTS_DIR / "full_transcript.txt"
    if transcript_file.exists():
        txt_size = transcript_file.stat().st_size
        print(f"  Transcript     : ~{txt_size // 4:,} tokens (est.)")

    _post_system_status(
        battery_ok=battery_ok,
        battery_pct=battery_pct,
        charging=charging,
        disk_ok=disk_ok,
        disk_free_gb=round(disk_free_gb, 1),
        disk_hours=round(disk_hours, 1),
    )
    return disk_ok, battery_ok


def _session_end_timer(end_dt):
    delay = (end_dt - datetime.now()).total_seconds()
    if delay > 0:
        time.sleep(delay)
    print(f"\n[{get_timestamp()}] ⏰ Scheduled session end — stopping...")
    _thread.interrupt_main()

def _periodic_system_check_loop():
    """Re-check battery and disk every 20 minutes during recording."""
    while True:
        time.sleep(SYSTEM_CHECK_INTERVAL)
        print(f"\n[{get_timestamp()}] 🔄 Periodic system check ({SYSTEM_CHECK_INTERVAL // 60} min)...")
        check_system_status()

def main():
    check_system_status()

    end_dt = None
    if _args.end_time:
        try:
            now = datetime.now()
            h, m = map(int, _args.end_time.split(":"))
            end_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if end_dt <= now:
                end_dt += timedelta(days=1)
            mins = int((end_dt - now).total_seconds() / 60)
            print(f"[{get_timestamp()}] ⏰ Auto-stop at {end_dt.strftime('%H:%M')} ({mins} min)")
        except Exception:
            print(f"[{get_timestamp()}] Could not parse end time: {_args.end_time}")

    print("Starting automated workshop recording...")
    print(f"Chunk duration: {CHUNK_DURATION}s with {OVERLAP}s overlap")
    print("Press Ctrl+C to stop\n")

    recorder = ContinuousRecorder()
    recorder.start_recording()

    if end_dt:
        threading.Thread(target=_session_end_timer, args=(end_dt,), daemon=True).start()
    threading.Thread(target=_periodic_system_check_loop, daemon=True).start()

    recording_started_at = datetime.now().isoformat()
    _post_recorder_status(recording=True, chunk=0, started_at=recording_started_at)

    chunk_num = 1
    processing_thread = None

    try:
        while True:
            print(f"[{get_timestamp()}] 📦 Getting chunk {chunk_num}...")
            audio_data = recorder.get_chunk(CHUNK_DURATION)

            output_file = CHUNKS_DIR / f"chunk_{chunk_num:03d}.wav"
            wav.write(str(output_file), SAMPLE_RATE, audio_data)
            print(f"[{get_timestamp()}] ✓ Chunk {chunk_num} saved")
            rms = print_audio_stats(audio_data, chunk_num)

            if rms < RMS_SILENCE_THRESHOLD:
                print(f"[{get_timestamp()}] ⏭️ Chunk {chunk_num} skipped — silent (RMS {rms:.4f} < {RMS_SILENCE_THRESHOLD})")
                chunk_num += 1
                continue

            _post_recorder_status(recording=True, chunk=chunk_num, started_at=recording_started_at)

            if processing_thread:
                processing_thread.join()

            processing_thread = threading.Thread(
                target=process_chunk_async,
                args=(chunk_num, output_file)
            )
            processing_thread.start()

            chunk_num += 1

    except KeyboardInterrupt:
        recorder.stop()
        _post_recorder_status(recording=False, chunk=chunk_num - 1, started_at=recording_started_at)
        if processing_thread:
            processing_thread.join()
        print(f"\nStopped. Processed {chunk_num - 1} chunks.")

if __name__ == "__main__":
    main()