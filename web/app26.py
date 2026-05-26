from flask import Flask, render_template, jsonify, request, send_file
from pathlib import Path
from datetime import datetime
import json
import os
import sys
import signal
import platform
import subprocess
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_client26 import generate_yhteenveto, chat_with_transcript, set_language
from image_gen26 import generate_image_gemini, generate_image_openai, providers_available, generate_diagram_excel
app = Flask(__name__)
CORS(app)

_network_status = {"ok": True, "message": "", "ip": ""}
_recorder_status = {"recording": False, "chunk": 0, "started_at": None}
_recorder_proc = None
_system_status = {"battery_ok": True, "battery_pct": None, "charging": None,
                  "disk_ok": True, "disk_free_gb": None, "disk_hours": None}
_mic_status = {"ok": True, "message": ""}

BASE_DIR = Path(__file__).parent.parent.absolute()
SESSIONS_DIR = BASE_DIR / "sessions"

def get_session_dirs():
    """Return (analysis_dir, transcripts_dir) for the current active session, or (None, None)."""
    pointer = SESSIONS_DIR / "current_session.txt"
    if not pointer.exists():
        return None, None
    session_id = pointer.read_text(encoding="utf-8").strip()
    session_dir = SESSIONS_DIR / session_id
    return session_dir / "analysis", session_dir / "transcripts"

def get_context_dir():
    pointer = SESSIONS_DIR / "current_session.txt"
    if not pointer.exists():
        return None
    return SESSIONS_DIR / pointer.read_text(encoding="utf-8").strip() / "context"

def _get_session_language():
    pointer = SESSIONS_DIR / "current_session.txt"
    if not pointer.exists():
        return "fi"
    session_dir = SESSIONS_DIR / pointer.read_text(encoding="utf-8").strip()
    lang_file = session_dir / "language.txt"
    return lang_file.read_text(encoding="utf-8").strip() if lang_file.exists() else "fi"

def _read_json_list(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        return []

def _read_context_mode(context_dir):
    try:
        return json.loads((context_dir / "context_mode.json").read_text(encoding="utf-8"))
    except Exception:
        return {"mode": "main", "isolation_start_chunk": None, "main_state_backup": None}

@app.route('/api/network-status', methods=['GET'])
def get_network_status():
    return jsonify(_network_status)

@app.route('/api/network-status', methods=['POST'])
def set_network_status():
    global _network_status
    _network_status = request.get_json()
    return jsonify({"received": True})

@app.route('/api/recorder-status', methods=['GET'])
def get_recorder_status():
    # Self-correct: if the process died without posting its own status, fix the flag
    if _recorder_status.get('recording') and (_recorder_proc is None or _recorder_proc.poll() is not None):
        _recorder_status['recording'] = False
    return jsonify(_recorder_status)

@app.route('/api/recorder-status', methods=['POST'])
def set_recorder_status():
    global _recorder_status
    _recorder_status = request.get_json()
    return jsonify({"received": True})

@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    return jsonify(_system_status)

@app.route('/api/system-status', methods=['POST'])
def set_system_status():
    global _system_status
    _system_status = request.get_json()
    return jsonify({"received": True})

@app.route('/api/mic-status', methods=['GET'])
def get_mic_status():
    return jsonify(_mic_status)

@app.route('/api/mic-status', methods=['POST'])
def set_mic_status():
    global _mic_status
    _mic_status = request.get_json()
    return jsonify({"received": True})

@app.route('/')
def index():
    return render_template('index26.html')

@app.route('/settings')
def settings():
    return render_template('settings26.html')

@app.route('/api/latest')
def get_latest():
    analysis_dir, _ = get_session_dirs()
    if not analysis_dir:
        return jsonify({"error": "No active session"})

    latest_file = analysis_dir / "gaps_assumptions_synthesis_latest.json"
    if not latest_file.exists():
        return jsonify({"error": "No analysis yet"})

    with open(latest_file, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

@app.route('/api/structured')
def get_structured():
    analysis_dir, _ = get_session_dirs()
    empty = {'gaps': [], 'assumptions': [], 'synthesis': ''}

    if not analysis_dir:
        return jsonify(empty)

    latest_file = analysis_dir / "gaps_assumptions_synthesis_latest.json"
    if not latest_file.exists():
        return jsonify(empty)

    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    analysis = data.get('analysis', {})
    return jsonify({
        'gaps':        list(reversed(analysis.get('tietoaukot', [])))[:10],
        'assumptions': list(reversed(analysis.get('oletukset',  [])))[:10],
        'synthesis':   analysis.get('synteesi', ''),
        'timestamp':   data.get('timestamp', ''),
    })

@app.route('/api/transcript')
def get_transcript():
    analysis_dir, transcripts_dir = get_session_dirs()

    if not transcripts_dir:
        return jsonify({"error": "No active session"})

    transcript_file = transcripts_dir / "full_transcript.txt"
    if not transcript_file.exists():
        return jsonify({"error": "No transcripts yet"})

    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    chunk_count = 0
    if analysis_dir:
        latest_file = analysis_dir / "gaps_assumptions_synthesis_latest.json"
        if latest_file.exists():
            with open(latest_file, 'r', encoding='utf-8') as f:
                chunk_count = json.load(f).get('chunk_num', 0)

    return jsonify({'transcript': transcript, 'chunk_count': chunk_count})

@app.route('/api/analysis')
def get_analysis():
    analysis_dir, _ = get_session_dirs()
    empty = {'topics': [], 'agreements': [], 'disagreements': []}

    if not analysis_dir:
        return jsonify(empty)

    topics_file = analysis_dir / "topics_latest.json"
    if not topics_file.exists():
        return jsonify(empty)

    with open(topics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return jsonify(data.get('topics_analysis', empty))

@app.route('/api/all')
def get_all():
    analysis_dir, _ = get_session_dirs()

    if not analysis_dir:
        return jsonify([])

    log_file = analysis_dir / "gaps_assumptions_synthesis_log.json"
    if not log_file.exists():
        return jsonify([])

    with open(log_file, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

@app.route('/api/yhteenveto', methods=['POST'])
def create_yhteenveto():
    analysis_dir, transcripts_dir = get_session_dirs()

    if not transcripts_dir:
        return jsonify({"error": "Ei aktiivista sessiota"}), 400

    transcript_file = transcripts_dir / "full_transcript.txt"
    if not transcript_file.exists():
        return jsonify({"error": "Ei vielä litteraattia"}), 400

    data = request.get_json()
    user_prompt = (data or {}).get('prompt', '').strip()
    if not user_prompt:
        return jsonify({"error": "Prompti puuttuu"}), 400

    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    yhteenveto, model_used = generate_yhteenveto(transcript, user_prompt)

    # Save to disk — overwrite so only the latest is kept
    if analysis_dir:
        with open(analysis_dir / "yhteenveto.txt", 'w', encoding='utf-8') as f:
            f.write(yhteenveto)

    return jsonify({'yhteenveto': yhteenveto, 'model_used': model_used})


@app.route('/api/chat', methods=['POST'])
def chat():
    analysis_dir, transcripts_dir = get_session_dirs()
    if not transcripts_dir:
        return jsonify({"error": "Ei aktiivista sessiota"}), 400

    transcript_file = transcripts_dir / "full_transcript.txt"
    if not transcript_file.exists():
        return jsonify({"error": "Ei vielä litteraattia — aloita tallentaminen ensin"}), 400

    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Viesti puuttuu"}), 400

    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    set_language(_get_session_language())
    response, model_used = chat_with_transcript(transcript, history, user_message)
    return jsonify({'response': response, 'model_used': model_used})


@app.route('/api/diagramit', methods=['POST'])
def create_diagramit():
    data = request.get_json()
    user_prompt = (data or {}).get('prompt', '').strip()
    if not user_prompt:
        return jsonify({"error": "Prompt puuttuu"}), 400

    buf, model_used = generate_diagram_excel(user_prompt)
    return send_file(
        buf,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='diagram_assumptions.xlsx',
    )


@app.route('/api/image-providers')
def image_providers():
    return jsonify(providers_available())


@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = (data or {}).get('prompt', '').strip()
    provider = (data or {}).get('provider', 'gemini')

    if not prompt:
        return jsonify({'error': 'Prompt puuttuu'}), 400

    results = {}

    if provider in ('gemini', 'both'):
        try:
            results['gemini'] = generate_image_gemini(prompt)
        except Exception as e:
            results['gemini_error'] = str(e)

    if provider in ('openai', 'both'):
        try:
            results['openai'] = generate_image_openai(prompt)
        except Exception as e:
            results['openai_error'] = str(e)

    return jsonify(results)


@app.route('/api/log', methods=['POST'])
def ui_log():
    pointer = SESSIONS_DIR / "current_session.txt"
    if not pointer.exists():
        return jsonify({"ok": False})
    session_id = pointer.read_text(encoding="utf-8").strip()
    session_dir = SESSIONS_DIR / session_id
    data = request.get_json() or {}
    entry = {"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), **data}
    with open(session_dir / "ui_log.jsonl", 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})


@app.route('/api/save-insight', methods=['POST'])
def save_insight():
    analysis_dir, _ = get_session_dirs()
    if not analysis_dir:
        return jsonify({"ok": False, "error": "No active session"})
    data = request.get_json() or {}
    entry = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "type": data.get("type", ""),
        "text": data.get("text", ""),
    }
    with open(analysis_dir / "saved_insights.jsonl", 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})


@app.route('/api/saved-insights')
def get_saved_insights():
    analysis_dir, _ = get_session_dirs()
    if not analysis_dir:
        return jsonify([])
    filepath = analysis_dir / "saved_insights.jsonl"
    if not filepath.exists():
        return jsonify([])
    seen = set()
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = (entry.get('type', ''), entry.get('text', ''))
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
            except Exception:
                pass
    return jsonify(entries)


@app.route('/api/session-language')
def get_session_language():
    return jsonify({"language": _get_session_language()})


@app.route('/api/session-id')
def get_session_id():
    pointer = SESSIONS_DIR / "current_session.txt"
    if not pointer.exists():
        return jsonify({"session_id": None})
    return jsonify({"session_id": pointer.read_text(encoding="utf-8").strip()})


@app.route('/api/context-mode')
def get_context_mode():
    context_dir = get_context_dir()
    if not context_dir:
        return jsonify({"mode": "main", "isolation_start_chunk": None})
    ctx = _read_context_mode(context_dir)
    return jsonify({"mode": ctx["mode"], "isolation_start_chunk": ctx["isolation_start_chunk"]})


@app.route('/api/isolate', methods=['POST'])
def isolate():
    context_dir = get_context_dir()
    analysis_dir, _ = get_session_dirs()
    if not context_dir or not analysis_dir:
        return jsonify({"ok": False, "error": "No active session"}), 400

    ctx = _read_context_mode(context_dir)
    if ctx["mode"] == "isolated":
        return jsonify({"ok": False, "error": "Already isolated"}), 400

    gaps_log   = _read_json_list(analysis_dir / "gaps_assumptions_synthesis_log.json")
    topics_log = _read_json_list(analysis_dir / "topics_log.json")

    isolation_start_chunk = _recorder_status.get("chunk", 0) + 1
    ctx = {
        "mode": "isolated",
        "isolation_start_chunk": isolation_start_chunk,
        "main_state_backup": {"gaps_log": gaps_log, "topics_log": topics_log}
    }
    (context_dir / "context_mode.json").write_text(
        json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonify({"ok": True, "isolation_start_chunk": isolation_start_chunk})


@app.route('/api/merge', methods=['POST'])
def merge():
    context_dir = get_context_dir()
    analysis_dir, _ = get_session_dirs()
    if not context_dir or not analysis_dir:
        return jsonify({"ok": False, "error": "No active session"}), 400

    ctx = _read_context_mode(context_dir)
    if ctx["mode"] != "isolated":
        return jsonify({"ok": False, "error": "Not in isolated mode"}), 400

    # Read live disk logs rather than the stale backup snapshot.
    # The backup was taken when isolation started; any analysis that was already
    # running at that moment (chunk_num < isolation_start_chunk) completed after
    # the snapshot and wrote to the main log on disk, so the live file is more
    # complete.  Isolated-mode chunks always go to small_group_*_log.json and
    # never to the main log, so reading the live file is safe.
    main_gaps   = _read_json_list(analysis_dir / "gaps_assumptions_synthesis_log.json")
    main_topics = _read_json_list(analysis_dir / "topics_log.json")

    small_gaps   = _read_json_list(analysis_dir / "small_group_gaps_log.json")
    small_topics = _read_json_list(analysis_dir / "small_group_topics_log.json")

    merged_gaps   = main_gaps   + small_gaps
    merged_topics = main_topics + small_topics

    (analysis_dir / "gaps_assumptions_synthesis_log.json").write_text(
        json.dumps(merged_gaps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis_dir / "topics_log.json").write_text(
        json.dumps(merged_topics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    post_start = _recorder_status.get("chunk", 0) + 1
    reset = {"mode": "main", "isolation_start_chunk": None, "post_isolation_start_chunk": post_start, "main_state_backup": None}
    (context_dir / "context_mode.json").write_text(
        json.dumps(reset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonify({"ok": True})


@app.route('/api/recorder-running')
def recorder_running():
    running = _recorder_proc is not None and _recorder_proc.poll() is None
    return jsonify({"running": running})


@app.route('/api/start', methods=['POST'])
def start_recorder():
    global _recorder_proc
    if _recorder_proc and _recorder_proc.poll() is None:
        return jsonify({"ok": False, "error": "Already running"}), 400

    data = request.get_json() or {}
    recorder_path = str(BASE_DIR / "recorder26.py")
    args = [sys.executable, recorder_path]

    args += ['--whisper-backend', data.get('whisper_backend', 'faster')]
    args += ['--model-size',      data.get('model_size', 'medium')]
    args += ['--channels',        str(data.get('channels', 1))]
    args += ['--language',        data.get('language', 'fi')]
    args += ['--chunk-duration',  str(data.get('chunk_duration', 300))]
    args += ['--ai-model',        data.get('ai_model', 'claude')]

    gw = data.get('gaps_window')
    if gw is not None:
        args += ['--gaps-window', str(gw)]

    et = data.get('end_time')
    if et:
        args += ['--end-time', et]

    args += ['--local-do-synthesis',    str(data.get('local_do_synthesis', 1))]
    args += ['--local-synteesi-every-n', str(data.get('local_synteesi_every_n', 1))]
    args += ['--local-gaps-window',     str(data.get('local_gaps_window', 1))]

    try:
        if platform.system() == "Windows":
            _recorder_proc = subprocess.Popen(
                args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            _recorder_proc = subprocess.Popen(args)
        return jsonify({"ok": True, "pid": _recorder_proc.pid})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_recorder():
    global _recorder_proc
    if not _recorder_proc or _recorder_proc.poll() is not None:
        return jsonify({"ok": False, "error": "Not running"}), 400
    try:
        if platform.system() == "Windows":
            # CTRL_BREAK_EVENT triggers a forrtl crash inside numpy/scipy before Python
            # can catch it — use terminate() instead (TerminateProcess); outcome is the same
            _recorder_proc.terminate()
        else:
            _recorder_proc.send_signal(signal.SIGINT)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)