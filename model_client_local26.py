"""
Local Ollama model client — drop-in replacement for model_client
when a local model is selected.

Pipeline:
  Joka chunk        — tietoaukot + oletukset nykyisestä chunkista
  Joka N. chunk     — mini-tiivistelmä N chunkista lisätään lokiin,
                      sitten koko loki konsolidoidaan 6 session-teemaksi
  N valitaan käynnistyksen yhteydessä (1 = joka chunk, 2 = joka 2. chunk)
"""
import json
import re
import time
from pathlib import Path
from openai import OpenAI


OLLAMA_BASE_URL = "http://localhost:11434/v1"

MODEL_IDS = {
    "phi":       "phi4:14b",
    "phi-mini":  "phi4-mini-reasoning:3.8b",
    "gemma":     "gemma3:27b",
    "gemma12":   "gemma3:12b",
    "magistral": "magistral:24b",
}

_MODEL_NAME        = None
_MODEL_KEY         = None
_USE_JSON_MODE     = False   # phi ja phi-mini tukevat json_object-modea
_STRIP_THINK       = False   # phi-mini tuottaa <think>-lohkoja
_transcripts_dir   = None
_analysis_dir      = None
_synteesi_log      = None
_latest_synteesi   = ""
_SYNTEESI_EVERY_N  = 1       # 1 = joka chunk, 2 = joka 2. chunk
_SYNTEESI_NUM_CTX  = 4096    # 4096 riittää 1 chunkille, 8192 kahdelle
_DO_SYNTHESIS      = True
_GAPS_WINDOW       = 1       # how many recent chunks to use for gaps/assumptions
_GAPS_NUM_CTX      = 4096
_LANGUAGE          = "fi"    # set via init(): 'fi' | 'en'


def init(model: str, session_dir, language: str = "fi"):
    global _MODEL_NAME, _MODEL_KEY, _USE_JSON_MODE, _STRIP_THINK, _LANGUAGE
    global _transcripts_dir, _analysis_dir, _synteesi_log, _latest_synteesi
    global _SYNTEESI_EVERY_N, _SYNTEESI_NUM_CTX, _DO_SYNTHESIS
    global _GAPS_WINDOW, _GAPS_NUM_CTX
    _MODEL_KEY       = model
    _LANGUAGE        = language
    _MODEL_NAME      = MODEL_IDS[model]
    _USE_JSON_MODE   = model in ("phi", "phi-mini")
    _STRIP_THINK     = model in ("phi-mini", "magistral")
    _transcripts_dir = Path(session_dir) / "transcripts"
    _analysis_dir    = Path(session_dir) / "analysis"
    _synteesi_log    = _analysis_dir / "synteesi_log.txt"
    _latest_synteesi = ""

    print("\nAnalysis type?  [1] Gaps & assumptions only  [2] Gaps, assumptions & synthesis")
    _analysis_choice = input("Choice [1/2, default 2]: ").strip() or "2"
    _DO_SYNTHESIS = _analysis_choice != "1"

    if _DO_SYNTHESIS:
        print("\nSynthesis frequency?  [1] every chunk  [2] every 2nd chunk")
        _choice = input("Choice [1/2, default 1]: ").strip() or "1"
        _SYNTEESI_EVERY_N = 2 if _choice == "2" else 1
        _SYNTEESI_NUM_CTX = 8192 if _SYNTEESI_EVERY_N == 2 else 4096

    print("\nGaps/assumptions window?  [1] 1 chunk  [2] 2 chunks  [3] 3 chunks")
    _gaps_choice = input("Choice [1/2/3, default 1]: ").strip() or "1"
    _GAPS_WINDOW = {"2": 2, "3": 3}.get(_gaps_choice, 1)
    _GAPS_NUM_CTX = {1: 4096, 2: 8192, 3: 12288}[_GAPS_WINDOW]

    print(f"Local runner: {_MODEL_NAME}  json_mode={_USE_JSON_MODE}  "
          f"think_strip={_STRIP_THINK}  gaps_window={_GAPS_WINDOW}  "
          f"synthesis={'every ' + str(_SYNTEESI_EVERY_N) + ' chunk(s)' if _DO_SYNTHESIS else 'disabled'}")


# ── Apufunktiot ────────────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    if _STRIP_THINK:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text.strip()


def _extract_json(text: str) -> str:
    text = _strip_think(text)
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _repair_json(text: str) -> str:
    # Trailing commas before ] or }
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Missing comma between adjacent quoted strings on separate lines
    text = re.sub(r'"\s*\n(\s*)"', r'",\n\1"', text)
    return text


def _parse_json(raw: str) -> dict:
    extracted = _extract_json(raw)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair_json(extracted))
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON-korjaus epäonnistui: {e}")
        print(f"  Raw (500 merkkiä): {extracted[:500]}")
        raise


def _call(prompt: str, json_mode: bool = False, num_ctx: int = 4096) -> str:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    kwargs = {
        "model":      _MODEL_NAME,
        "messages":   [{"role": "user", "content": prompt}],
        "extra_body": {"options": {"num_ctx": num_ctx}},
    }
    if json_mode and _USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ── Promptit ───────────────────────────────────────────────────────────────────

def _prompt_analysis(transcript: str) -> str:
    if _LANGUAGE == "en":
        return f"""Analyze this English workshop transcript. Identify:
1. Knowledge gaps (tietoaukot) – things recognised as lacking research knowledge or empirical evidence. Each gap must be addressable by academic research or scientific data. Express each as a statement, not a question.
2. Assumptions (oletukset) – things assumed without evidence or confirmation, such as cognitive biases, false interpretations, incorrect causality, or oversimplification.

This is speech-to-text output that may contain transcription errors. Interpret likely intended words.
Transcript:
{transcript}

Respond ONLY with valid JSON in this format:
{{
  "tietoaukot": ["gap 1", "gap 2", "gap 3"],
  "oletukset": ["assumption 1", "assumption 2", "assumption 3"]
}}

Rules:
- List 3-4 most important gaps and assumptions
- Each item max 1-2 sentences in English
- If no clear gaps found, return ["No knowledge gaps identified"]
- If no clear assumptions found, return ["No assumptions identified"]
- Write assumptions as direct statements, NOT as "it is assumed that..."
  Bad: "It is assumed that digitalization improves efficiency."
  Good: "Digitalization automatically improves efficiency."
- Do NOT use quotation marks (" ") inside JSON strings"""
    return f"""Analysoi tämä suomalaisen työpajan litteraatio. Tunnista:
1. Tietoaukot – asiat, jotka tunnistetaan tutkimustiedon tai empiirisen näytön puutteeksi. Tietoaukon täytyy olla aukko, johon akateeminen tutkimus tai tieteellinen data voisi vastata. Ilmaise jokainen tietoaukko väittämänä, ei kysymyksenä.
2. Oletukset – asiat, joita oletetaan ilman näyttöä tai vahvistusta, kuten kognitiiviset harhat, väärät tulkinnat, virheellinen kausaalisuus tai liiallinen yksinkertaistaminen.

Tämä on puheentunnistuksen tuottama teksti, jossa saattaa olla litterointivirheitä. Tulkitse todennäköisesti tarkoitetut sanat äläkä käytä virheellistä tekstiä sellaisenaan.
Transkriptio:
{transcript}

Vastaa VAIN tässä muodossa olevalla JSON:lla:
{{
  "tietoaukot": ["tietoaukko 1", "tietoaukko 2", "tietoaukko 3"],
  "oletukset": ["oletus 1", "oletus 2", "oletus 3"]
}}

Säännöt:
- Listaa 3–4 tärkeintä tietoaukkoa ja oletusta
- Jokainen kohta enintään 1–2 lausetta suomeksi
- Jos selkeitä tietoaukkoja ei löydy, palauta ["Ei tunnistettuja tietoaukkoja"]
- Jos selkeitä oletuksia ei löydy, palauta ["Ei tunnistettuja oletuksia"]
- Kirjoita oletukset suorina väittäminä suomeksi, EI "oletetaan että..." -muodossa, EI englanniksi
  Huono: "Oletetaan että digitalisaatio parantaa tehokkuutta."
  Hyvä:  "Digitalisaatio parantaa tehokkuutta automaattisesti."
- Kirjoita kaikki suomeksi
- ÄLÄ käytä lainausmerkkejä (" ") JSON-merkkijonojen sisällä — kirjoita sanat ilman lainausmerkkejä"""


def _prompt_mini_summary(transcript: str) -> str:
    if _LANGUAGE == "en":
        return f"""Summarize the key discussion points of this English workshop transcript in 3-4 sentences. Focus on the most important topics, arguments, and conclusions.

This is speech-to-text output that may contain errors. Interpret likely intended words.
Transcript:
{transcript}

Write only the summary in English, max 4 sentences. Do not add an introduction or other text."""
    return f"""Tiivistä tämän suomalaisen työpajan litteraation keskeiset keskustelupisteet 3–4 lauseessa. Keskity tärkeimpiin aiheisiin, argumentteihin ja johtopäätöksiin.

Tämä on puheentunnistuksen tuottama teksti, jossa saattaa olla virheitä. Tulkitse todennäköisesti tarkoitetut sanat.
Transkriptio:
{transcript}

Kirjoita vain tiivistelmä suomeksi, enintään 4 lausetta. Älä lisää johdantoa tai muuta tekstiä."""


def _prompt_consolidation(summaries_log: str) -> str:
    if _LANGUAGE == "en":
        return f"""Below are summaries from different phases of an English workshop in chronological order. Based on these, identify the 6 most important themes or topics discussed throughout the session.

Summaries:
{summaries_log}

Respond ONLY with valid JSON in this format:
{{
  "synteesi": [
    "## Theme 1\\nExplanation in 1-2 sentences.",
    "## Theme 2\\nExplanation in 1-2 sentences.",
    "## Theme 3\\nExplanation in 1-2 sentences.",
    "## Theme 4\\nExplanation in 1-2 sentences.",
    "## Theme 5\\nExplanation in 1-2 sentences.",
    "## Theme 6\\nExplanation in 1-2 sentences."
  ]
}}

Rules:
- Exactly 6 strings in the list
- Each starts with ## heading and contains 1-2 sentences in English
- Reflect the entire session, not just the latest part
- Write everything in English"""
    return f"""Alla on tiivistelmiä suomalaisen työpajan eri vaiheista aikajärjestyksessä. Tunnista näiden perusteella 6 tärkeintä teemaa tai aihetta, joita on käsitelty koko session aikana.

Tiivistelmät:
{summaries_log}

Vastaa VAIN tässä muodossa olevalla JSON:lla:
{{
  "synteesi": [
    "## Teema 1\\nSelitys 1–2 lauseella.",
    "## Teema 2\\nSelitys 1–2 lauseella.",
    "## Teema 3\\nSelitys 1–2 lauseella.",
    "## Teema 4\\nSelitys 1–2 lauseella.",
    "## Teema 5\\nSelitys 1–2 lauseella.",
    "## Teema 6\\nSelitys 1–2 lauseella."
  ]
}}

Säännöt:
- Täsmälleen 6 merkkijonoa listassa
- Jokainen alkaa ## -otsikolla ja sisältää 1–2 lausetta suomeksi
- Heijasta koko sessiota, ei vain viimeisintä osaa
- Kirjoita kaikki suomeksi"""


# ── Julkinen rajapinta ─────────────────────────────────────────────────────────

def analyze_transcript(transcript: str):
    """Palauttaa (result_dict, model_name) — sama allekirjoitus kuin model_client."""
    global _latest_synteesi
    t0 = time.time()
    try:
        files = sorted(_transcripts_dir.glob("chunk_*.txt"))
        if not files:
            raise FileNotFoundError("Ei löydy chunk-transkriptiotiedostoja")
        current_chunk_num  = len(files)
        window_files       = files[-_GAPS_WINDOW:]
        current_chunk_text = "\n\n".join(f.read_text(encoding="utf-8") for f in window_files)

        # Tietoaukot + oletukset viimeisimmistä _GAPS_WINDOW chunkista
        raw1   = _call(_prompt_analysis(current_chunk_text), json_mode=True, num_ctx=_GAPS_NUM_CTX)
        parsed = _parse_json(raw1)
        result = {
            "tietoaukot": parsed.get("tietoaukot", []),
            "oletukset":  parsed.get("oletukset",  []),
            "synteesi":   _latest_synteesi,
        }

        # Determine active synthesis log — use separate file during pienryhmä isolation
        try:
            ctx = json.loads((_analysis_dir.parent / "context" / "context_mode.json").read_text(encoding="utf-8"))
            active_log = (_analysis_dir / "small_group_synteesi_log.txt") if ctx.get("mode") == "isolated" else _synteesi_log
        except Exception:
            active_log = _synteesi_log

        # Joka N. chunk: mini-tiivistelmä + konsolidointi
        if _DO_SYNTHESIS and current_chunk_num % _SYNTEESI_EVERY_N == 0:
            # Vaihe 1: mini-tiivistelmä viimeisimmistä N chunkista
            summary_chunks = "\n\n".join(f.read_text(encoding="utf-8") for f in files[-_SYNTEESI_EVERY_N:])
            raw_summary    = _strip_think(_call(_prompt_mini_summary(summary_chunks), num_ctx=_SYNTEESI_NUM_CTX))
            start          = current_chunk_num - _SYNTEESI_EVERY_N + 1
            period         = f"Jakso {start}–{current_chunk_num}" if _SYNTEESI_EVERY_N > 1 else f"Jakso {current_chunk_num}"
            entry       = f"=== {period} ===\n{raw_summary.strip()}\n\n"
            with open(active_log, "a", encoding="utf-8") as fh:
                fh.write(entry)
            print(f"  ✓ Mini-tiivistelmä lisätty ({period})")

            # Vaihe 2: konsolidoi koko loki → 6 session-teemaa
            log_text   = active_log.read_text(encoding="utf-8")
            raw_con    = _call(_prompt_consolidation(log_text), json_mode=True, num_ctx=8192)
            con_parsed = _parse_json(raw_con)
            synteesi_raw = con_parsed.get("synteesi", _latest_synteesi)
            if isinstance(synteesi_raw, list):
                _latest_synteesi = "\n\n".join(synteesi_raw)
            else:
                _latest_synteesi = synteesi_raw
            result["synteesi"] = _latest_synteesi
            print(f"  ✓ Session-synteesi päivitetty")

        elapsed = time.time() - t0
        print(f"✓ {_MODEL_NAME} analyze_transcript ({elapsed:.1f}s)")
        return result, _MODEL_NAME

    except Exception as e:
        print(f"❌ Local model analyze_transcript epäonnistui: {e}")
        return {"tietoaukot": [], "oletukset": [], "synteesi": _latest_synteesi}, None


def analyze_topics_and_agreements(transcript: str):
    """Palauttaa (result_dict, model_name) — sama allekirjoitus kuin model_client."""
    return {"topics": [], "disagreements": [], "not_available": True}, _MODEL_NAME
