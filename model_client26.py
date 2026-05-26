# model_client.py third proto
import os
import json
import httpx
from openai import OpenAI
import anthropic
import google.genai as genai
from dotenv import load_dotenv
load_dotenv()
# Aalto API setup
AALTO_API_KEY = os.environ.get("AALTO_OPENAI_API_KEY")

# Gemini setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Anthropic setup
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Model configurations: (type, identifier, endpoint_path)
MODELS = [
    # Fast models
    ("openai", "gpt-4.1", "/v1/openai/deployments/gpt-4.1-2025-04-14"),
    #("openai", "gpt-4.1-mini", "/v1/openai/deployments/gpt-4.1-mini-2025-04-14"),
    #("openai", "gpt-4o-mini", "/v1/openai/deployments/gpt-4o-mini-2024-07-18"),

    # Reasoning models (slow)
    #("openai", "o1-2024-12-17", "/v1/openai/deployments/o1-2024-12-17"),
    #("openai", "o3-mini-2025-01-31", "/v1/openai/deployments/o3-mini-2025-01-31"),

    # Responses API models (GPT-5 family)
    ("responses", "gpt-4o-2024-11-20", None),
    ("responses", "gpt-5.1-2025-11-13", None),
    ("responses", "gpt-5-2025-08-07", None),

    #("responses", "gpt-5-mini-2025-08-07", None),
    #("responses", "gpt-5-nano-2025-08-07", None),

    # Gemini models
    ("gemini", "gemini-2.5-flash-lite", None),
    ("gemini", "gemini-2.5-flash", None),
    ("gemini", "gemini-2.5-pro", None),

    # Claude fallback models
    ("claude", "claude-haiku-4-5-20251001", None),
    ("claude", "claude-sonnet-4-6", None),
]

_MODEL_PREFERENCE = None  # set by recorder at startup: 'aaltoai' | 'claude' | 'gemini'
_LANGUAGE = "fi"          # set by recorder at startup: 'fi' | 'en'

def set_model_preference(pref: str):
    global _MODEL_PREFERENCE
    _MODEL_PREFERENCE = pref

def set_language(lang: str):
    global _LANGUAGE
    _LANGUAGE = lang

def get_openai_client(endpoint_path):
    """Create OpenAI client for Aalto API with specific endpoint"""
    def update_base_url(request: httpx.Request) -> None:
        if request.url.path == "/chat/completions":
            new_path = f"{endpoint_path}/chat/completions"
            request.url = request.url.copy_with(path=new_path)
        else:
            request.url.path = endpoint_path

    return OpenAI(
        base_url="https://aalto-openai-apigw.azure-api.net",
        api_key=False,
        default_headers={
            "Ocp-Apim-Subscription-Key": AALTO_API_KEY,
        },
        http_client=httpx.Client(
            event_hooks={"request": [update_base_url]}
        ),
    )

def get_responses_client():
    """Create client for Responses API"""
    def update_base_url(request: httpx.Request) -> None:
        request.url = request.url.copy_with(path='/v1/openai/responses')

    return OpenAI(
        base_url="https://aalto-openai-apigw.azure-api.net",
        api_key=False,
        default_headers={
            "Ocp-Apim-Subscription-Key": AALTO_API_KEY,
        },
        http_client=httpx.Client(
            event_hooks={"request": [update_base_url]}
        ),
    )

def try_openai_model(client, prompt, model_name):
    """Try an OpenAI-compatible model"""
    response = client.chat.completions.create(
        model="no_effect",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def try_responses_model(model_name, prompt):
    """Try a Responses API model (stateful API)"""
    client = get_responses_client()
    response = client.responses.create(
        model=model_name,
        input=prompt
    )

    if not hasattr(response, 'output') or response.output is None:
        raise ValueError(f"Response has no output or output is None")

    if not isinstance(response.output, list) or len(response.output) == 0:
        raise ValueError("Response output is not a list or is empty")

    # Responses API returns multiple items: reasoning, message, etc.
    # We need to find the message item with actual content
    for item in response.output:
        item_type = getattr(item, 'type', None)

        # Skip reasoning items (they have content=None)
        if item_type == 'reasoning':
            continue

        # Look for message items
        if item_type == 'message':
            # Message has content list with text objects
            if hasattr(item, 'content') and isinstance(item.content, list):
                for content_item in item.content:
                    if hasattr(content_item, 'text'):
                        return content_item.text
                    elif hasattr(content_item, 'content'):
                        return content_item.content

        # Fallback: try to extract content directly
        if hasattr(item, 'content'):
            if item.content is not None and not isinstance(item.content, list):
                return item.content

    raise ValueError(f"Could not find message content in response.output with {len(response.output)} items")

def try_gemini_model(model_name, prompt):
    """Try a Gemini model"""
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return response.text

def try_claude_model(model_name, prompt):
    """Try an Anthropic Claude model"""
    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _strip_fence(text: str) -> str:
    """Remove ```json ... ``` wrapper if present."""
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1]
        text = text.rsplit('```', 1)[0]
    return text.strip()


def _repair_json(text: str) -> str:
    """Escape literal control characters inside JSON string values."""
    result = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            result.append(ch)
            escape = False
        elif ch == '\\' and in_string:
            result.append(ch)
            escape = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
    return ''.join(result)


def _call_model(prompt: str):
    """Try each model in MODELS order. Returns (raw_text, model_name). Raises if all fail."""
    if _MODEL_PREFERENCE == "claude":
        models = [(t, n, e) for t, n, e in MODELS if t == "claude"] + [(t, n, e) for t, n, e in MODELS if t == "gemini"]
    elif _MODEL_PREFERENCE == "gemini":
        models = [(t, n, e) for t, n, e in MODELS if t == "gemini"] + [(t, n, e) for t, n, e in MODELS if t == "claude"]
    elif _MODEL_PREFERENCE == "aaltoai":
        models = [(t, n, e) for t, n, e in MODELS if t in ("openai", "responses")]#, "claude", "gemini"
    else:
        models = MODELS
    for model_type, model_name, endpoint_path in models:
        try:
            if model_type == "openai":
                result = try_openai_model(get_openai_client(endpoint_path), prompt, model_name)
            elif model_type == "responses":
                result = try_responses_model(model_name, prompt)
            elif model_type == "claude":
                result = try_claude_model(model_name, prompt)
            else:
                result = try_gemini_model(model_name, prompt)
            print(f"✓ {model_name}")
            return result, model_name
        except Exception as e:
            print(f"⚠️ {model_name} failed: {e}")
    raise RuntimeError("All models failed")


def analyze_transcript(transcript, full_context=True):
    
    #Analyze transcript for gaps, assumptions, synthesis.
    #Returns: {"tietoaukot": [...], "oletukset": [...], "synteesi": "..."}
    
    _lang     = "English" if _LANGUAGE == "en" else "Finnish"
    _no_gaps  = "No knowledge gaps identified" if _LANGUAGE == "en" else "Ei tunnistettuja tietoaukkoja"
    _no_assum = "No assumptions identified"    if _LANGUAGE == "en" else "Ei tunnistettuja oletuksia"
    if _LANGUAGE == "en":
        _bold_ex  = 'Bold (**bold**) the 1-2 key concept words in each gap. Example: "There is a lack of evidence on how **remote work** affects long-term team performance." / "No data exists on digitalization\'s impact on employee **workload**."'
        _assum_ex = 'Bad: "It is assumed that digitalization improves efficiency." Good: "Digitalization automatically improves efficiency."'
    else:
        _bold_ex  = 'Bold (**bold**) the 1-2 key concept words in each tietoaukko. Example: "Puuttuu tutkimusnäyttöä siitä, miten **etätyö** vaikuttaa tiimien pitkän aikavälin suorituskykyyn." / "Ei ole tietoa, mikä on digitalisaation vaikutus henkilöstön **työkuormaan**."'
        _assum_ex = 'Bad: "Oletetaan että digitalisaatio parantaa tehokkuutta." Good: "Digitalisaatio parantaa tehokkuutta automaattisesti."'

    prompt = f"""Analyze this {_lang} workshop transcript. Identify:
1. Tietoaukot - things being recognised as lack of research knowledge or lack of empirical evidence. Tietoaukko has to be a gap that academic research or scientific data could address. Express each tietoaukko as a statement, not a question.
2. Assumptions (oletukset) - things being assumed without evidence or explicit confirmation, such as cognitive biases, false interpretations, wrong causal framing, oversimplification.
3. Key discussion points (synteesi) - most relevant topics discussed

This is {_lang} speech-to-text output with potential transcription errors. Interpret likely intended words rather than using garbled text verbatim.
Transcript:
{transcript}

Respond ONLY with valid JSON in this exact format:
{{
  "tietoaukot": ["tietoaukko  1", "tietoaukko  2", "tietoaukko  3"],
  "oletukset": ["assumption 1", "assumption 2", "assumption 3"],
  "synteesi": "## Concept 1\nExplanation of this point.\n\n## Concept 2\nAnother explanation."
}}

Rules:
- List most important tietoaukot and assumptions, 3 or 4 each
- Each item 1-2 sentences max in {_lang}
- If no clear tietoaukko found, return ["{_no_gaps}"]
- If no clear assumptions found, return ["{_no_assum}"]
- {_bold_ex}
- Synthesis should have 4-6 subheadings (## Concept name in {_lang}), each followed by 1-2 sentences of plain explanation
- Write assumptions as direct statements, NOT as "oletetaan että..." constructions. {_assum_ex}
- Do not use **bold** in synthesis text
- Write everything in {_lang}"""

    try:
        text, model_name = _call_model(prompt)
        try:
            return json.loads(_repair_json(_strip_fence(text))), model_name
        except Exception as json_err:
            print(f"❌ analyze_transcript JSON parse failed ({model_name}): {json_err}")
            print(f"   Raw response (first 300 chars): {repr(text[:300])}")
            return {"tietoaukot": [], "oletukset": [], "synteesi": "Analysis failed - all models exhausted"}, None
    except RuntimeError as e:
        print(f"❌ analyze_transcript: all models exhausted — {e}")
        return {"tietoaukot": [], "oletukset": [], "synteesi": "Analysis failed - all models exhausted"}, None


def analyze_topics_and_agreements(transcript):
    """
    Analyze transcript for most discussed topics and agreements/disagreements.
    Returns: {"topics": [...], "disagreements": [...]}
    """
    _lang      = "English" if _LANGUAGE == "en" else "Finnish"
    _no_disagr = "No clear disagreements" if _LANGUAGE == "en" else "Ei selviä erimielisyyksiä"

    prompt = f"""Analyze this {_lang} workshop transcript. Identify:

1. TOPICS: What topics have been discussed most? Count approximate mentions for each.
2. "disagreements": Situations where participants express conflicting views (one says X, another says not-X)
Transcript:
{transcript}

Respond ONLY with valid JSON in this exact format:
{{
  "topics": [
    {{"topic": "topic name in {_lang}", "count": 5}},
    {{"topic": "another topic", "count": 3}}
  ],
  "disagreements": ["point of disagreement 1", "point of disagreement 2"]
}}

Rules:
- List topics in descending order by count (most discussed first)
- If no clear disagreements found, return ["{_no_disagr}"]
- Maximum 10 topics
- Maximum 5 disagreements
- All text in {_lang}
- Be concise, 1 sentence per item"""

    try:
        text, model_name = _call_model(prompt)
        try:
            return json.loads(_repair_json(_strip_fence(text))), model_name
        except Exception as json_err:
            print(f"❌ analyze_topics JSON parse failed ({model_name}): {json_err}")
            print(f"   Raw response (first 300 chars): {repr(text[:300])}")
            return {"topics": [], "agreements": [], "disagreements": []}, None
    except RuntimeError as e:
        print(f"❌ analyze_topics: all models exhausted — {e}")
        return {"topics": [], "agreements": [], "disagreements": []}, None


DIAGRAM_SYSTEM_PROMPT = """Real talk, no sugarcoating. No fluff. You are identifying explicit and implicit and hidden assumptions. Be critical, honest and direct."""

def generate_diagram_data(user_prompt):
    """
    Extract central concept, subtopics, and per-subtopic assumptions.
    Returns: ({"central_concept": "...", "subtopics": [{"name": "...", "assumptions": [...]}, ...]}, model_used)
    """
    _lang = "English" if _LANGUAGE == "en" else "Finnish"
    prompt = f"""{DIAGRAM_SYSTEM_PROMPT}

Read the following text and extract:
1. The single central concept: what the text is trying to achieve or make happen (1-2 words max, a goal or outcome, not a topic).
2. 4-6 key subtopics or themes the text is actually about. Identify them yourself from the content — do not just copy headings.
3. For each subtopic: 2-3 assumptions the text bets on being true — especially ones that could plausibly fail. Not just what the text says, but what it needs to be true to hold up. Max 10 words each.
4. 1-3 meta-level assumptions: zoom out entirely. What does this narrative take for granted that might be completely wrong? Look for hidden dependencies, optimistic beliefs about human behavior, or systemic blind spots. Be direct. Max 12 words each.
Text:
{user_prompt}

Respond in {_lang} ONLY with valid JSON in this exact format:
{{
  "central_concept": "1-2 words",
  "meta_assumptions": ["meta assumption 1", "meta assumption 2"],
  "subtopics": [
    {{
      "name": "subtopic name (2-4 words)",
      "assumptions": ["assumption 1", "assumption 2"]
    }}
  ]
}}"""

    try:
        text, model_name = _call_model(prompt)
        parsed = json.loads(_strip_fence(text))
        parsed['meta_assumptions'] = parsed.get('meta_assumptions', [])[:3]
        parsed['subtopics'] = parsed.get('subtopics', [])[:6]
        for st in parsed['subtopics']:
            st['assumptions'] = st.get('assumptions', [])[:3]
        return parsed, model_name
    except Exception:
        print("❌ All models failed for diagram")
        return {"central_concept": "Unknown", "subtopics": []}, None


def chat_with_transcript(transcript: str, history: list, user_message: str):
    """
    Chat with LLM using the full transcript as context.
    history: list of {"role": "user"|"assistant", "content": str}
    Returns: (response_text, model_used)
    """
    history_str = ""
    for msg in history:
        label = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{label}: {msg['content']}\n"

    _lang = "English" if _LANGUAGE == "en" else "Finnish"
    prompt = (
        f"You are an expert assistant with access to the transcript of a {_lang} workshop. "
        "Answer the user's questions based on the transcript. "
        "If the question is not related to the transcript, you may answer from general knowledge. "
        "The transcript is raw speech-to-text output without speaker labels or diarisation — "
        "it is one continuous block of text from multiple speakers.\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        + (f"CONVERSATION HISTORY:\n{history_str}\n" if history_str else "")
        + f"User: {user_message}\nAssistant:"
    )
    try:
        text, model_name = _call_model(prompt)
        return text.strip(), model_name
    except Exception:
        return "Vastaus epäonnistui – kaikki mallit käytettiin.", None


def generate_yhteenveto(transcript, user_prompt):
    """
    Generate a yhteenveto (summary) using user-provided prompt and full transcript.
    Returns: (yhteenveto_text, model_used)
    """
    prompt = f"{user_prompt}\n\nLitteraatti:\n{transcript}"
    try:
        text, model_name = _call_model(prompt)
        return text.strip(), model_name
    except Exception:
        print("❌ All models failed for yhteenveto")
        return "Yhteenveto epäonnistui – kaikki mallit käytettiin.", None