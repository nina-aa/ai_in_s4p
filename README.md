# AI Workshop Assistant

Records workshop audio in real time, transcribes speech with Whisper, and uses AI to identify knowledge gaps, assumptions, and key discussion themes. Results appear in a browser during the session.

Designed for workshops, seminars, and collaborative meetings (Finnish or English).

\---

## Getting started

See the setup guide for your operating system:

* [SETUP\_MAC.md](SETUP_MAC.md) — Mac
* [SETUP\_WINDOWS.md](SETUP_WINDOWS.md) — Windows

For a visual overview of the interface, see Interface.pdf.

\---

## What the tool does

* Records audio continuously in chunks (default 5 min)
* Transcribes each chunk using [Whisper](https://github.com/openai/whisper)
* Sends transcripts to an AI model (Claude, Gemini, or a local model) for analysis
* Displays results in the browser with a 6–7 min delay:

  * **Knowledge gaps** — things the discussion recognises as lacking research evidence
  * **Assumptions** — things being taken for granted without confirmation
  * **Synthesis** — rolling summary of key themes
  * **Topics** — most discussed topics and points of disagreement
  * **Transcript** — full speech-to-text output
  * **Summary** — on-demand summary with a custom prompt
  * **Chat** — ask questions using the transcript
  * **Images / Diagrams** — image generation and assumption diagram

\---

## File map

|File|What it does|
|-|-|
|`launch.py`|Starts the web server and opens the browser|
|`recorder26.py`|Audio recording, Whisper transcription, triggers AI analysis|
|`model\_client26.py`|Cloud AI client (Claude, Gemini)|
|`model\_client\_local26.py`|Local AI client (phi, gemma)|
|`web/app26.py`|Flask web server, all API routes|
|`web/image\_gen26.py`|Image generation and Excel diagram|
|`web/templates/index26.html`|Main browser UI|
|`web/templates/settings26.html`|Settings and session control UI|
|`.env`|API keys|
|`sessions/`|All recorded session data (audio, transcripts, analysis)|
|`SETUP\_MAC.md`|Mac installation guide|
|`SETUP\_WINDOWS.md`|Windows installation guide|

\---

## AI model options

**Cloud (requires API key):**

* Claude (Anthropic)
* Gemini (Google) — free tier available

**Local (no API key, requires** [**Ollama**](https://ollama.com)**):**

* phi 14b, phi-mini 3.8b, gemma 12b/27b

\---

## License

Copyright (c) 2026 nina-aa. Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Non-commercial use only.

