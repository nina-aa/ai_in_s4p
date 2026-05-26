# Setup Guide — Mac

This tool records workshop audio, transcribes speech, and uses AI to identify knowledge gaps and assumptions in real time.

**Estimated setup time:** 30–60 minutes (most of it is downloading).

---

## What you need

- A Mac (Apple Silicon = M1/M2/M3/M4 chip recommended; Intel Mac also works)
- Internet connection
- At least one API key — Claude (Anthropic) or Gemini (Google) — see Step 5. Or if you want to run a local open source free version, you can do that too, without an API key.
- If you get stuck at any point, you can describe your problem to ChatGPT or Claude and ask for help

---

## Before you start — Download the project

1. Go to the GitHub page for this tool
2. Click the green **Code** button
3. Click **Download ZIP**
4. Once downloaded, find the ZIP file (usually in Downloads), double-click it to unzip
5. Note where the extracted folder is — you will need it in Step 3

---

## Step 1 — Open Terminal

Terminal is the text-based control panel of your Mac. You type commands into it.

**How to open it:**
1. Press **Command + Space** to open Spotlight search
2. Type `Terminal`
3. Press **Enter**

A window opens with a blinking cursor. After each command in this guide, press **Enter** to run it.

---

## Step 2 — Install Python

Python is the programming language the tool runs on. You need version **3.10 or newer**.

**Check if you already have it:**
```
python3 --version
```

- If you see `Python 3.10.x` or higher (3.11, 3.12, etc.) — you are good, skip to Step 3
- If you see `Python 3.8` or `3.9` — too old, follow the installation steps below
- If you get an error — Python is not installed

**To install or update:**
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Click the big **Download Python** button
3. Open the downloaded `.pkg` file and follow the installer
4. When done, re-open Terminal and run `python3 --version` to confirm

If you reopen Terminal and `python3 --version` still gives an error, Python is not on your PATH. Ask ChatGPT or Claude how to add it.

---

## Step 3 — Find your way around the terminal

Before continuing, learn two essential commands:

**See your current folder:**
```
pwd
```

**See what files are in your current folder:**
```
ls
```

**Go to a folder where you downloaded the tool to:**
```
cd documents/foldername
or
cd downloads/foldername
```

Replace `foldername` with the actual folder name. Navigate to the project folder now. Run `ls` and confirm you can see files like `launch.py` and `recorder26.py`.

If you are not sure what the path is, open **Finder**, navigate to the project folder, right-click on it, select **Get Info**, and look at the **Where** field. That gives you the full path to copy.

---

## Step 4 — Install Homebrew and PortAudio

Homebrew is a package manager for Mac. PortAudio is required for audio recording.

**Check if you already have Homebrew:**
```
brew --version
```

If it prints a version, skip the install step and go straight to installing PortAudio.

**Install Homebrew if needed:**
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. At the end it may tell you to run two additional commands to finish setup — do those too.

**Install PortAudio:**
```
brew install portaudio
```

---

## Step 5 — Get API keys

API keys are like passwords that let the tool use AI services. You need at least one — unless you plan to use a local model (phi, gemma, magistral), in which case no API key is needed. If so, you can skip this step.

### Claude (Anthropic) — recommended, best quality

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account or log in
3. Click **API Keys** in the left menu → **Create Key**, give it a name
4. Copy the key — it starts with `sk-ant-`
5. Add a credit card and some funds. Usage is a few cents per workshop session.

### Gemini (Google) — generous free tier

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account
3. Click **Get API key** → **Create API key**
4. Copy the key — it starts with `AIza`
5. Free tier is enough to get started


### Aalto University users only

If you have access to Aalto's OpenAI gateway, add that key as `AALTO_OPENAI_API_KEY`.

---

## Step 6 — Add your API keys to the .env file

The project folder already contains a `.env` file with placeholder text. Open it and replace the placeholders with your real keys.

Make sure you are in the project folder (run `pwd` to check), then open the file:
```
open -e .env
```

This opens it in TextEdit. You will see lines like:
```
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
```

Replace `your-key-here` with your actual keys. Do not remove quotation marks, they are necessary. 
Leave lines for services you do not have — they will simply not activate.

Save with **Command + S** and close TextEdit.

**Verify it saved:** run `cat .env` in Terminal — you should see your keys printed out.

---

## Step 7 — Install Python packages

Run the command for your chip type from anywhere in Terminal:

**How to check your chip:** Apple menu → About This Mac. If it says M1, M2, M3, or M4 — you have Apple Silicon. If it says Intel — you have an Intel Mac.

**Apple Silicon (M1/M2/M3/M4):**
```
pip3 install flask flask-cors openai anthropic google-genai python-dotenv httpx openpyxl sounddevice scipy numpy psutil mlx-whisper
```

**Intel Mac:**
```
pip3 install flask flask-cors openai anthropic google-genai python-dotenv httpx openpyxl sounddevice scipy numpy psutil faster-whisper openai-whisper
```

> **If `pip3` is not found:** Try `pip` instead.

> **If sounddevice fails:** Make sure you ran `brew install portaudio` first (Step 4), then try again.

---

## Step 8 — Run the tool

Make sure you are in the project folder, then:
```
python3 launch.py
```

The script starts the server. After **about 30 seconds**, your browser opens automatically to the settings page. Nothing appearing right away is completely normal — wait for it.

If the browser does not open automatically, go to `http://localhost:5001/settings` yourself.

> **Keep this Terminal window open** the entire time you use the tool. The server runs inside it — if you close it, the tool stops immediately and the browser will show an error. If that happens accidentally, just run `python3 launch.py` again and your session data will still be there.

> **Things generally take longer than you might expect.** The server takes time to start. After you start a session, loading the Whisper model on first run takes another minute or two. Subsequent starts are faster. Be patient throughout.

---

## Step 9 — Configure and start a session

The settings page is open in your browser. Work through the options:

**Language** — Finnish or English, matching what will be spoken

**Sound input** — Mono is correct for almost all microphones

**Analysis frequency** — How often audio is transcribed and analysed. 5 min recommended.

**Whisper model:**
- Apple Silicon Mac → **mlx (Apple)** — fast and accurate
- Intel Mac → **faster-whisper** — recommended; works without extra setup. The **Standard** option also exists but requires ffmpeg (`brew install ffmpeg`) — if Standard Whisper fails, that is the reason.

**Whisper model size** — Medium is the best balance. Large is more accurate but needs 16+ GB of RAM and is slower. Use Small if your Mac is old or slow.

**AI model** — Select **Claude** or **Gemini** based on which key you added. If you select a local model (phi, gemma), new options appear below — these run fully on your Mac with no internet, but require [Ollama](https://ollama.com) installed separately and are slower and less accurate than cloud models.

**Analysis window** — Only visible for cloud AI models. 

**End time** — Optional. The session stops automatically at this time.

Click **Start session**. The browser switches to the main view. The red **REC** indicator in the top left confirms recording is active.

> **First session:** The Whisper model downloads the first time (hundreds of MB). This takes a few extra minutes. Subsequent starts load from disk.

---

## Stopping a session

You can stop in two ways:
- **In the browser:** click the gear icon ⚙, then click **Stop session**
- **In Terminal:** press **Control + C**

Stopping from the browser is cleaner — it lets the tool finish processing the last audio chunk before shutting down.

---

## Where your data is saved

All session data is saved automatically inside the project folder under `sessions/`. Each session gets its own timestamped subfolder (e.g. `sessions/2025-05-26_14-30-00/`) containing:
- `chunks/` — raw audio recordings
- `transcripts/` — text transcriptions of each chunk
- `analysis/` — AI analysis results (gaps, assumptions, synthesis)

Sessions are never deleted automatically. You can open these folders in Finder to access or back up your data.

---

## Troubleshooting

**Browser shows an error when opening localhost** — The server may still be starting. Wait 30–60 seconds and refresh. Check Terminal for error messages.

**"No module named X"** — A package is missing. Run `pip3 install X`.

**"All models failed" in the analysis** — An API key is wrong or has no funds. Open `.env` and check the keys.

**Analysis results missing or timestamps have gaps** — An AI call may have failed silently. The browser does not show an error — check the Terminal window for lines containing "model failed" or "failed:".

**Microphone not working / no transcript appears:**
- Go to **System Settings → Privacy & Security → Microphone**
- Make sure Terminal is in the allowed list and has access turned on

**mlx-whisper errors on Intel Mac** — Switch to `faster-whisper` in Settings.

**Port 5001 already in use** — Another process is using that port. Quit it, or change `port=5001` at the bottom of `web/app26.py`.
