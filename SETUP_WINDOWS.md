# Setup Guide — Windows

This tool records workshop audio, transcribes speech, and uses AI to identify knowledge gaps and assumptions in real time.

**Estimated setup time:** 30–60 minutes (most of it is downloading).

---

## What you need

- Windows 10 or 11
- Internet connection
- At least one API key — Claude (Anthropic) or Gemini (Google) — see Step 5. Or if you want to run a local open source free version, you can do that too, without API key.
- If you get stuck at any point, you can describe your problem to ChatGPT or Claude and ask for help

---

## Before you start — Download the project

1. Go to the GitHub page for this tool
2. Click the green **Code** button
3. Click **Download ZIP**
4. Once downloaded, find the ZIP file (usually in your Downloads folder), right-click it and select **Extract All**
5. Choose where to extract it and note that location — you will need it in Step 3

---

## Step 1 — Open a terminal

A terminal is a window where you type commands. On Windows you can use **PowerShell** or **Terminal** — both work the same way for this guide.

**How to open PowerShell:**
1. Press the **Windows key**, type `PowerShell`
2. Click **Windows PowerShell** — do not choose "ISE" or "x86"

A blue or dark window opens with a blinking cursor. After each command in this guide, press **Enter** to run it.

---

## Step 2 — Install Python

Python is the programming language the tool runs on. You need version **3.10 or newer**.

**Check if you already have it:**
```
python --version
```

- If you see `Python 3.10.x` or higher (3.11, 3.12, etc.) — you are good, skip to Step 3
- If you see `Python 3.8` or `3.9` — too old, follow the installation steps below
- If you get an error or nothing happens — Python is not installed

**To install or update:**
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Click the big **Download Python** button
3. Run the downloaded `.exe` file
4. **On the first installer screen:** check the box **"Add Python to PATH"** — this is easy to miss and causes problems if skipped
5. Click **Install Now** and let it finish
6. Close and re-open PowerShell, then run `python --version` to confirm

If you reopen PowerShell and `python --version` still gives an error, Python is not on your PATH. Ask ChatGPT or Claude how to add it.

---

## Step 3 — Find your way around the terminal

Before continuing, learn two essential commands:

**See your current folder:**
```
pwd
```

**See what files are in your current folder:**
```
dir
```

(You can also type `ls` — it does the same thing in PowerShell.)

**Go to a folder where you downloaded the tool, for example:**
```
cd Documents/foldername
```

PowerShell starts in your home folder, so you usually do not need to type the full path. Replace `foldername` with the actual folder name. If that does not work, use the full path:
```
cd C:/Users/YourName/Documents/foldername
```

PowerShell accepts both forward slashes `/` and backslashes `\` — use whichever feels natural.

If you are not sure what the path is, open **File Explorer**, navigate to the project folder, right-click on it, select **Properties**, and look at the **Location** field. That gives you the full path to copy.

Navigate to the project folder now. Run `dir` and confirm you can see files like `launch.py` and `recorder26.py` in the list.

---

## Step 4 — Check your GPU

The Whisper speech-to-text model runs faster on a dedicated graphics card (GPU). Check what you have:

1. Open Task Manager from Windows search
2. Click the **Performance** tab
3. Look at the left column — you will see CPU, Memory, Disk, and GPU entries
4. Click on a **GPU** entry and note the brand name shown

**faster-whisper works for everyone** and is the recommended choice — it requires no extra setup and works well on both CPU and GPU.

If you have an NVIDIA GPU and want better transcription quality, you can use **Standard** Whisper instead — but it requires ffmpeg to be installed separately, which is extra work. If Standard Whisper fails, ffmpeg is almost certainly the reason. Instructions for installing ffmpeg on Windows are widely available by searching "install ffmpeg Windows".

Write down your GPU brand — you will need it in Step 9.

---

## Step 5 — Get API keys

API keys are like passwords that let the tool use AI services. You need at least one.

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

Make sure you are in the project folder (run `pwd` to check), then:
```
notepad .env
```

You will see lines like:
```
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
```

Replace `your-key-here` with your actual keys. Make sure to leave quotation marks, they are needed for the key.Leave lines for services you do not have — they will simply not activate.

Save with **Ctrl + S** and close Notepad.

**Verify it saved:** run `Get-Content .env` in PowerShell — you should see your keys. If nothing appears, Notepad saved it as `.env.txt`. Fix this: in Notepad go to **File → Save As**, set "Save as type" to **All Files**, name it `.env` exactly, and save again.

---

## Step 7 — Install Python packages

Make sure you are in the project folder, then run:
```
pip install flask flask-cors openai anthropic google-genai python-dotenv httpx openpyxl sounddevice scipy numpy psutil faster-whisper openai-whisper
```

This installs everything needed. It may take several minutes.

> **If `pip` is not found:** Try `python -m pip install ...` with the same package list.

> **If you get a "Microsoft Visual C++ required" error:**
> 1. Download [aka.ms/vs/17/release/vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)
> 2. Run it, select **Desktop development with C++**, install
> 3. Re-run the pip command

---

## Step 8 — Run the tool

Make sure you are in the project folder (check using pwd), then:
```
python launch.py
```

The script starts the server. After **about 30 seconds**, your browser opens automatically to the settings page. Nothing appearing right away is completely normal — wait for it.

If the browser does not open automatically, go to `http://localhost:5001/settings` yourself.

If Windows asks whether to allow Python through the firewall, click **Allow access**.

> **Keep this terminal window open** the entire time you use the tool. The server runs inside it — if you close it, the tool stops immediately and the browser will show an error. If that happens accidentally, just run `python launch.py` again and your session data will still be there.

> **Things generally take longer than you might expect.** The server takes time to start. After you start a session, loading the Whisper model on first run takes another minute or two. Subsequent starts are faster. Be patient throughout.

---

## Step 9 — Configure and start a session

The settings page is open in your browser. Work through the options:

**Language** — Finnish or English, matching what will be spoken

**Sound input** — Mono is correct for almost all microphones

**Analysis frequency** — How often audio is transcribed and analysed. 5 min recommended.

**Whisper model** — Based on your GPU from Step 4:
- NVIDIA GPU → **Standard**
- AMD / Intel integrated → **faster-whisper**

**Whisper model size** — Medium is the best balance of speed and accuracy. Use Small if your computer is slow or old.

**AI model** —  Select **Claude** or **Gemini** based on which key you added. If you select a local model (phi, gemma), new options appear below — these run fully on your computer with no internet, but require [Ollama](https://ollama.com) installed separately and are slower and less accurate than cloud models.

**Analysis window** — Only visible for cloud AI models. 

**End time** — Optional. The session stops automatically at this time.

Click **Start session**. The browser switches to the main view. The red **REC** indicator in the top left confirms recording is active.

> **First session:** The Whisper model downloads the first time (hundreds of MB to over 1 GB). This takes a few extra minutes. Subsequent starts load from disk.

---

## Where your data is saved

All session data is saved automatically inside the project folder under `sessions\`. Each session gets its own timestamped subfolder (e.g. `sessions\2025-05-26_14-30-00\`) containing:
- `chunks\` — raw audio recordings
- `transcripts\` — text transcriptions of each chunk
- `analysis\` — AI analysis results (gaps, assumptions, synthesis)

Sessions are never deleted automatically. You can open these folders in File Explorer to access or back up your data.

---

## Stopping a session

You can stop in two ways:
- **In the browser:** click the gear icon ⚙, then click **Stop session**
- **In PowerShell:** press **Ctrl + C**

Stopping from the browser is cleaner — it lets the tool finish processing the last audio chunk before shutting down.

---

## Troubleshooting

**Browser shows an error when opening localhost** — The server may still be starting. Wait 30–60 seconds and refresh. Check PowerShell for error messages.

**"No module named X"** — A package is missing. Run `pip install X`.

**"All models failed" in the analysis** — An API key is wrong or has no funds. Run `Get-Content .env` to check the keys.

**Analysis results missing or timestamps have gaps** — An AI call may have failed silently. The browser does not show an error — check the PowerShell window for lines containing "model failed" or "failed:".

**No transcript appears / microphone not working:**
- Go to **Windows Settings → Privacy & Security → Microphone**
- Turn on **Microphone access** and **Let desktop apps access your microphone**
- Right-click the speaker icon in the taskbar → **Sound settings** → confirm the correct microphone is set as default input

**faster-whisper is very slow** — Without a dedicated GPU, transcribing 5 minutes of audio can take 3–5 minutes. This is expected. Use the Small model size or increase Analysis frequency to 10 min.

**Port 5001 already in use** — Restart your computer, or change `port=5001` at the bottom of `web/app26.py`.

**Antivirus blocking the tool** — Add an exception for your project folder in your antivirus settings.
