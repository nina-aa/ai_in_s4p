import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "web" / "app26.py"
    proc = subprocess.Popen([sys.executable, str(app_path)])
    time.sleep(1.5)
    webbrowser.open("http://localhost:5001/settings")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
