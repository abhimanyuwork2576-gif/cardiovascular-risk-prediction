"""
Public Link Launcher for Cardiovascular Risk & Stenosis Prediction System
Starts FastAPI server & Cloudflare Tunnel to generate a global HTTPS link.
"""

import os
import sys
import re
import time
import socket
import subprocess
import webbrowser
import urllib.request

# Safe UTF-8 encoding for Windows console
try:
    if sys.stdout:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

CLOUDFLARED_BIN = os.path.join(BASE_DIR, "cloudflared.exe")

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def copy_to_clipboard(text):
    try:
        p = subprocess.Popen(['clip'], stdin=subprocess.PIPE, shell=True)
        p.communicate(text.encode('utf-8'))
        return True
    except Exception:
        return False

def save_links(url):
    # Save text file
    txt_path = os.path.join(BASE_DIR, "ACTIVE_PUBLIC_LINK.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("   CARDIOVASCULAR RISK PREDICTION - PUBLIC ACCESS LINK\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Public URL (Anyone anywhere can open this):\n")
        f.write(f"{url}\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("NOTE: This link works as long as the server script remains open on your computer.\n")
    
    # Save Windows desktop shortcut
    url_path = os.path.join(BASE_DIR, "Open_Public_Website.url")
    with open(url_path, "w", encoding="utf-8") as f:
        f.write("[InternetShortcut]\n")
        f.write(f"URL={url}\n")

def is_server_ready(port=8000):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/model/info", timeout=1.5) as res:
            return res.status == 200
    except Exception:
        return False

def main():
    print("=" * 76)
    print("  CARDIOVASCULAR RISK & VESSEL STENOSIS PREDICTION SYSTEM")
    print("  LAUNCHING APPLICATION & CREATING GLOBAL PUBLIC ACCESS LINK")
    print("=" * 76)

    if not os.path.exists(CLOUDFLARED_BIN):
        print(f"\n[ERROR] cloudflared.exe not found at {CLOUDFLARED_BIN}")
        print("Please ensure cloudflared.exe is in the project root.")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # 1. Start FastAPI server
    print("\n[1/3] Starting backend application server on port 8000...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
        cwd=BASE_DIR
    )

    # Wait for server to be responsive
    print("      Waiting for server to initialize ML models...")
    for _ in range(30):
        if is_server_ready():
            break
        time.sleep(0.5)

    if not is_server_ready():
        print("      [Warning] Server took longer than expected to report ready, proceeding...")

    print("      Backend server is running!")

    # 2. Start Cloudflare Tunnel
    print("\n[2/3] Connecting to Cloudflare Global Edge network to create public URL...")
    tunnel_proc = subprocess.Popen(
        [CLOUDFLARED_BIN, "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=BASE_DIR
    )

    public_url = None
    start_time = time.time()

    # Capture trycloudflare URL from logs
    while time.time() - start_time < 30:
        line = tunnel_proc.stdout.readline()
        if not line:
            break
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if m:
            public_url = m.group(0)
            break

    if not public_url:
        print("\n[ERROR] Could not extract Cloudflare tunnel URL within 30 seconds.")
        print("Ensure you have an active internet connection.")
        server_proc.terminate()
        tunnel_proc.terminate()
        input("\nPress Enter to exit...")
        sys.exit(1)

    # 3. Save files and copy to clipboard
    save_links(public_url)
    copied = copy_to_clipboard(public_url)
    local_ip = get_local_ip()

    # 4. Display Clean Dashboard
    print("\n" + "=" * 76)
    print("  SUCCESS! YOUR PROJECT IS NOW LIVE AND ACCESSIBLE FROM ANYWHERE!")
    print("=" * 76)
    print(f"\n  >> PUBLIC LINK (Share this with ANYONE globally):")
    print(f"     URL: {public_url}")
    if copied:
        print("     [OK] Link has been automatically copied to your clipboard!")
    print("\n  >> LOCAL LAPTOP LINK:")
    print(f"     URL: http://127.0.0.1:8000")
    print("\n  >> LOCAL WI-FI / HOTSPOT LINK:")
    print(f"     URL: http://{local_ip}:8000")
    print("\n" + "-" * 76)
    print("  Shortcuts created: 'Open_Public_Website.url' & 'ACTIVE_PUBLIC_LINK.txt'")
    print("  This link works on mobile phones, tablets, and remote computers worldwide.")
    print("  Keep this window open! The link stays active as long as this runs.")
    print("=" * 76)
    print("\n[Press Ctrl+C at any time to shut down the server and tunnel]")

    # Open in user's browser
    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(public_url)
        except Exception:
            pass

    # Keep alive until interrupted
    try:
        while True:
            time.sleep(1)
            if server_proc.poll() is not None or tunnel_proc.poll() is not None:
                print("\n[Notice] One of the background processes has exited.")
                break
    except KeyboardInterrupt:
        print("\nShutting down server and public tunnel...")
    finally:
        try:
            tunnel_proc.terminate()
        except Exception:
            pass
        try:
            server_proc.terminate()
        except Exception:
            pass
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
