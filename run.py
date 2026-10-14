"""
Application Runner Script
Launches the FastAPI server with Uvicorn.
Accessible locally and across local network / Wi-Fi.
"""

import sys
import socket
import uvicorn
import webbrowser
import threading
import time

def get_local_ip():
    """Detects local LAN/Wi-Fi IPv4 address for sharing."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == '__main__':
    local_ip = get_local_ip()
    print("=" * 72)
    print("  🫀 TRACK A: CARDIOVASCULAR RISK VISUALIZATION & PREDICTION")
    print("=" * 72)
    print(f"  👉 Local Laptop URL:      http://127.0.0.1:8000")
    print(f"  👉 Network Sharing URL:   http://{local_ip}:8000")
    print("=" * 72)
    print("  Share the 'Network Sharing URL' with anyone on your Wi-Fi or Mobile Hotspot!")
    print("  Clinical Notice: For educational & decision-support purposes only.")
    print("=" * 72)

    # Launch browser automatically if not running in headless mode
    if "--no-browser" not in sys.argv:
        threading.Thread(target=open_browser, daemon=True).start()

    # host="0.0.0.0" allows other devices on the same Wi-Fi / Hotspot to connect
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
