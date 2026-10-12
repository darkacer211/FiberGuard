"""
FiberGuard - Startup Launcher
Starts the FastAPI backend with Uvicorn and launches the web browser.
"""

import sys
import os
import time
import webbrowser
import threading
import uvicorn

def open_browser():
    time.sleep(1.2)
    url = "http://127.0.0.1:8000"
    print(f"\n[FiberGuard] Opening browser at {url} ...")
    webbrowser.open(url)

def main():
    print("=" * 70)
    print("   FIBERGUARD: AI FAULT DETECTION FOR OPTICAL FIBER NETWORKS")
    print("   Standard: ITU-T G.652.D / G.657 | Band: C-Band (1550 nm)")
    print("   ML Engine: Scikit-learn (Random Forest, Gradient Boosting, kNN, MLP)")
    print("=" * 70)

    # Launch browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn server
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()
