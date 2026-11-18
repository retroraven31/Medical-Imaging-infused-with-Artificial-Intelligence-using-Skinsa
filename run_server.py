"""
Server launcher for skinsa web application.
Run with: python run_server.py
"""

import os
import sys
import subprocess
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          skinsa — Skin Lesion Classifier                     ║
║   Web UI + FastAPI Backend                                   ║
╠══════════════════════════════════════════════════════════════╣
║  Frontend: http://localhost:8000                             ║
║  API Docs: http://localhost:8000/docs                        ║
╚══════════════════════════════════════════════════════════════╝

⚠️  NOTE: Model uses randomly-initialized weights.
   For real predictions, train on HAM10000 and reload model weights.
""")

    try:
        import uvicorn
        uvicorn.run(
            "skinsa.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
