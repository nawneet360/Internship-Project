"""
web_server.py
=============
Root-level entry point delegating to the modular backend in backend/.
Maintained for 100% backward compatibility with existing commands, scripts, and README.
"""

import os
import sys
from pathlib import Path

# Add backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("FLASK_ENV", "development")

from app import create_app
from config import get_config

cfg = get_config()()
app = create_app()

if __name__ == "__main__":
    app.run(
        host=cfg.HOST,
        port=cfg.PORT,
        debug=cfg.DEBUG,
        use_reloader=False,
    )
