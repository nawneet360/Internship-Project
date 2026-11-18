"""
run.py
------
Local development entry point.

Usage:
    cd backend
    python run.py
"""

import os
os.environ.setdefault("FLASK_ENV", "development")

from app import create_app
from config import get_config

cfg = get_config()
app = create_app()

if __name__ == "__main__":
    app.run(
        host  = cfg().HOST,
        port  = cfg().PORT,
        debug = cfg().DEBUG,
        use_reloader = False,   # reloader conflicts with model pre-loading
    )
