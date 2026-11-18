"""
app.py
------
Flask application factory.

Usage
-----
  Development:   python run.py
  Production:    gunicorn "app:create_app()" --bind 0.0.0.0:$PORT --workers 2 --timeout 120
  Testing:       FLASK_ENV=testing pytest
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS


def _setup_logging(cfg) -> None:
    """Configure root logger: console + optional rotating file."""
    level = getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO)
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if cfg.LOG_TO_FILE:
        log_file = cfg.LOGS_DIR / "disasterai.log"
        rh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        rh.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(rh)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    # Silence noisy third-party loggers in production
    if not cfg.DEBUG:
        for noisy in ("urllib3", "sklearn", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def create_app(config_object=None) -> Flask:
    """
    Application factory.
    Pass a config object explicitly (useful in tests) or let it
    auto-detect via the FLASK_ENV environment variable.
    """
    # ── Config ────────────────────────────────────────────────────────────────
    if config_object is None:
        from config import get_config
        config_object = get_config()

    cfg = config_object()

    # ── Logging ───────────────────────────────────────────────────────────────
    _setup_logging(cfg)
    log = logging.getLogger("disasterai.app")

    # ── Flask app ─────────────────────────────────────────────────────────────
    # Serve the HTML dashboard from the project root (one level up from backend/)
    static_folder = str(cfg.MODELS_DIR.parent)   # …/Mini Project/
    app = Flask(__name__, static_folder=static_folder, static_url_path="")
    app.config.from_object(cfg)

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = cfg.CORS_ORIGINS
    CORS(app, resources={r"/api/*": {"origins": origins}})
    log.info("CORS enabled for origins: %s", origins)

    # ── Database ──────────────────────────────────────────────────────────────
    from core import database as db
    db.init_db(cfg.DATABASE_PATH)

    # ── Blueprints ────────────────────────────────────────────────────────────
    from api.routes import api as api_blueprint
    from api.auth_routes import auth_bp
    app.register_blueprint(api_blueprint)
    app.register_blueprint(auth_bp)

    # ── Static file serving ───────────────────────────────────────────────────
    root_dir = Path(static_folder)

    @app.get("/")
    def index():
        """Serve the main HTML dashboard."""
        return send_from_directory(str(root_dir), "disaster_prediction_system.html")

    @app.get("/login")
    @app.get("/register")
    def auth_page():
        """Serve the authentication page."""
        login_html = root_dir / "login.html"
        if login_html.exists():
            return send_from_directory(str(root_dir), "login.html")
        return send_from_directory(str(root_dir), "disaster_prediction_system.html")

    @app.get("/disaster_prediction_system.css")
    def serve_css():
        return send_from_directory(str(root_dir), "disaster_prediction_system.css",
                                   mimetype="text/css")

    @app.get("/disaster_prediction_system.js")
    def serve_js():
        return send_from_directory(str(root_dir), "disaster_prediction_system.js",
                                   mimetype="application/javascript")

    # ── Global error handlers ─────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "path": str(e)}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def server_error(e):
        log.exception("Unhandled server error")
        return jsonify({"error": "Internal server error"}), 500

    # ── Boot ML models ────────────────────────────────────────────────────────
    # Import here to avoid circular imports
    from core import model_manager as mm

    with app.app_context():
        log.info("=" * 60)
        log.info("  🚀  DisasterAI Pro — starting up")
        log.info("=" * 60)
        print("\n" + "=" * 60)
        print("  🚀  DisasterAI Pro — starting models")
        print("=" * 60)

        mm.initialise(
            models_dir  = cfg.MODELS_DIR,
            examples_dir= cfg.EXAMPLES_DIR,
            n_samples   = cfg.TRAIN_SAMPLES,
            load_saved  = cfg.LOAD_SAVED_MODELS,
        )

        status = mm.status()
        all_ok = status["status"] == "ok"
        icon   = "✅" if all_ok else "⚠️ "
        print(f"\n  {icon}  {'All models ready' if all_ok else 'Partial start — check logs'}")
        print(f"  🌐  API running at  http://{cfg.HOST}:{cfg.PORT}/api/health")
        print("=" * 60 + "\n")
        log.info("Boot complete — status=%s", status["status"])

    return app
