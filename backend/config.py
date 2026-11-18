"""
config.py
---------
All environment-aware settings live here.
Never hardcode secrets — always read from environment variables.

Usage:
    from config import get_config
    cfg = get_config()       # returns DevelopmentConfig or ProductionConfig
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # …/backend
ROOT_DIR = BASE_DIR.parent                          # …/Mini Project
MODELS_DIR = ROOT_DIR / "models"
EXAMPLES_DIR = ROOT_DIR / "examples"
LOGS_DIR = BASE_DIR / "logs"

# Create logs dir at import time so handlers never fail
LOGS_DIR.mkdir(exist_ok=True)


class BaseConfig:
    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG: bool = False
    TESTING: bool = False

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins; "*" means allow all
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # ── Server ─────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8080))

    # ── Paths ──────────────────────────────────────────────────────────────────
    MODELS_DIR: Path = MODELS_DIR
    EXAMPLES_DIR: Path = EXAMPLES_DIR
    LOGS_DIR: Path = LOGS_DIR
    DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "disasterai.db")))

    # ── Authentication & JWT ───────────────────────────────────────────────────
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 30))

    # ── Model settings ─────────────────────────────────────────────────────────
    # Number of synthetic samples used to train if no saved model is found
    TRAIN_SAMPLES: int = int(os.getenv("TRAIN_SAMPLES", 1500))
    # Whether to try loading pkl files from models/ before training
    LOAD_SAVED_MODELS: bool = os.getenv("LOAD_SAVED_MODELS", "true").lower() == "true"

    # ── Prediction history ─────────────────────────────────────────────────────
    # In-memory cap; not persisted across restarts
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", 100))

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(BaseConfig):
    DEBUG = False
    LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
    # In production always set SECRET_KEY via env var
    SECRET_KEY = os.getenv("SECRET_KEY", "PLEASE-SET-ME")


class TestingConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    LOAD_SAVED_MODELS = False
    TRAIN_SAMPLES = 200          # tiny dataset for fast tests


_CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
}


def get_config() -> BaseConfig:
    """Return the config class matching the FLASK_ENV environment variable."""
    env = os.getenv("FLASK_ENV", "development").lower()
    return _CONFIG_MAP.get(env, DevelopmentConfig)
