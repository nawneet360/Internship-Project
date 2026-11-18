"""
core/model_manager.py
---------------------
Single source of truth for all ML model lifecycle:
  • Load saved pkl files  →  instant startup
  • Train from scratch    →  fallback when no pkl exists
  • Thread-safe access via a simple module-level registry
  • Predict helpers called by route handlers
"""

from __future__ import annotations

import logging
import pickle
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("disasterai.model_manager")

# ── Global model registry (populated by initialise()) ─────────────────────────
_lock = threading.Lock()
_registry: dict[str, Any] = {
    "flood":      None,
    "earthquake": None,
    "wildfire":   None,
    "storm":      None,
}
_training_errors: dict[str, str] = {}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _add_examples_to_path(examples_dir: Path) -> None:
    """Ensure the examples/ folder (ML classes) is importable."""
    s = str(examples_dir)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_pkl(path: Path) -> Any | None:
    """Try to load a pickle file; return None on any error."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if getattr(obj, "is_trained", False):
            log.info("Loaded saved model: %s", path.name)
            return obj
        log.warning("pkl exists but is_trained=False: %s", path.name)
        return None
    except Exception as exc:
        log.warning("Could not load %s: %s", path.name, exc)
        return None


def _train_flood(examples_dir: Path, n_samples: int) -> Any:
    _add_examples_to_path(examples_dir)
    from flood_prediction import FloodPredictionSystem  # type: ignore
    sys = FloodPredictionSystem()
    data = sys.generate_sample_data(n_samples)
    sys.train_model(data)
    return sys


def _train_earthquake(examples_dir: Path, n_samples: int) -> Any:
    _add_examples_to_path(examples_dir)
    from earthquake_prediction import EarthquakePredictionSystem  # type: ignore
    sys = EarthquakePredictionSystem()
    data = sys.generate_sample_data(n_samples)
    sys.train_models(data)
    return sys


def _train_wildfire(examples_dir: Path, n_samples: int) -> Any:
    _add_examples_to_path(examples_dir)
    from wildfire_prediction import WildfirePredictionSystem  # type: ignore
    sys = WildfirePredictionSystem()
    data = sys.generate_sample_data(n_samples)
    sys.train_models(data)
    return sys


def _train_storm(examples_dir: Path, n_samples: int) -> Any:
    _add_examples_to_path(examples_dir)
    from storm_prediction import StormPredictionSystem  # type: ignore
    sys = StormPredictionSystem()
    data = sys.generate_sample_data(n_samples)
    sys.train_models(data)
    return sys


_TRAINERS = {
    "flood":      _train_flood,
    "earthquake": _train_earthquake,
    "wildfire":   _train_wildfire,
    "storm":      _train_storm,
}

_PKL_NAMES = {
    "flood":      "flood_model.pkl",
    "earthquake": "earthquake_model.pkl",
    "wildfire":   "wildfire_model.pkl",
    "storm":      "storm_model.pkl",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def initialise(models_dir: Path, examples_dir: Path,
               n_samples: int = 1500, load_saved: bool = True) -> None:
    """
    Boot all four models.
    Called once at application startup (from app.py).

    Strategy:
      1. If load_saved=True  →  try loading the .pkl file from models/
      2. If pkl missing/stale →  train fresh on synthetic data
    """
    _add_examples_to_path(examples_dir)

    emojis = {
        "flood": "🌊", "earthquake": "🌍",
        "wildfire": "🔥", "storm":  "🌪️ ",
    }

    for name, trainer in _TRAINERS.items():
        emoji = emojis[name]
        t0 = time.perf_counter()

        # ── 1. Try loading a saved model ──────────────────────────────────────
        if load_saved:
            pkl = models_dir / _PKL_NAMES[name]
            obj = _load_pkl(pkl)
            if obj is not None:
                with _lock:
                    _registry[name] = obj
                elapsed = time.perf_counter() - t0
                log.info("%s %s — loaded from disk  (%.2f s)", emoji, name, elapsed)
                print(f"  {emoji} {name:<12} ✅  loaded from {pkl.name}  ({elapsed:.2f}s)")
                continue

        # ── 2. Train from scratch ─────────────────────────────────────────────
        log.info("%s %s — training on %d synthetic samples…", emoji, name, n_samples)
        print(f"  {emoji} {name:<12} ⏳  training on {n_samples:,} synthetic samples…",
              end="", flush=True)
        try:
            obj = trainer(examples_dir, n_samples)
            with _lock:
                _registry[name] = obj
            elapsed = time.perf_counter() - t0
            log.info("%s %s — ready  (%.2f s)", emoji, name, elapsed)
            print(f" ✅  ({elapsed:.2f}s)")
        except Exception as exc:
            _training_errors[name] = str(exc)
            log.error("%s %s — FAILED: %s", emoji, name, exc)
            print(f" ❌  {exc}")


def get_model(name: str) -> Any | None:
    with _lock:
        return _registry.get(name)


def status() -> dict:
    """Return a status dict for the /api/health endpoint."""
    with _lock:
        ready = {k: v is not None for k, v in _registry.items()}
    return {
        "status":  "ok" if all(ready.values()) else "partial",
        "models":  ready,
        "errors":  _training_errors,
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }


# ── Prediction helpers ─────────────────────────────────────────────────────────

def predict_flood(inputs: dict) -> dict:
    model = get_model("flood")
    if model is None:
        raise RuntimeError("Flood model not ready")

    alert = model.generate_alert(inputs, inputs.pop("location", "your area"))
    return {
        "risk_score":  alert["risk_score"],
        "risk_level":  alert["risk_level"],
        "message":     alert["message"],
        "timestamp":   alert["timestamp"],
        "location":    alert["location"],
        "factors": {
            "rainfall_impact": min(100, round(inputs["rainfall_mm"] / 50 * 100)),
            "river_impact":    min(100, max(0, round((inputs["river_level_m"] - 3) / 7 * 100))),
            "terrain_impact":  min(100, max(0, round((1 - inputs["elevation_m"] / 100) * 100))),
            "drainage_impact": min(100, round((1 - inputs["soil_permeability"]) * 100)),
        },
    }


def predict_earthquake(inputs: dict) -> dict:
    model = get_model("earthquake")
    if model is None:
        raise RuntimeError("Earthquake model not ready")

    location = inputs.pop("location", "your area")
    alert = model.generate_earthquake_alert(inputs, location)

    def _tremor(v):
        return "Normal" if v < 2 else ("Elevated" if v < 5 else "High")
    def _stress(v):
        return "Low" if v < 5 else ("Moderate" if v < 10 else "High")
    def _fault(v):
        return "Safe distance" if v > 100 else ("Moderate" if v > 20 else "Very close")
    def _radon(v):
        return "Normal" if v < 30 else ("Elevated" if v < 50 else "High")

    return {
        "probability":          alert["earthquake_probability"],
        "predicted_occurrence": alert["predicted_occurrence"],
        "predicted_magnitude":  alert["predicted_magnitude"],
        "risk_level":           alert["risk_level"],
        "message":              alert["alert_message"],
        "timestamp":            alert["timestamp"],
        "location":             location,
        "indicators": {
            "micro_tremors":   _tremor(inputs["micro_tremors"]),
            "tectonic_stress": _stress(inputs["tectonic_stress"]),
            "fault_proximity": _fault(inputs["fault_distance_km"]),
            "radon_levels":    _radon(inputs["radon_emission"]),
        },
    }


def predict_wildfire(inputs: dict) -> dict:
    model = get_model("wildfire")
    if model is None:
        raise RuntimeError("Wildfire model not ready")

    location = inputs.pop("location", "your area")
    alert = model.generate_wildfire_alert(inputs, location)
    return {
        "fire_probability":     alert["fire_probability"],
        "predicted_occurrence": alert["predicted_occurrence"],
        "predicted_intensity":  alert["predicted_intensity"],
        "risk_level":           alert["risk_level"],
        "message":              alert["alert_message"],
        "timestamp":            alert["timestamp"],
        "location":             location,
        "factors": {
            "temperature_impact": min(100, max(0, round((inputs["temperature_c"] - 15) / 25 * 100))),
            "humidity_impact":    min(100, max(0, round((100 - inputs["humidity_pct"]) / 100 * 100))),
            "wind_impact":        min(100, round(inputs["wind_speed_kmh"] / 50 * 100)),
            "vegetation_impact":  min(100, round(inputs["vegetation_dryness"] * 100)),
            "terrain_impact":     min(100, round(inputs["slope_degrees"] / 40 * 100)),
        },
    }


def predict_storm(inputs: dict) -> dict:
    model = get_model("storm")
    if model is None:
        raise RuntimeError("Storm model not ready")

    location = inputs.pop("location", "your area")
    alert = model.generate_storm_alert(inputs, location)
    return {
        "storm_probability":    alert["storm_probability"],
        "predicted_occurrence": alert["predicted_occurrence"],
        "predicted_intensity":  alert["predicted_intensity"],
        "risk_level":           alert["risk_level"],
        "message":              alert["alert_message"],
        "timestamp":            alert["timestamp"],
        "location":             location,
        "factors": {
            "wind_impact":     min(100, round(max(0, inputs["wind_speed_kmh"] - 20) / 60 * 100)),
            "pressure_impact": min(100, max(0, round((1010 - inputs["central_pressure_hpa"]) / 50 * 100))),
            "temp_impact":     min(100, max(0, round((inputs["sea_surface_temp_c"] - 24) / 6 * 100))),
            "latitude_impact": min(100, round(abs(inputs["latitude_degrees"]) / 40 * 100)),
            "coastal_impact":  min(100, max(0, round((10 - inputs["coastal_slope_degrees"]) / 10 * 100))),
        },
    }
