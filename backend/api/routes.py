"""
api/routes.py
-------------
All REST endpoints registered on a Blueprint called `api`.

Endpoints
─────────
GET  /api/health                      → model status + uptime
GET  /api/status                      → alias kept for legacy JS
POST /api/predict/flood               → flood risk prediction
POST /api/predict/earthquake          → earthquake risk prediction
POST /api/predict/wildfire            → wildfire risk prediction
POST /api/predict/storm               → storm risk prediction
GET  /api/history                     → last N predictions (all types)
GET  /api/history/<disaster_type>     → last N predictions for one type
DELETE /api/history                   → clear history
GET  /api/multi-hazard/<region>       → combined regional snapshot
GET  /api/sample-inputs/<disaster>    → example inputs for the UI
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, current_app, g, jsonify, request

from api.validators import (
    validate_earthquake,
    validate_flood,
    validate_storm,
    validate_wildfire,
)
from core import database as db
from core import model_manager as mm
from core.auth import jwt_optional

log = logging.getLogger("disasterai.api")

api = Blueprint("api", __name__, url_prefix="/api")

# ── Prediction history (in-memory ring buffer) ─────────────────────────────────
_history: deque[dict] = deque(maxlen=200)   # trimmed per-request by MAX_HISTORY

_START_TIME = time.time()


# ── Utilities ──────────────────────────────────────────────────────────────────

def _error(message: str, details: Any = None, code: int = 400):
    body = {"error": message}
    if details:
        body["details"] = details
    return jsonify(body), code


def _ok(data: dict, code: int = 200):
    return jsonify(data), code


def _record(disaster: str, inputs: dict, result: dict) -> None:
    """Append one prediction to the history ring buffer and save to database."""
    max_h = current_app.config.get("MAX_HISTORY", 100)
    user_id = getattr(g, "current_user", {}).get("id") if getattr(g, "current_user", None) else None
    pred_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    entry = {
        "id":        pred_id,
        "user_id":   user_id,
        "type":      disaster,
        "timestamp": ts,
        "inputs":    inputs,
        "result":    result,
    }
    _history.appendleft(entry)
    while len(_history) > max_h:
        _history.pop()

    try:
        risk_score = (
            result.get("risk_score")
            or result.get("probability")
            or result.get("fire_probability")
            or result.get("storm_probability")
            or 0.0
        )
        risk_level = result.get("risk_level", "Unknown")
        location = result.get("location")
        db.save_prediction(
            pred_id=pred_id,
            user_id=user_id,
            disaster_type=disaster,
            risk_level=risk_level,
            risk_score=float(risk_score),
            location=location,
            inputs=inputs,
            result=result,
            created_at=ts,
        )
    except Exception as exc:
        log.warning("Could not persist prediction to database: %s", exc)


def _require_model(name: str):
    """Decorator: return 503 if the named model isn't loaded yet."""
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if mm.get_model(name) is None:
                return _error(f"{name.title()} model is not available yet", code=503)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def _get_json() -> tuple[dict, None] | tuple[None, Any]:
    """Return (data, None) on success or (None, error_response) on bad JSON."""
    data = request.get_json(silent=True)
    if data is None:
        return None, _error("Request body must be valid JSON", code=400)
    if not isinstance(data, dict):
        return None, _error("JSON body must be an object/dict", code=400)
    return data, None


# ── Health / status ────────────────────────────────────────────────────────────

@api.get("/health")
def health():
    """Detailed health check — used by Render's health-check URL."""
    s = mm.status()
    uptime = round(time.time() - _START_TIME)
    s["uptime_seconds"] = uptime
    s["uptime_human"] = _fmt_uptime(uptime)
    code = 200 if s["status"] == "ok" else 207          # 207 = partial content
    return jsonify(s), code


@api.get("/status")
def status():
    """Legacy endpoint kept for existing frontend JS."""
    s = mm.status()
    return jsonify({
        "status":     s["status"],
        "flood":      s["models"]["flood"],
        "earthquake": s["models"]["earthquake"],
        "wildfire":   s["models"]["wildfire"],
        "storm":      s["models"]["storm"],
    })


def _fmt_uptime(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


# ── Prediction endpoints ───────────────────────────────────────────────────────

@api.post("/predict/flood")
@_require_model("flood")
@jwt_optional
def predict_flood():
    """
    Flood risk prediction.

    Required  : rainfall_mm, river_level_m
    Optional  : temperature_c, humidity_pct, elevation_m, slope_degrees,
                soil_permeability, rainfall_7day_avg, location
    """
    data, err = _get_json()
    if err:
        return err

    cleaned, errors = validate_flood(data)
    if errors:
        return _error("Validation failed", errors, 400)

    try:
        result = mm.predict_flood(cleaned)
        _record("flood", data, result)
        log.info("flood  risk=%s loc=%s", result["risk_level"], result.get("location"))
        return _ok(result)
    except Exception as exc:
        log.exception("Flood prediction error")
        return _error("Prediction failed", str(exc), 500)


@api.post("/predict/earthquake")
@_require_model("earthquake")
@jwt_optional
def predict_earthquake():
    """
    Earthquake risk prediction.

    Required  : micro_tremors, fault_distance_km, tectonic_stress
    Optional  : p_wave_velocity, s_wave_velocity, rock_density,
                crustal_thickness, ground_deformation, recent_activity,
                max_historical_mag, groundwater_change, radon_emission, location
    """
    data, err = _get_json()
    if err:
        return err

    cleaned, errors = validate_earthquake(data)
    if errors:
        return _error("Validation failed", errors, 400)

    try:
        result = mm.predict_earthquake(cleaned)
        _record("earthquake", data, result)
        log.info("earthquake risk=%s prob=%.2f", result["risk_level"], result["probability"])
        return _ok(result)
    except Exception as exc:
        log.exception("Earthquake prediction error")
        return _error("Prediction failed", str(exc), 500)


@api.post("/predict/wildfire")
@_require_model("wildfire")
@jwt_optional
def predict_wildfire():
    """
    Wildfire risk prediction.

    Required  : temperature_c, humidity_pct, wind_speed_kmh, vegetation_dryness
    Optional  : elevation_m, slope_degrees, recent_fires, fuel_load, location
    """
    data, err = _get_json()
    if err:
        return err

    cleaned, errors = validate_wildfire(data)
    if errors:
        return _error("Validation failed", errors, 400)

    try:
        result = mm.predict_wildfire(cleaned)
        _record("wildfire", data, result)
        log.info("wildfire risk=%s prob=%.2f", result["risk_level"], result["fire_probability"])
        return _ok(result)
    except Exception as exc:
        log.exception("Wildfire prediction error")
        return _error("Prediction failed", str(exc), 500)


@api.post("/predict/storm")
@_require_model("storm")
@jwt_optional
def predict_storm():
    """
    Storm/cyclone risk prediction.

    Required  : wind_speed_kmh, central_pressure_hpa, sea_surface_temp_c
    Optional  : coastal_slope_degrees, sea_elevation_m, latitude_degrees,
                recent_storms, ocean_current_ms, location
    """
    data, err = _get_json()
    if err:
        return err

    cleaned, errors = validate_storm(data)
    if errors:
        return _error("Validation failed", errors, 400)

    try:
        result = mm.predict_storm(cleaned)
        _record("storm", data, result)
        log.info("storm  risk=%s prob=%.2f", result["risk_level"], result["storm_probability"])
        return _ok(result)
    except Exception as exc:
        log.exception("Storm prediction error")
        return _error("Prediction failed", str(exc), 500)


# ── History endpoints ──────────────────────────────────────────────────────────

@api.get("/history")
def get_history():
    """Return the last N predictions across all disaster types."""
    n = min(int(request.args.get("n", 20)), 100)
    return _ok({"count": min(n, len(_history)), "predictions": list(_history)[:n]})


@api.get("/history/<disaster_type>")
def get_history_by_type(disaster_type: str):
    """Return the last N predictions for a specific disaster type."""
    allowed = {"flood", "earthquake", "wildfire", "storm"}
    if disaster_type not in allowed:
        return _error(f"Unknown type. Choose from: {sorted(allowed)}", code=404)
    n = min(int(request.args.get("n", 20)), 100)
    filtered = [p for p in _history if p["type"] == disaster_type]
    return _ok({"count": min(n, len(filtered)), "predictions": filtered[:n]})


@api.delete("/history")
def clear_history():
    """Clear prediction history."""
    _history.clear()
    return _ok({"message": "History cleared"})


# ── Multi-hazard regional snapshot ─────────────────────────────────────────────

_REGIONAL_PROFILES: dict[str, dict] = {
    "coastal": {
        "label": "Coastal Lowland",
        "flood": dict(rainfall_mm=55, river_level_m=7.5, temperature_c=16,
                      humidity_pct=95, elevation_m=5, slope_degrees=1,
                      soil_permeability=0.15, rainfall_7day_avg=40),
        "earthquake": dict(p_wave_velocity=4.9, s_wave_velocity=2.8, micro_tremors=9.5,
                           fault_distance_km=3, rock_density=2.9, crustal_thickness=20,
                           tectonic_stress=16, ground_deformation=12, recent_activity=4,
                           max_historical_mag=7.5, groundwater_change=-5, radon_emission=52),
        "wildfire": dict(temperature_c=20, humidity_pct=80, wind_speed_kmh=20,
                         vegetation_dryness=0.3, elevation_m=10, slope_degrees=2,
                         recent_fires=0, fuel_load=2),
        "storm": dict(wind_speed_kmh=70, central_pressure_hpa=975, sea_surface_temp_c=28,
                      coastal_slope_degrees=1.5, sea_elevation_m=100, latitude_degrees=20,
                      recent_storms=1, ocean_current_ms=0.8),
    },
    "pacific": {
        "label": "Pacific Coast",
        "flood": dict(rainfall_mm=35, river_level_m=6, temperature_c=18,
                      humidity_pct=80, elevation_m=30, slope_degrees=8,
                      soil_permeability=0.4, rainfall_7day_avg=25),
        "earthquake": dict(p_wave_velocity=5.3, s_wave_velocity=3.1, micro_tremors=6.2,
                           fault_distance_km=8, rock_density=2.8, crustal_thickness=28,
                           tectonic_stress=12, ground_deformation=6, recent_activity=2,
                           max_historical_mag=6.8, groundwater_change=-2, radon_emission=38),
        "wildfire": dict(temperature_c=35, humidity_pct=18, wind_speed_kmh=40,
                         vegetation_dryness=0.85, elevation_m=600, slope_degrees=22,
                         recent_fires=2, fuel_load=8),
        "storm": dict(wind_speed_kmh=35, central_pressure_hpa=1005, sea_surface_temp_c=22,
                      coastal_slope_degrees=4, sea_elevation_m=800, latitude_degrees=38,
                      recent_storms=0, ocean_current_ms=-0.5),
    },
    "midwest": {
        "label": "Midwest River Valley",
        "flood": dict(rainfall_mm=65, river_level_m=8, temperature_c=22,
                      humidity_pct=90, elevation_m=15, slope_degrees=2,
                      soil_permeability=0.2, rainfall_7day_avg=45),
        "earthquake": dict(p_wave_velocity=6.5, s_wave_velocity=3.8, micro_tremors=0.3,
                           fault_distance_km=180, rock_density=2.6, crustal_thickness=42,
                           tectonic_stress=1, ground_deformation=0.2, recent_activity=0,
                           max_historical_mag=2.1, groundwater_change=0.1, radon_emission=10),
        "wildfire": dict(temperature_c=28, humidity_pct=55, wind_speed_kmh=25,
                         vegetation_dryness=0.5, elevation_m=200, slope_degrees=5,
                         recent_fires=0, fuel_load=3),
        "storm": dict(wind_speed_kmh=25, central_pressure_hpa=1010, sea_surface_temp_c=20,
                      coastal_slope_degrees=8, sea_elevation_m=2000, latitude_degrees=42,
                      recent_storms=0, ocean_current_ms=0),
    },
    "mountain": {
        "label": "Mountain Desert",
        "flood": dict(rainfall_mm=8, river_level_m=3, temperature_c=28,
                      humidity_pct=35, elevation_m=200, slope_degrees=20,
                      soil_permeability=0.9, rainfall_7day_avg=5),
        "earthquake": dict(p_wave_velocity=5.8, s_wave_velocity=3.4, micro_tremors=2.8,
                           fault_distance_km=35, rock_density=2.7, crustal_thickness=35,
                           tectonic_stress=6, ground_deformation=2, recent_activity=1,
                           max_historical_mag=4.5, groundwater_change=-0.8, radon_emission=22),
        "wildfire": dict(temperature_c=42, humidity_pct=10, wind_speed_kmh=50,
                         vegetation_dryness=0.92, elevation_m=1500, slope_degrees=30,
                         recent_fires=3, fuel_load=9),
        "storm": dict(wind_speed_kmh=20, central_pressure_hpa=1015, sea_surface_temp_c=15,
                      coastal_slope_degrees=10, sea_elevation_m=5000, latitude_degrees=35,
                      recent_storms=0, ocean_current_ms=0),
    },
}


@api.get("/multi-hazard/<region>")
def multi_hazard(region: str):
    """
    Pre-loaded regional profiles — combined risk snapshot.
    Available regions: coastal | pacific | midwest | mountain
    """
    if region not in _REGIONAL_PROFILES:
        return _error(
            f'Unknown region "{region}"',
            {"available": list(_REGIONAL_PROFILES)},
            404,
        )

    p = _REGIONAL_PROFILES[region]
    out: dict[str, Any] = {"region": p["label"]}

    if mm.get_model("flood"):
        fa = mm.predict_flood(dict(p["flood"]))
        out["flood"] = {k: fa[k] for k in ("risk_score", "risk_level", "message")}

    if mm.get_model("earthquake"):
        ea = mm.predict_earthquake(dict(p["earthquake"]))
        out["earthquake"] = {k: ea[k]
                             for k in ("probability", "predicted_magnitude",
                                       "risk_level", "message")}

    if mm.get_model("wildfire"):
        wa = mm.predict_wildfire(dict(p["wildfire"]))
        out["wildfire"] = {k: wa[k]
                           for k in ("fire_probability", "predicted_intensity",
                                     "risk_level", "message")}

    if mm.get_model("storm"):
        sa = mm.predict_storm(dict(p["storm"]))
        out["storm"] = {k: sa[k]
                        for k in ("storm_probability", "predicted_intensity",
                                  "risk_level", "message")}

    # Combined composite risk score (average of available scores)
    scores = []
    if "flood"      in out: scores.append(out["flood"]["risk_score"])
    if "earthquake" in out: scores.append(out["earthquake"]["probability"])
    if "wildfire"   in out: scores.append(out["wildfire"]["fire_probability"])
    if "storm"      in out: scores.append(out["storm"]["storm_probability"])

    if scores:
        combined = sum(scores) / len(scores)
        out["combined"] = {
            "risk_score": round(combined, 3),
            "risk_level": (
                "Extreme"  if combined >= 0.70 else
                "High"     if combined >= 0.50 else
                "Moderate" if combined >= 0.30 else
                "Low"
            ),
        }

    return _ok(out)


# ── Sample inputs (useful for Swagger / frontend dev) ──────────────────────────

_SAMPLES: dict[str, dict] = {
    "flood": {
        "rainfall_mm":       120,
        "river_level_m":     7.5,
        "temperature_c":     28,
        "humidity_pct":      90,
        "elevation_m":       10,
        "slope_degrees":     2,
        "soil_permeability": 0.2,
        "rainfall_7day_avg": 80,
        "location":          "Mumbai, India",
    },
    "earthquake": {
        "p_wave_velocity":    5.1,
        "s_wave_velocity":    2.9,
        "micro_tremors":      9.2,
        "fault_distance_km":  4,
        "rock_density":       2.95,
        "crustal_thickness":  22,
        "tectonic_stress":    14,
        "ground_deformation": 9.5,
        "recent_activity":    4,
        "max_historical_mag": 7.9,
        "groundwater_change": -5.2,
        "radon_emission":     58,
        "location":           "Tokyo, Japan",
    },
    "wildfire": {
        "temperature_c":      38,
        "humidity_pct":       12,
        "wind_speed_kmh":     35,
        "vegetation_dryness": 0.88,
        "elevation_m":        600,
        "slope_degrees":      22,
        "recent_fires":       2,
        "fuel_load":          8.5,
        "location":           "California, USA",
    },
    "storm": {
        "wind_speed_kmh":        110,
        "central_pressure_hpa":  920,
        "sea_surface_temp_c":    31,
        "coastal_slope_degrees": 2.5,
        "sea_elevation_m":       500,
        "latitude_degrees":      15,
        "recent_storms":         2,
        "ocean_current_ms":      2.0,
        "location":              "Western Pacific",
    },
}


@api.get("/sample-inputs/<disaster>")
def sample_inputs(disaster: str):
    if disaster not in _SAMPLES:
        return _error(
            f'Unknown disaster "{disaster}"',
            {"available": list(_SAMPLES)},
            404,
        )
    return _ok({"disaster": disaster, "sample": _SAMPLES[disaster]})
