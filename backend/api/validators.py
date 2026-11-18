"""
api/validators.py
-----------------
Lightweight field-level input validation — no external library needed.
Each validator returns (cleaned_dict, errors_dict).
If errors is non-empty the route returns 400 immediately.
"""

from __future__ import annotations
from typing import Any


def _field(
    data: dict,
    key: str,
    typ: type,
    required: bool = True,
    default: Any = None,
    min_val: float | None = None,
    max_val: float | None = None,
    out: dict | None = None,
    errors: dict | None = None,
) -> None:
    """
    Pull one field from `data`, coerce to `typ`, range-check, store in `out`.
    Validation problems are appended to `errors`.
    """
    raw = data.get(key)

    if raw is None:
        if required:
            errors[key] = "required field is missing"
            return
        out[key] = default
        return

    try:
        val = typ(raw)
    except (TypeError, ValueError):
        errors[key] = f"must be a {typ.__name__}"
        return

    if min_val is not None and val < min_val:
        errors[key] = f"must be ≥ {min_val} (got {val})"
        return
    if max_val is not None and val > max_val:
        errors[key] = f"must be ≤ {max_val} (got {val})"
        return

    out[key] = val


# ── Per-disaster validators ────────────────────────────────────────────────────

def validate_flood(data: dict) -> tuple[dict, dict]:
    out, errors = {}, {}
    out["location"] = str(data.get("location", "your area"))[:100]

    _field(data, "rainfall_mm",       float, True,  25,   0,    500,  out, errors)
    _field(data, "river_level_m",     float, True,   5,   0,     20,  out, errors)
    _field(data, "temperature_c",     float, False, 20,  -20,    60,  out, errors)
    _field(data, "humidity_pct",      float, False, 70,   0,    100,  out, errors)
    _field(data, "elevation_m",       float, False, 50,   0,   5000,  out, errors)
    _field(data, "slope_degrees",     float, False,  5,   0,     90,  out, errors)
    _field(data, "soil_permeability", float, False, 0.5,  0,      1,  out, errors)
    _field(data, "rainfall_7day_avg", float, False, 20,   0,    500,  out, errors)
    return out, errors


def validate_earthquake(data: dict) -> tuple[dict, dict]:
    out, errors = {}, {}
    out["location"] = str(data.get("location", "your area"))[:100]

    _field(data, "p_wave_velocity",    float, False, 6.0,  3.0,  10.0, out, errors)
    _field(data, "s_wave_velocity",    float, False, 3.5,  1.5,   6.0, out, errors)
    _field(data, "micro_tremors",      float, True,  2.0,  0,    20.0, out, errors)
    _field(data, "fault_distance_km",  float, True, 50.0,  0,   500,   out, errors)
    _field(data, "rock_density",       float, False, 2.7,  2.0,   4.0, out, errors)
    _field(data, "crustal_thickness",  float, False,35.0,  5,    100,  out, errors)
    _field(data, "tectonic_stress",    float, True,  5.0,  0,    30,   out, errors)
    _field(data, "ground_deformation", float, False, 0.0, -20,   30,   out, errors)
    _field(data, "recent_activity",    int,   False,  0,   0,    100,  out, errors)
    _field(data, "max_historical_mag", float, False, 5.0,  0,     9.5, out, errors)
    _field(data, "groundwater_change", float, False, 0.0, -50,   50,   out, errors)
    _field(data, "radon_emission",     float, False,20.0,  0,   200,   out, errors)
    return out, errors


def validate_wildfire(data: dict) -> tuple[dict, dict]:
    out, errors = {}, {}
    out["location"] = str(data.get("location", "your area"))[:100]

    _field(data, "temperature_c",      float, True,  28,  -20,  60,  out, errors)
    _field(data, "humidity_pct",       float, True,  50,    0, 100,  out, errors)
    _field(data, "wind_speed_kmh",     float, True,  15,    0, 300,  out, errors)
    _field(data, "vegetation_dryness", float, True, 0.5,    0,   1,  out, errors)
    _field(data, "elevation_m",        float, False,500,    0,5000,  out, errors)
    _field(data, "slope_degrees",      float, False, 10,    0,  90,  out, errors)
    _field(data, "recent_fires",       float, False,  0,    0, 100,  out, errors)
    _field(data, "fuel_load",          float, False,  3,    0,  20,  out, errors)
    return out, errors


def validate_storm(data: dict) -> tuple[dict, dict]:
    out, errors = {}, {}
    out["location"] = str(data.get("location", "your area"))[:100]

    _field(data, "wind_speed_kmh",        float, True,   30,  0,   350, out, errors)
    _field(data, "central_pressure_hpa",  float, True, 1000, 850, 1050, out, errors)
    _field(data, "sea_surface_temp_c",    float, True,   26,  -2,   40, out, errors)
    _field(data, "coastal_slope_degrees", float, False,   2,   0,   45, out, errors)
    _field(data, "sea_elevation_m",       float, False, 100,   0, 6000, out, errors)
    _field(data, "latitude_degrees",      float, False,  20, -90,   90, out, errors)
    _field(data, "recent_storms",         float, False,   0,   0,   50, out, errors)
    _field(data, "ocean_current_ms",      float, False,   0,  -5,    5, out, errors)
    return out, errors
