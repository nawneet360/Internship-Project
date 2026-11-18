

import os
import sys
import warnings
import zipfile
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent.parent  
DATA_DIR  = BASE_DIR / "data"                      
DATA_DIR.mkdir(exist_ok=True)

log = logging.getLogger("kaggle_loader")
logging.basicConfig(level=logging.INFO,
                    format="  %(message)s")



def _kaggle_available() -> bool:

    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        log.warning(
            
        )
        return False

    try:
        import kaggle  
        kaggle.api.authenticate()
        return True
    except ImportError:
        log.warning("kaggle package not found — run: .venv/bin/pip install kaggle")
        return False
    except (Exception, SystemExit):
        log.warning("Kaggle authentication failed. Falling back to synthetic data.")
        return False


def _download_dataset(slug: str, dest_dir: Path) -> bool:
    """
    Download and unzip a Kaggle dataset into dest_dir.
    Returns True on success, False on failure.
    slug format: 'owner/dataset-name'
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle
        kaggle.api.authenticate()
        log.info(f"Downloading kaggle.com/datasets/{slug} …")
        kaggle.api.dataset_download_files(
            slug, path=str(dest_dir), unzip=True, quiet=False
        )
        return True
    except Exception as exc:
        log.warning(f"Download failed for {slug}: {exc}")
        return False


def _csv_in(folder: Path, keywords: list[str] | None = None) -> Path | None:
    """Return the first CSV file in folder, optionally filtered by keyword in name."""
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        return None
    if keywords:
        for kw in keywords:
            for f in csvs:
                if kw.lower() in f.name.lower():
                    return f
    return csvs[0]



FLOOD_SLUG   = "khalidalam980/predicting-flood-probability-features"
FLOOD_DIR    = DATA_DIR / "flood"



def load_flood_data() -> pd.DataFrame | None:

    if not _kaggle_available():
        return None

    csv_path = _csv_in(FLOOD_DIR)
    if csv_path is None:
        ok = _download_dataset(FLOOD_SLUG, FLOOD_DIR)
        if not ok:
            return None
        csv_path = _csv_in(FLOOD_DIR)
    if csv_path is None:
        log.warning("Flood: could not locate CSV after download.")
        return None

    log.info(f"Flood: reading {csv_path.name} …")
    raw = pd.read_csv(csv_path)
    log.info(f"Flood: raw shape {raw.shape}, columns: {list(raw.columns[:5])} …")


    raw = raw.dropna(subset=["FloodProbability"])

    def _scale(col, lo, hi):
        mn, mx = raw[col].min(), raw[col].max()
        if mx == mn:
            return pd.Series(np.full(len(raw), (lo + hi) / 2), index=raw.index)
        return lo + (raw[col] - mn) / (mx - mn) * (hi - lo)

    df = pd.DataFrame()
    df["rainfall_mm"]       = _scale("MonsoonIntensity",    5,   120)
    df["rainfall_7day_avg"] = df["rainfall_mm"] * np.random.uniform(0.6, 1.0, len(raw))
    df["temperature_c"]     = np.clip(
        np.random.normal(25, 8, len(raw)), 5, 45
    )
    df["humidity_pct"]      = _scale("DrainageSystems",    30,   100).clip(30, 100)
 
    df["elevation_m"]       = (1 - _scale("TopographyDrainage", 0, 1)) * 200
    df["slope_degrees"]     = np.clip(np.random.exponential(5, len(raw)), 0.5, 30)

    df["soil_permeability"] = 1 - _scale("Siltation", 0, 0.9)

    df["river_level_m"]     = _scale("RiverManagement", 1, 12)

    df["flood_risk"]        = raw["FloodProbability"].values

    df = df.astype(float).dropna()
    log.info(f"Flood: ✅ {len(df):,} training records ready  "
             f"(avg risk = {df['flood_risk'].mean():.2%})")
    return df




EQ_SLUG  = "usgs/earthquake-database"
EQ_DIR   = DATA_DIR / "earthquake"


def load_earthquake_data() -> pd.DataFrame | None:

    if not _kaggle_available():
        return None

    csv_path = _csv_in(EQ_DIR, ["database", "earthquake"])
    if csv_path is None:
        ok = _download_dataset(EQ_SLUG, EQ_DIR)
        if not ok:
            return None
        csv_path = _csv_in(EQ_DIR, ["database", "earthquake"])
    if csv_path is None:
        log.warning("Earthquake: could not locate CSV after download.")
        return None

    log.info(f"Earthquake: reading {csv_path.name} …")
    raw = pd.read_csv(csv_path, low_memory=False)
    # Keep only tectonic earthquakes with valid magnitude
    raw = raw[raw["Type"] == "Earthquake"].copy()
    raw["Magnitude"] = pd.to_numeric(raw["Magnitude"], errors="coerce")
    raw["Depth"]     = pd.to_numeric(raw["Depth"],     errors="coerce")
    raw = raw.dropna(subset=["Magnitude", "Depth"])
    log.info(f"Earthquake: raw tectonic events shape {raw.shape}")

    np.random.seed(42)
    n = len(raw)
    df = pd.DataFrame()

    mag  = raw["Magnitude"].values
    dep  = raw["Depth"].values.clip(1, 700)

    df["crustal_thickness"]   = dep.clip(5, 80)
    df["p_wave_velocity"]     = (5.0 + dep / 100 * 2.5).clip(4.0, 8.5)
    df["s_wave_velocity"]     = df["p_wave_velocity"] / np.random.uniform(1.6, 1.8, n)


    gap = pd.to_numeric(raw.get("Azimuthal Gap", pd.Series(np.full(n, 90))),
                         errors="coerce").fillna(90).values
    df["micro_tremors"]       = np.clip((180 - gap) / 18 + np.random.exponential(1, n), 0, 15)

    mss = pd.to_numeric(
        raw.get("Magnitude Seismic Stations", pd.Series(np.full(n, 2))),
        errors="coerce"
    ).fillna(2).values
    df["recent_activity"]     = np.clip(mss / 5, 0, 10).astype(int)

    df["tectonic_stress"]     = np.clip(mag * 1.5 + np.random.normal(0, 1, n), 0, 20)
    df["fault_distance_km"]   = np.clip(
        np.random.exponential(40, n) / (mag / 5 + 0.1), 1, 200
    )
    df["rock_density"]        = np.random.normal(2.7, 0.2, n).clip(2.2, 3.3)
    df["ground_deformation"]  = np.clip(mag * 1.2 + np.random.normal(0, 1, n), -5, 15)
    df["max_historical_mag"]  = mag
    df["groundwater_change"]  = np.random.normal(0, 3, n)
    df["radon_emission"]      = np.clip(
        15 + (mag - 3) * 8 + np.random.exponential(10, n), 5, 80
    )

    # ── Targets ──────────────────────────────────────────────────────────────
    occurrence = (mag >= 4.5).astype(int)
    df["earthquake_occurrence"]   = occurrence
    df["earthquake_magnitude"]    = np.where(occurrence == 1, mag, 0.0)

    # probability: logistic-ish mapping from magnitude
    raw_prob = 1 / (1 + np.exp(-1.5 * (mag - 4.0)))
    df["earthquake_probability"]  = raw_prob.clip(0, 1)

    df = df.astype(float).dropna()
    n_eq = df["earthquake_occurrence"].sum()
    log.info(f"Earthquake: ✅ {len(df):,} records  "
             f"({int(n_eq):,} events ≥ M4.5,  "
             f"avg M = {df['max_historical_mag'].mean():.1f})")
    return df




WF_SLUG  = "elikplim/forest-fires"
WF_DIR   = DATA_DIR / "wildfire"



def load_wildfire_data() -> pd.DataFrame | None:
    
    if not _kaggle_available():
        return None

    csv_path = _csv_in(WF_DIR)
    if csv_path is None:
        ok = _download_dataset(WF_SLUG, WF_DIR)
        if not ok:
            return None
        csv_path = _csv_in(WF_DIR)
    if csv_path is None:
        log.warning("Wildfire: could not locate CSV after download.")
        return None

    log.info(f"Wildfire: reading {csv_path.name} …")
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.strip()
    for col in ["temp", "RH", "wind", "rain", "area", "FFMC", "DMC", "DC", "ISI"]:
        raw[col] = pd.to_numeric(raw.get(col, pd.Series(np.zeros(len(raw)))),
                                  errors="coerce")
    raw = raw.dropna(subset=["temp", "RH", "wind", "area"])
    log.info(f"Wildfire: raw shape {raw.shape}")

   
    np.random.seed(42)
    n_orig = len(raw)
    n_aug  = max(0, 2000 - n_orig)

    def _bootstrap(series, noise_std=0.05):
        sampled = series.sample(n_aug, replace=True).values
        return sampled * (1 + np.random.normal(0, noise_std, n_aug))

    if n_aug > 0:
        aug = pd.DataFrame({
            "temp":  _bootstrap(raw["temp"],  0.08),
            "RH":    _bootstrap(raw["RH"],    0.08),
            "wind":  _bootstrap(raw["wind"],  0.10),
            "rain":  _bootstrap(raw["rain"],  0.20).clip(0),
            "area":  _bootstrap(raw["area"],  0.15).clip(0),
            "FFMC":  _bootstrap(raw["FFMC"],  0.05),
            "DMC":   _bootstrap(raw["DMC"],   0.08),
            "DC":    _bootstrap(raw["DC"],    0.05),
            "ISI":   _bootstrap(raw["ISI"],   0.08),
        })
        combined = pd.concat([raw[["temp","RH","wind","rain","area","FFMC","DMC","DC","ISI"]],
                               aug], ignore_index=True)
    else:
        combined = raw[["temp","RH","wind","rain","area","FFMC","DMC","DC","ISI"]].copy()

    n = len(combined)
    df = pd.DataFrame()
    df["temperature_c"]      = combined["temp"].clip(-10, 50)
    df["humidity_pct"]       = combined["RH"].clip(5, 100)
    df["wind_speed_kmh"]     = (combined["wind"] * 3.6).clip(0, 120)   

    dc_max = combined["DC"].clip(0, 900)
    df["vegetation_dryness"] = (dc_max / 900).clip(0, 1)

  
    df["fuel_load"]          = (combined["ISI"] / 5).clip(0.1, 10)

 
    df["elevation_m"]        = np.random.uniform(100, 1500, n)
    df["slope_degrees"]      = np.abs(np.random.normal(10, 6, n)).clip(0, 40)
    df["recent_fires"]       = (combined["rain"] > 0).astype(int).values

  
    area = combined["area"].values
    df["fire_occurrence"]    = (area > 0).astype(int)
    log_area = np.log1p(area)
    max_la   = log_area.max() if log_area.max() > 0 else 1
    df["fire_intensity"]     = np.where(area > 0, log_area / max_la * 9, 0.0)

  
    ffmc_norm = (combined["FFMC"].clip(0, 100) / 100).values
    isi_norm  = (combined["ISI"].clip(0, 20) / 20).values
    df["fire_probability"]   = np.clip(0.5 * ffmc_norm + 0.5 * isi_norm
                                        + np.random.normal(0, 0.05, n), 0, 1)

    df = df.astype(float).dropna()
    n_fires = int(df["fire_occurrence"].sum())
    log.info(f"Wildfire: ✅ {len(df):,} records  "
             f"({n_fires:,} fire events,  "
             f"avg intensity = {df.loc[df['fire_intensity']>0,'fire_intensity'].mean():.2f})")
    return df




STORM_SLUG  = "noaa/hurricane-database"
STORM_DIR   = DATA_DIR / "storm"


def load_storm_data() -> pd.DataFrame | None:
    
    if not _kaggle_available():
        return None

    csv_path = _csv_in(STORM_DIR, ["atlantic", "pacific"])
    if csv_path is None:
        ok = _download_dataset(STORM_SLUG, STORM_DIR)
        if not ok:
            return None
        csv_path = _csv_in(STORM_DIR, ["atlantic", "pacific"])
    if csv_path is None:
        log.warning("Storm: could not locate CSV after download.")
        return None

    csvs = list(STORM_DIR.glob("*.csv"))
    frames = []
    for f in csvs:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = raw.columns.str.strip()
    log.info(f"Storm: raw shape {raw.shape}")

    # Column names may vary; try multiple aliases
    wind_col = next((c for c in raw.columns
                     if "wind" in c.lower() and "max" in c.lower()), None) or \
               next((c for c in raw.columns if "wind" in c.lower()), None)
    pres_col = next((c for c in raw.columns
                     if "pressure" in c.lower() or "pres" in c.lower()), None)
    lat_col  = next((c for c in raw.columns if "lat" in c.lower()), None)

    if wind_col is None or lat_col is None:
        log.warning(f"Storm: could not identify required columns in {list(raw.columns[:8])}")
        return None

    raw[wind_col] = pd.to_numeric(raw[wind_col], errors="coerce")
    if pres_col:
        raw[pres_col] = pd.to_numeric(raw[pres_col], errors="coerce")


    def parse_lat(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        try:
            return float(v)
        except ValueError:
            if v.endswith("N"):
                return float(v[:-1])
            if v.endswith("S"):
                return -float(v[:-1])
            return np.nan

    raw["_lat"] = raw[lat_col].apply(parse_lat)
    raw = raw.dropna(subset=[wind_col, "_lat"])
    raw = raw[raw[wind_col] > 0]

    np.random.seed(42)
    n = len(raw)
    df = pd.DataFrame()

    wind_kt  = raw[wind_col].values
    wind_kmh = wind_kt * 1.852             # knots → km/h
    df["wind_speed_kmh"]    = wind_kmh.clip(0, 350)

    if pres_col:
        pres = raw[pres_col].fillna(1010).clip(850, 1025).values
    else:
        # Pressure estimated from wind via the Atkinson-Holliday formula proxy
        pres = 1010 - (wind_kt / 3.92) ** (1 / 0.644)
        pres = np.clip(pres, 850, 1025)
    df["central_pressure_hpa"] = pres

    lat = raw["_lat"].values
    df["latitude_degrees"]  = lat

    # SST estimated from latitude & month (warm tropics = high SST)
    df["sea_surface_temp_c"] = np.clip(30 - np.abs(lat) * 0.3
                                        + np.random.normal(0, 1.5, n), 18, 34)
    df["coastal_slope_degrees"] = np.random.uniform(0.5, 10, n)
    df["sea_elevation_m"]       = np.random.exponential(200, n).clip(20, 5000)
    df["recent_storms"]         = np.random.poisson(0.2, n)
    df["ocean_current_ms"]      = np.random.uniform(-2, 2, n)

    # Saffir-Simpson category from wind speed (km/h)
    def _cat(w):
        if w < 119:   return 0   # TD / TS
        if w < 153:   return 1
        if w < 177:   return 2
        if w < 209:   return 3
        if w < 252:   return 4
        return 5

    cats = np.array([_cat(w) for w in wind_kmh])
    df["storm_occurrence"] = (cats >= 1).astype(int)
    df["storm_intensity"]  = np.where(cats >= 1, cats.astype(float), 0.0)

    raw_prob = 1 / (1 + np.exp(-0.05 * (wind_kmh - 80)))
    df["storm_probability"] = raw_prob.clip(0, 1)

    df = df.astype(float).dropna()
    n_storms = int(df["storm_occurrence"].sum())
    log.info(f"Storm: ✅ {len(df):,} records  "
             f"({n_storms:,} hurricane events,  "
             f"avg wind = {df['wind_speed_kmh'].mean():.0f} km/h)")
    return df


def load_all_data() -> dict:
    
    return {
        "flood":      load_flood_data(),
        "earthquake": load_earthquake_data(),
        "wildfire":   load_wildfire_data(),
        "storm":      load_storm_data(),
    }


# ── Standalone dry-run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌍  DisasterAI Pro — Kaggle Data Loader  (dry-run)\n" + "─" * 52)
    results = load_all_data()
    print("\nSummary")
    print("─" * 30)
    for name, df in results.items():
        if df is not None:
            print(f"  ✅  {name:<12} {len(df):>7,} rows  |  "
                  f"columns: {list(df.columns)[:3]} …")
        else:
            print(f"  ⚠️   {name:<12} — will fall back to synthetic data")
    print()
