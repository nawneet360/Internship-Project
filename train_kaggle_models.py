#!/usr/bin/env python3
"""
train_kaggle_models.py
======================
Standalone script to download Kaggle datasets and train all 4 models.
Run this AFTER setting up your Kaggle API token.

Usage:
    python3 train_kaggle_models.py

What it does:
    1. Checks for Kaggle API token
    2. Downloads datasets from Kaggle (or uses cached data/)
    3. Trains all 4 models on real data
    4. Saves trained models to models/ folder
    5. Displays performance metrics
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# Add examples folder to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'examples'))

import numpy as np
import pandas as pd

try:
    from flood_prediction import FloodPredictionSystem
    from earthquake_prediction import EarthquakePredictionSystem
    from wildfire_prediction import WildfirePredictionSystem
    from storm_prediction import StormPredictionSystem
except ImportError as e:
    print(f"❌ Error importing prediction modules: {e}")
    sys.exit(1)

try:
    from kaggle_data_loader import load_all_data
except ImportError as e:
    print(f"❌ Error importing kaggle_data_loader: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def train_flood_model(data=None):
    """Train flood prediction model on Kaggle data."""
    print("\n  🌊 FLOOD MODEL")
    print("  " + "─" * 50)
    
    model = FloodPredictionSystem()
    
    if data is None or data.get("flood") is None:
        print("    Loading synthetic data (Kaggle not available)…")
        data_to_train = model.generate_sample_data(1500)
        src = "Synthetic (1,500 records)"
    else:
        data_to_train = data["flood"]
        src = f"Kaggle ({len(data_to_train):,} records)"
        print(f"    Loading real data: {src}…")
    
    try:
        print(f"    Training on {src}…", end="", flush=True)
        metrics, eval_data = model.train_model(data_to_train)
        
        print(" ✅")
        print(f"    Accuracy: {metrics['accuracy']:.2%}")
        print(f"    Precision: {metrics['precision']:.2%}")
        print(f"    Recall: {metrics['recall']:.2%}")
        print(f"    F1-Score: {metrics['f1']:.2%}")
        print(f"    ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
        
        return model, metrics
    except Exception as e:
        print(f" ❌ ({e})")
        return None, None


def train_earthquake_model(data=None):
    """Train earthquake prediction model on Kaggle data."""
    print("\n  🌍 EARTHQUAKE MODEL")
    print("  " + "─" * 50)
    
    model = EarthquakePredictionSystem()
    
    if data is None or data.get("earthquake") is None:
        print("    Loading synthetic data (Kaggle not available)…")
        data_to_train = model.generate_sample_data(1500)
        src = "Synthetic (1,500 records)"
    else:
        data_to_train = data["earthquake"]
        src = f"Kaggle ({len(data_to_train):,} records)"
        print(f"    Loading real data: {src}…")
    
    try:
        print(f"    Training on {src}…", end="", flush=True)
        metrics, eval_data = model.train_models(data_to_train)
        
        print(" ✅")
        print(f"    Occurrence Accuracy: {metrics['occurrence_accuracy']:.2%}")
        
        if 'magnitude_mae' in metrics and metrics['magnitude_mae'] > 0:
            print(f"    Magnitude MAE: ±{metrics['magnitude_mae']:.2f} points")
        if 'magnitude_r2' in metrics:
            print(f"    Magnitude R²: {metrics['magnitude_r2']:.3f}")
        
        return model, metrics
    except Exception as e:
        print(f" ❌ ({e})")
        return None, None


def train_wildfire_model(data=None):
    """Train wildfire prediction model on Kaggle data."""
    print("\n  🔥 WILDFIRE MODEL")
    print("  " + "─" * 50)
    
    model = WildfirePredictionSystem()
    
    if data is None or data.get("wildfire") is None:
        print("    Loading synthetic data (Kaggle not available)…")
        data_to_train = model.generate_sample_data(1500)
        src = "Synthetic (1,500 records)"
    else:
        data_to_train = data["wildfire"]
        src = f"Kaggle ({len(data_to_train):,} records)"
        print(f"    Loading real data: {src}…")
    
    try:
        print(f"    Training on {src}…", end="", flush=True)
        metrics, eval_data = model.train_models(data_to_train)
        
        print(" ✅")
        print(f"    Accuracy: {metrics['accuracy']:.2%}")
        print(f"    Precision: {metrics['precision']:.2%}")
        print(f"    Recall: {metrics['recall']:.2%}")
        print(f"    F1-Score: {metrics['f1']:.2%}")
        print(f"    ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
        
        return model, metrics
    except Exception as e:
        print(f" ❌ ({e})")
        return None, None


def train_storm_model(data=None):
    """Train storm prediction model on Kaggle data."""
    print("\n  🌪️  STORM MODEL")
    print("  " + "─" * 50)
    
    model = StormPredictionSystem()
    
    if data is None or data.get("storm") is None:
        print("    Loading synthetic data (Kaggle not available)…")
        data_to_train = model.generate_sample_data(1500)
        src = "Synthetic (1,500 records)"
    else:
        data_to_train = data["storm"]
        src = f"Kaggle ({len(data_to_train):,} records)"
        print(f"    Loading real data: {src}…")
    
    try:
        print(f"    Training on {src}…", end="", flush=True)
        metrics, eval_data = model.train_models(data_to_train)
        
        print(" ✅")
        print(f"    Occurrence Accuracy: {metrics['occurrence_accuracy']:.2%}")
        
        if 'intensity_mae' in metrics and metrics['intensity_mae'] > 0:
            print(f"    Intensity MAE: ±{metrics['intensity_mae']:.2f}")
        
        return model, metrics
    except Exception as e:
        print(f" ❌ ({e})")
        return None, None


def save_models(models_dict):
    """Save trained models to disk."""
    print("\n  💾 Saving Models to Disk")
    print("  " + "─" * 50)
    
    import pickle
    
    for name, model in models_dict.items():
        if model is None:
            continue
        
        filepath = MODELS_DIR / f"{name}_model.pkl"
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"    ✅ {name}_model.pkl")
        except Exception as e:
            print(f"    ❌ {name}_model.pkl: {e}")


def print_header():
    """Print ASCII art header."""
    print("\n" + "=" * 60)
    print("  🚀 DisasterAI Pro — Kaggle Model Training")
    print("=" * 60)


def print_summary(all_metrics):
    """Print training summary."""
    print("\n" + "=" * 60)
    print("  📊 TRAINING SUMMARY")
    print("=" * 60)
    
    for name, metrics in all_metrics.items():
        if metrics is None:
            print(f"\n  {name}: ❌ Failed")
            continue
        
        print(f"\n  {name}: ✅ Success")
        
        if 'accuracy' in metrics:
            print(f"    • Accuracy: {metrics['accuracy']:.2%}")
        if 'occurrence_accuracy' in metrics:
            print(f"    • Occurrence Accuracy: {metrics['occurrence_accuracy']:.2%}")
        if 'magnitude_mae' in metrics and metrics['magnitude_mae'] > 0:
            print(f"    • Magnitude MAE: ±{metrics['magnitude_mae']:.2f}")
        if 'magnitude_r2' in metrics:
            print(f"    • Magnitude R²: {metrics['magnitude_r2']:.3f}")
    
    print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print_header()
    
    # Step 1: Load Kaggle data
    print("\n  📦 Loading Kaggle Datasets")
    print("  " + "─" * 50)
    
    kaggle_data = load_all_data()
    
    if kaggle_data:
        for name, df in kaggle_data.items():
            if df is not None:
                print(f"    ✅ {name}: {len(df):,} records")
            else:
                print(f"    ⚠️  {name}: Not available (will use synthetic data)")
    else:
        print("    ⚠️  No Kaggle data found (will use synthetic data)")
    
    # Step 2: Train all models
    all_metrics = {}
    models_dict = {}
    
    flood_model, flood_metrics = train_flood_model(kaggle_data)
    models_dict['flood'] = flood_model
    all_metrics['Flood'] = flood_metrics
    
    eq_model, eq_metrics = train_earthquake_model(kaggle_data)
    models_dict['earthquake'] = eq_model
    all_metrics['Earthquake'] = eq_metrics
    
    wf_model, wf_metrics = train_wildfire_model(kaggle_data)
    models_dict['wildfire'] = wf_model
    all_metrics['Wildfire'] = wf_metrics
    
    storm_model, storm_metrics = train_storm_model(kaggle_data)
    models_dict['storm'] = storm_model
    all_metrics['Storm'] = storm_metrics
    
    # Step 3: Save models
    save_models(models_dict)
    
    # Step 4: Print summary
    print_summary(all_metrics)
    
    print("\n  ✅ Training Complete!")
    print("  Next: Start web server to use these models")
    print("  Command: python3 web_server.py\n")


if __name__ == "__main__":
    main()
