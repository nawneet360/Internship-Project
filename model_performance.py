

import pickle
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

examples_dir = BASE_DIR / "examples"
sys.path.insert(0, str(examples_dir))

print("\n" + "=" * 70)
print(" DisasterAI Pro — Model Performance Summary")
print("=" * 70 + "\n")

if not MODELS_DIR.exists():
    print(" No models directory found. Run train_on_kaggle.py first.\n")
    sys.exit(1)

models = [
    ("🌊 Flood", "flood_model.pkl"),
    ("🌍 Earthquake", "earthquake_model.pkl"),
    ("🔥 Wildfire", "wildfire_model.pkl"),
    ("🌀 Storm", "storm_model.pkl"),
]

total_size = 0

for label, filename in models:
    model_path = MODELS_DIR / filename
    
    if not model_path.exists():
        print(f"❌  {label:<20} — Model file not found")
        continue
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    total_size += size_mb
    
    try:
        with open(model_path, 'rb') as f:
            system = pickle.load(f)
        
        print(f"✅  {label}")
        print(f"    File: {filename} ({size_mb:.1f} MB)")
        
        if system.is_trained:
            print(f"    Status: ✅ Ready for predictions")
            
            # Show specific model info
            if "Flood" in label:
                if hasattr(system, 'model'):
                    print(f"    Model type: {type(system.model).__name__}")
                    print(f"    Features: {len(system.feature_names) if system.feature_names else 'N/A'}")
                    if system.feature_names:
                        print(f"    → {', '.join(system.feature_names[:3])} ...")
            else:
                print(f"    Occurrence Model: {type(system.occurrence_model).__name__} (🎯 Classification)")
                if hasattr(system, 'intensity_model'):
                    print(f"    Intensity Model: {type(system.intensity_model).__name__} (📈 Regression)")
                print(f"    Features: {len(system.feature_names) if system.feature_names else 'N/A'}")
                if system.feature_names:
                    print(f"    → {', '.join(system.feature_names[:3])} ...")
        else:
            print(f"    Status: ⚠️  Not trained")
        
        print()
    
    except Exception as e:
        print(f"⚠️   {label:<20} — Error loading model: {e}\n")

print("─" * 70)
print(f"📁  Models Directory: {MODELS_DIR}")
print(f"💾  Total Size: {total_size:.1f} MB")
print("\n🚀  Ready for deployment!\n")
print("    Deploy with: python3 web_server.py")
print("    Then open: disaster_prediction_system.html\n")
print("=" * 70 + "\n")
