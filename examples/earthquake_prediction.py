
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                             mean_absolute_error, r2_score)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
class EarthquakePredictionSystem:
    def __init__(self):  
        self.occurrence_model = RandomForestClassifier(
            n_estimators=150, random_state=42
        )
        self.magnitude_model = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        self.is_trained    = False
        self.feature_names = None

    def generate_sample_data(self, n_samples=2000):
        np.random.seed(42)
        p_wave = np.random.normal(6.0, 1.5, n_samples)  
        s_wave = np.random.normal(3.5, 0.8, n_samples) 
      
        micro_tremors = np.random.exponential(2.0, n_samples)  

        fault_distance   = np.random.exponential(50, n_samples)  
        rock_density     = np.random.normal(2.7, 0.3, n_samples) 
        crustal_thick    = np.random.normal(35, 8, n_samples)    

        tectonic_stress    = np.random.gamma(2, 3, n_samples)    
        ground_deformation = np.random.normal(0, 2, n_samples)   

        recent_activity = np.random.poisson(0.5, n_samples)       
        max_historical  = np.random.gamma(1.5, 2, n_samples) 

      
        groundwater = np.random.normal(0, 5, n_samples)      
        radon       = np.random.exponential(20, n_samples)       

        prob = (
            0.25 * np.maximum(0, (micro_tremors - 1) / 10) +
            0.20 * np.maximum(0, (tectonic_stress - 2) / 15) +
            0.15 * np.maximum(0, 1 - fault_distance / 100) +
            0.15 * np.minimum(1, recent_activity / 5) +
            0.10 * np.maximum(0, (max_historical - 3) / 7) +
            0.08 * np.maximum(0, np.abs(ground_deformation) / 10) +
            0.05 * np.maximum(0, (radon - 15) / 50) +
            0.02 * np.maximum(0, np.abs(groundwater) / 10)
        )
        prob = np.clip(prob + np.random.normal(0, 0.05, n_samples), 0, 1) 
        occurrence = (prob > 0.3).astype(int)
        base_mag = 2 + 6 * prob
        magnitude = np.where(
            occurrence == 1,
            np.clip(base_mag + np.random.normal(0, 0.5, n_samples), 2.0, 9.0),
            0.0
        )

        return pd.DataFrame({
            'p_wave_velocity':   p_wave,
            's_wave_velocity':   s_wave,
            'micro_tremors':     micro_tremors,
            'fault_distance_km': fault_distance,
            'rock_density':      rock_density,
            'crustal_thickness': crustal_thick,
            'tectonic_stress':   tectonic_stress,
            'ground_deformation':ground_deformation,
            'recent_activity':   recent_activity,
            'max_historical_mag':max_historical,
            'groundwater_change':groundwater,
            'radon_emission':    radon,
            'earthquake_probability': prob,
            'earthquake_occurrence':  occurrence,
            'earthquake_magnitude':   magnitude,
        })


    def train_models(self, data):
        
        targets = ['earthquake_probability', 'earthquake_occurrence',
                   'earthquake_magnitude']
        X = data.drop(columns=targets)
        self.feature_names = X.columns.tolist()

        y_occ = data['earthquake_occurrence']
        y_mag = data['earthquake_magnitude']

        (X_train, X_test,
         y_occ_train, y_occ_test,
         y_mag_train, y_mag_test) = train_test_split(
            X, y_occ, y_mag,
            test_size=0.2, random_state=42, stratify=y_occ
        )

        self.occurrence_model.fit(X_train, y_occ_train)

        eq_mask = y_mag_train > 0
        if eq_mask.sum() > 10:
            self.magnitude_model.fit(X_train[eq_mask], y_mag_train[eq_mask])

        self.is_trained = True

        occ_pred = self.occurrence_model.predict(X_test)
        occ_prob = self.occurrence_model.predict_proba(X_test)[:, 1]

        eq_test_mask = y_mag_test > 0
        mag_pred = np.zeros_like(y_mag_test, dtype=float)
        if eq_test_mask.sum() > 0:
            mag_pred[eq_test_mask] = self.magnitude_model.predict(
                X_test[eq_test_mask]
            )

        metrics = {
            'occurrence_accuracy': accuracy_score(y_occ_test, occ_pred),
            'occurrence_report':   classification_report(
                y_occ_test, occ_pred, output_dict=True
            ),
            'magnitude_mae': mean_absolute_error(
                y_mag_test[eq_test_mask], mag_pred[eq_test_mask]
            ) if eq_test_mask.sum() > 0 else 0,
            'magnitude_r2': r2_score(
                y_mag_test[eq_test_mask], mag_pred[eq_test_mask]
            ) if eq_test_mask.sum() > 0 else 0,
        }

        return metrics, (X_test, y_occ_test, y_mag_test,
                         occ_pred, occ_prob, mag_pred)

    # ------------------------------------------------------------------
    # Prediction & alerting
    # ------------------------------------------------------------------

    def predict_earthquake(self, seismic_data):
        
        if not self.is_trained:
            raise RuntimeError(
                "Please train the models first by calling train_models()."
            )
        if isinstance(seismic_data, dict):
            seismic_data = pd.DataFrame([seismic_data])

        prob       = self.occurrence_model.predict_proba(seismic_data)[:, 1]
        occurrence = self.occurrence_model.predict(seismic_data)

        magnitude = np.zeros_like(occurrence, dtype=float)
        eq_mask   = occurrence == 1
        if eq_mask.sum() > 0:
            magnitude[eq_mask] = self.magnitude_model.predict(
                seismic_data[eq_mask]
            )

        return prob, occurrence, magnitude

    def risk_level(self, probability, magnitude=0.0):

        if probability < 0.20:
            return "Low"
        if probability < 0.40:
            return "Moderate"
        if probability < 0.70:
            return "Critical" if magnitude >= 6.0 else "High"
        return "Extreme" if magnitude >= 7.0 else ("Critical" if magnitude >= 6.0 else "High")

    def generate_earthquake_alert(self, seismic_data, location="your area"):
       
        prob, occurrence, magnitude = self.predict_earthquake(seismic_data)

        p    = float(prob[0])
        occ  = bool(occurrence[0])
        mag  = float(magnitude[0]) if occ else 0.0
        level = self.risk_level(p, mag)

        mag_str = f" (est. M{mag:.1f})" if occ else ""

        messages = {
            "Low":      (f"Seismic conditions in {location} are normal{mag_str}. "
                         f"No action needed ({p:.0%} probability)."),
            "Moderate": (f"⚠️  Slightly elevated seismic activity near {location}{mag_str}. "
                         f"Review your emergency plan just in case ({p:.0%} probability)."),
            "High":     (f"🚨 Earthquake risk is HIGH near {location}{mag_str}. "
                         f"Secure heavy furniture, identify shelter spots, "
                         f"keep shoes near your bed ({p:.0%} probability)."),
            "Critical": (f"🚨🚨 SERIOUS EARTHQUAKE RISK near {location}{mag_str}. "
                         f"Activate emergency protocols. Drop, cover, hold on "
                         f"if shaking starts ({p:.0%} probability)."),
            "Extreme":  (f"🚨🚨🚨 EXTREME EARTHQUAKE WARNING for {location}{mag_str}. "
                         f"Evacuate high-risk structures immediately. "
                         f"This is a life-safety emergency ({p:.0%} probability)."),
        }

        return {
            'timestamp':              datetime.now().isoformat(),
            'location':               location,
            'earthquake_probability': round(p, 4),
            'predicted_occurrence':   occ,
            'predicted_magnitude':    round(mag, 2),
            'risk_level':             level,
            'alert_message':          messages[level],
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def plot_feature_importance(self):
        """
        Show the sensors and measurements the model relies on most.

        Useful for briefing geologists or emergency managers:
        "Right now, tectonic stress is the biggest warning signal."
        """
        if not self.is_trained:
            raise RuntimeError("Train the models before plotting.")

        labels = {
            'p_wave_velocity':    'P-Wave Velocity (km/s)',
            's_wave_velocity':    'S-Wave Velocity (km/s)',
            'micro_tremors':      'Micro-Tremors (per hr)',
            'fault_distance_km':  'Distance to Fault (km)',
            'rock_density':       'Rock Density (g/cm³)',
            'crustal_thickness':  'Crustal Thickness (km)',
            'tectonic_stress':    'Tectonic Stress (MPa)',
            'ground_deformation': 'Ground Deformation (mm/yr)',
            'recent_activity':    'Recent Earthquakes (30 days)',
            'max_historical_mag': 'Historical Max Magnitude',
            'groundwater_change': 'Groundwater Change (m)',
            'radon_emission':     'Radon Emission (Bq/m³)',
        }

        occ_imp = pd.DataFrame({
            'feature':    self.feature_names,
            'importance': self.occurrence_model.feature_importances_,
        }).sort_values('importance', ascending=False)
        occ_imp['label'] = occ_imp['feature'].map(labels)

        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.suptitle("What signals drive earthquake risk?",
                     fontsize=14, fontweight='bold', y=1.01)

      
        colors_occ = plt.cm.Reds(np.linspace(0.35, 0.85, len(occ_imp)))
        bars = axes[0].barh(occ_imp['label'], occ_imp['importance'],
                            color=colors_occ, edgecolor='white', linewidth=0.4)
        axes[0].set_title('"Will an earthquake happen?" — feature importance',
                          fontsize=11, pad=8)
        axes[0].set_xlabel('Influence on prediction', fontsize=10)
        axes[0].invert_yaxis()
        for bar, val in zip(bars, occ_imp['importance']):
            axes[0].text(bar.get_width() + 0.001,
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', va='center', fontsize=8, color='#555')

        try:
            mag_imp = pd.DataFrame({
                'feature':    self.feature_names,
                'importance': self.magnitude_model.feature_importances_,
            }).sort_values('importance', ascending=False)
            mag_imp['label'] = mag_imp['feature'].map(labels)

            colors_mag = plt.cm.Oranges(np.linspace(0.35, 0.85, len(mag_imp)))
            bars2 = axes[1].barh(mag_imp['label'], mag_imp['importance'],
                                 color=colors_mag, edgecolor='white', linewidth=0.4)
            axes[1].set_title('"How strong will it be?" — feature importance',
                              fontsize=11, pad=8)
            axes[1].set_xlabel('Influence on magnitude estimate', fontsize=10)
            axes[1].invert_yaxis()
            for bar, val in zip(bars2, mag_imp['importance']):
                axes[1].text(bar.get_width() + 0.001,
                             bar.get_y() + bar.get_height() / 2,
                             f'{val:.3f}', va='center', fontsize=8, color='#555')
        except Exception:
            axes[1].text(0.5, 0.5,
                         'Magnitude model unavailable\n(too few earthquake samples)',
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=11, color='#888')
            axes[1].set_title('"How strong will it be?" — feature importance',
                              fontsize=11)

        plt.tight_layout()
        return fig




def run_demo():
    print("\n🌍  Earthquake Early Warning System — Live Demo")
    print("─" * 54)

    system = EarthquakePredictionSystem()

    print("  → Loading seismic & geological data…")
    data = system.generate_sample_data(2000)
    quakes = data[data['earthquake_occurrence'] == 1]
    print(f"     {len(data):,} records — {len(quakes):,} earthquake events "
          f"({len(quakes)/len(data):.0%} of dataset)")
    if len(quakes):
        print(f"     Avg magnitude: {quakes['earthquake_magnitude'].mean():.1f}  "
              f"Max: {quakes['earthquake_magnitude'].max():.1f}")

    print("  → Training two models (occurrence + magnitude)…")
    metrics, _ = system.train_models(data)
    print(f"     Occurrence accuracy: {metrics['occurrence_accuracy']:.2%}")
    print(f"     Magnitude error (MAE): ±{metrics['magnitude_mae']:.2f} Richter points")


    scenarios = [
        {
            "name": "Tokyo — active fault zone",
            "data": dict(p_wave_velocity=5.1, s_wave_velocity=2.9, micro_tremors=9.2,
                         fault_distance_km=4, rock_density=2.95, crustal_thickness=22,
                         tectonic_stress=14, ground_deformation=9.5, recent_activity=4,
                         max_historical_mag=7.9, groundwater_change=-5.2, radon_emission=58),
        },
        {
            "name": "Kansas City — stable craton",
            "data": dict(p_wave_velocity=6.8, s_wave_velocity=4.0, micro_tremors=0.3,
                         fault_distance_km=180, rock_density=2.55, crustal_thickness=45,
                         tectonic_stress=0.8, ground_deformation=0.2, recent_activity=0,
                         max_historical_mag=1.8, groundwater_change=0.1, radon_emission=9),
        },
        {
            "name": "Istanbul — North Anatolian Fault",
            "data": dict(p_wave_velocity=5.6, s_wave_velocity=3.2, micro_tremors=4.1,
                         fault_distance_km=18, rock_density=2.75, crustal_thickness=30,
                         tectonic_stress=8.5, ground_deformation=4.2, recent_activity=2,
                         max_historical_mag=5.8, groundwater_change=-2.1, radon_emission=32),
        },
    ]

    print("\n  → Running predictions…\n")
    for s in scenarios:
        alert = system.generate_earthquake_alert(s["data"], s["name"])
        mag_note = (f"  (predicted M{alert['predicted_magnitude']:.1f})"
                    if alert['predicted_occurrence'] else "")
        print(f"  📍 {alert['location']}")
        print(f"     Risk:   {alert['risk_level']}{mag_note}  "
              f"({alert['earthquake_probability']:.0%} probability)")
        print(f"     Alert:  {alert['alert_message']}")
        print()

    print("  → Building feature importance charts…")
    fig = system.plot_feature_importance()
    plt.show()
    print("  Done! \n")

    return system


if __name__ == "__main__":
    run_demo()
