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

class WildfirePredictionSystem:
    def __init__(self):
        self.occurrence_model = RandomForestClassifier(
            n_estimators=150, random_state=42
        )
        self.intensity_model = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        self.is_trained = False
        self.feature_names = None

    def generate_sample_data(self, n_samples=2000):
        np.random.seed(42)
        temperature = np.random.normal(28, 8, n_samples)   # °C
        humidity = np.random.uniform(10, 90, n_samples)    # %
        wind_speed = np.random.exponential(15, n_samples)  # km/h
        
        vegetation_dryness = np.random.uniform(0.2, 0.95, n_samples)
        elevation = np.random.uniform(0, 2000, n_samples)  # m
        slope = np.random.uniform(0, 40, n_samples)        # degrees
        
        recent_fires = np.random.poisson(0.3, n_samples)   # past 30 days
        fuel_load = np.random.gamma(2, 3, n_samples)       # tons/hectare
        
        # fire probability based on weather + fuel + terrain
        prob = (
            0.30 * np.maximum(0, (temperature - 20) / 15) +
            0.25 * np.maximum(0, (50 - humidity) / 50) +
            0.20 * np.minimum(1, wind_speed / 40) +
            0.15 * vegetation_dryness +
            0.05 * np.maximum(0, slope / 30) +
            0.03 * np.minimum(1, recent_fires / 3) +
            0.02 * np.minimum(1, fuel_load / 10)
        )
        prob = np.clip(prob + np.random.normal(0, 0.05, n_samples), 0, 1)
        
        occurrence = (prob > 0.35).astype(int)
        base_intensity = 1 + 8 * prob
        intensity = np.where(
            occurrence == 1,
            np.clip(base_intensity + np.random.normal(0, 0.8, n_samples), 1.0, 9.0),
            0.0
        )

        return pd.DataFrame({
            'temperature_c': temperature,
            'humidity_pct': humidity,
            'wind_speed_kmh': wind_speed,
            'vegetation_dryness': vegetation_dryness,
            'elevation_m': elevation,
            'slope_degrees': slope,
            'recent_fires': recent_fires,
            'fuel_load': fuel_load,
            'fire_probability': prob,
            'fire_occurrence': occurrence,
            'fire_intensity': intensity,
        })

    def train_models(self, data):
        targets = ['fire_probability', 'fire_occurrence', 'fire_intensity']
        X = data.drop(columns=targets)
        self.feature_names = X.columns.tolist()

        y_occ = data['fire_occurrence']
        y_int = data['fire_intensity']

        (X_train, X_test,
         y_occ_train, y_occ_test,
         y_int_train, y_int_test) = train_test_split(
            X, y_occ, y_int,
            test_size=0.2, random_state=42, stratify=y_occ
        )

        self.occurrence_model.fit(X_train, y_occ_train)

        fire_mask = y_int_train > 0
        if fire_mask.sum() > 10:
            self.intensity_model.fit(X_train[fire_mask], y_int_train[fire_mask])

        self.is_trained = True

        occ_pred = self.occurrence_model.predict(X_test)
        occ_prob = self.occurrence_model.predict_proba(X_test)[:, 1]

        fire_test_mask = y_int_test > 0
        int_pred = np.zeros_like(y_int_test, dtype=float)
        if fire_test_mask.sum() > 0:
            int_pred[fire_test_mask] = self.intensity_model.predict(
                X_test[fire_test_mask]
            )

        metrics = {
            'occurrence_accuracy': accuracy_score(y_occ_test, occ_pred),
            'occurrence_report': classification_report(
                y_occ_test, occ_pred, output_dict=True
            ),
            'intensity_mae': mean_absolute_error(
                y_int_test[fire_test_mask], int_pred[fire_test_mask]
            ) if fire_test_mask.sum() > 0 else 0,
            'intensity_r2': r2_score(
                y_int_test[fire_test_mask], int_pred[fire_test_mask]
            ) if fire_test_mask.sum() > 0 else 0,
        }

        return metrics, (X_test, y_occ_test, y_int_test, occ_pred, occ_prob, int_pred)

    def predict_wildfire(self, seismic_data):
        if not self.is_trained:
            raise RuntimeError("Please train the models first by calling train_models().")
        
        if isinstance(seismic_data, dict):
            seismic_data = pd.DataFrame([seismic_data])

        prob = self.occurrence_model.predict_proba(seismic_data)[:, 1]
        occurrence = self.occurrence_model.predict(seismic_data)

        intensity = np.zeros_like(occurrence, dtype=float)
        fire_mask = occurrence == 1
        if fire_mask.sum() > 0:
            intensity[fire_mask] = self.intensity_model.predict(
                seismic_data[fire_mask]
            )

        return prob, occurrence, intensity

    def risk_level(self, probability, intensity=0.0):
        if probability < 0.25:
            return "Low"
        if probability < 0.45:
            return "Moderate"
        if probability < 0.70:
            return "Critical" if intensity >= 6.0 else "High"
        return "Extreme" if intensity >= 7.5 else ("Critical" if intensity >= 6.0 else "High")

    def generate_wildfire_alert(self, weather_data, location="your area"):
        prob, occurrence, intensity = self.predict_wildfire(weather_data)

        p = float(prob[0])
        occ = bool(occurrence[0])
        intens = float(intensity[0]) if occ else 0.0
        level = self.risk_level(p, intens)

        intens_str = f" (est. intensity {intens:.1f})" if occ else ""

        messages = {
            "Low": (f"Fire conditions in {location} are normal{intens_str}. "
                    f"No immediate action needed ({p:.0%} probability)."),
            "Moderate": (f"⚠️  Fire weather risk elevated in {location}{intens_str}. "
                        f"Review fire prevention measures ({p:.0%} probability)."),
            "High": (f"🚨 Fire risk is HIGH in {location}{intens_str}. "
                    f"Restrict outdoor burning, monitor weather closely ({p:.0%} probability)."),
            "Critical": (f"🚨🚨 CRITICAL FIRE RISK in {location}{intens_str}. "
                        f"Activate firefighting protocols, increase patrols ({p:.0%} probability)."),
            "Extreme": (f"🚨🚨🚨 EXTREME FIRE WARNING for {location}{intens_str}. "
                       f"Evacuate if ordered, prepare for rapid spread ({p:.0%} probability)."),
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'fire_probability': round(p, 4),
            'predicted_occurrence': occ,
            'predicted_intensity': round(intens, 2),
            'risk_level': level,
            'alert_message': messages[level],
        }

    def plot_feature_importance(self):
        if not self.is_trained:
            raise RuntimeError("Train the models before plotting.")

        labels = {
            'temperature_c': 'Temperature (°C)',
            'humidity_pct': 'Humidity (%)',
            'wind_speed_kmh': 'Wind Speed (km/h)',
            'vegetation_dryness': 'Vegetation Dryness',
            'elevation_m': 'Elevation (m)',
            'slope_degrees': 'Slope (degrees)',
            'recent_fires': 'Recent Fires (30d)',
            'fuel_load': 'Fuel Load (t/ha)',
        }

        occ_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.occurrence_model.feature_importances_,
        }).sort_values('importance', ascending=False)
        occ_imp['label'] = occ_imp['feature'].map(labels)

        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.suptitle("What signals drive wildfire risk?",
                     fontsize=14, fontweight='bold', y=1.01)

        # Occurrence model
        colors_occ = plt.cm.Reds(np.linspace(0.35, 0.85, len(occ_imp)))
        bars = axes[0].barh(occ_imp['label'], occ_imp['importance'],
                            color=colors_occ, edgecolor='white', linewidth=0.4)
        axes[0].set_title('"Will a fire start?" — feature importance',
                          fontsize=11, pad=8)
        axes[0].set_xlabel('Influence on prediction', fontsize=10)
        axes[0].invert_yaxis()
        for bar, val in zip(bars, occ_imp['importance']):
            axes[0].text(bar.get_width() + 0.001,
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', va='center', fontsize=8, color='#555')

        # Intensity model
        try:
            int_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.intensity_model.feature_importances_,
            }).sort_values('importance', ascending=False)
            int_imp['label'] = int_imp['feature'].map(labels)

            colors_int = plt.cm.Oranges(np.linspace(0.35, 0.85, len(int_imp)))
            bars2 = axes[1].barh(int_imp['label'], int_imp['importance'],
                                 color=colors_int, edgecolor='white', linewidth=0.4)
            axes[1].set_title('"How intense will it be?" — feature importance',
                              fontsize=11, pad=8)
            axes[1].set_xlabel('Influence on intensity estimate', fontsize=10)
            axes[1].invert_yaxis()
            for bar, val in zip(bars2, int_imp['importance']):
                axes[1].text(bar.get_width() + 0.001,
                             bar.get_y() + bar.get_height() / 2,
                             f'{val:.3f}', va='center', fontsize=8, color='#555')
        except Exception:
            axes[1].text(0.5, 0.5,
                         'Intensity model unavailable\n(too few fire samples)',
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=11, color='#888')
            axes[1].set_title('"How intense will it be?" — feature importance',
                              fontsize=11)

        plt.tight_layout()
        return fig


def run_demo():
    print("\n🔥  Wildfire Prediction System — Live Demo")
    print("─" * 54)

    system = WildfirePredictionSystem()

    print("  → Loading weather & terrain data…")
    data = system.generate_sample_data(2000)
    fires = data[data['fire_occurrence'] == 1]
    print(f"     {len(data):,} records — {len(fires):,} fire events "
          f"({len(fires)/len(data):.0%} of dataset)")
    if len(fires):
        print(f"     Avg intensity: {fires['fire_intensity'].mean():.1f}  "
              f"Max: {fires['fire_intensity'].max():.1f}")

    print("  → Training two models (occurrence + intensity)…")
    metrics, _ = system.train_models(data)
    print(f"     Occurrence accuracy: {metrics['occurrence_accuracy']:.2%}")
    print(f"     Intensity error (MAE): ±{metrics['intensity_mae']:.2f}")

    # Three realistic scenarios
    scenarios = [
        {
            "name": "California — high risk zone",
            "data": dict(temperature_c=38, humidity_pct=12, wind_speed_kmh=35,
                         vegetation_dryness=0.88, elevation_m=600, slope_degrees=22,
                         recent_fires=2, fuel_load=8.5),
        },
        {
            "name": "Pacific Northwest — low risk zone",
            "data": dict(temperature_c=18, humidity_pct=75, wind_speed_kmh=8,
                         vegetation_dryness=0.25, elevation_m=400, slope_degrees=8,
                         recent_fires=0, fuel_load=2.1),
        },
        {
            "name": "Australia — extreme conditions",
            "data": dict(temperature_c=42, humidity_pct=8, wind_speed_kmh=50,
                         vegetation_dryness=0.95, elevation_m=200, slope_degrees=18,
                         recent_fires=3, fuel_load=9.2),
        },
    ]

    print("\n  → Running predictions…\n")
    for s in scenarios:
        alert = system.generate_wildfire_alert(s["data"], s["name"])
        intens_note = (f"  (predicted intensity {alert['predicted_intensity']:.1f})"
                       if alert['predicted_occurrence'] else "")
        print(f"  📍 {alert['location']}")
        print(f"     Risk:   {alert['risk_level']}{intens_note}  "
              f"({alert['fire_probability']:.0%} probability)")
        print(f"     Alert:  {alert['alert_message']}")
        print()

    print("  → Building feature importance charts…")
    fig = system.plot_feature_importance()
    plt.show()
    print("  Done! ✅\n")

    return system


if __name__ == "__main__":
    run_demo()
