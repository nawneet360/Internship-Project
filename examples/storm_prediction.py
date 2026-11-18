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

class StormPredictionSystem:
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
        wind_speed = np.random.exponential(25, n_samples)        # km/h
        central_pressure = np.random.normal(1000, 15, n_samples) # hPa
        sea_surface_temp = np.random.normal(26, 3, n_samples)    # °C
        
        coastal_slope = np.random.uniform(0.5, 10, n_samples)    # degrees
        sea_elevation = np.random.exponential(100, n_samples)    # m bathymetry
        latitude = np.random.uniform(-35, 35, n_samples)         # degrees
        
        recent_storms = np.random.poisson(0.2, n_samples)        # past 60 days
        ocean_current = np.random.uniform(-2, 2, n_samples)      # m/s anomaly
        
        prob = (
            0.30 * np.minimum(1, (wind_speed - 20) / 60) +
            0.25 * np.maximum(0, (1010 - central_pressure) / 50) +
            0.20 * np.maximum(0, (sea_surface_temp - 24) / 6) +
            0.10 * np.maximum(0, 1 - coastal_slope / 15) +
            0.07 * np.minimum(1, recent_storms / 2) +
            0.05 * np.maximum(0, np.abs(latitude) / 40) +
            0.03 * np.maximum(0, np.abs(ocean_current) / 2)
        )
        prob = np.clip(prob + np.random.normal(0, 0.05, n_samples), 0, 1)
        
        occurrence = (prob > 0.40).astype(int)
        base_intensity = 1 + 8 * prob
        intensity = np.where(
            occurrence == 1,
            np.clip(base_intensity + np.random.normal(0, 0.7, n_samples), 1.0, 5.0),
            0.0
        )

        return pd.DataFrame({
            'wind_speed_kmh': wind_speed,
            'central_pressure_hpa': central_pressure,
            'sea_surface_temp_c': sea_surface_temp,
            'coastal_slope_degrees': coastal_slope,
            'sea_elevation_m': sea_elevation,
            'latitude_degrees': latitude,
            'recent_storms': recent_storms,
            'ocean_current_ms': ocean_current,
            'storm_probability': prob,
            'storm_occurrence': occurrence,
            'storm_intensity': intensity,
        })

    def train_models(self, data):
        targets = ['storm_probability', 'storm_occurrence', 'storm_intensity']
        X = data.drop(columns=targets)
        self.feature_names = X.columns.tolist()

        y_occ = data['storm_occurrence']
        y_int = data['storm_intensity']

        (X_train, X_test,
         y_occ_train, y_occ_test,
         y_int_train, y_int_test) = train_test_split(
            X, y_occ, y_int,
            test_size=0.2, random_state=42, stratify=y_occ
        )

        self.occurrence_model.fit(X_train, y_occ_train)

        storm_mask = y_int_train > 0
        if storm_mask.sum() > 10:
            self.intensity_model.fit(X_train[storm_mask], y_int_train[storm_mask])

        self.is_trained = True

        occ_pred = self.occurrence_model.predict(X_test)
        occ_prob = self.occurrence_model.predict_proba(X_test)[:, 1]

        storm_test_mask = y_int_test > 0
        int_pred = np.zeros_like(y_int_test, dtype=float)
        if storm_test_mask.sum() > 0:
            int_pred[storm_test_mask] = self.intensity_model.predict(
                X_test[storm_test_mask]
            )

        metrics = {
            'occurrence_accuracy': accuracy_score(y_occ_test, occ_pred),
            'occurrence_report': classification_report(
                y_occ_test, occ_pred, output_dict=True
            ),
            'intensity_mae': mean_absolute_error(
                y_int_test[storm_test_mask], int_pred[storm_test_mask]
            ) if storm_test_mask.sum() > 0 else 0,
            'intensity_r2': r2_score(
                y_int_test[storm_test_mask], int_pred[storm_test_mask]
            ) if storm_test_mask.sum() > 0 else 0,
        }

        return metrics, (X_test, y_occ_test, y_int_test, occ_pred, occ_prob, int_pred)

    def predict_storm(self, ocean_data):
        if not self.is_trained:
            raise RuntimeError("Please train the models first by calling train_models().")
        
        if isinstance(ocean_data, dict):
            ocean_data = pd.DataFrame([ocean_data])

        prob = self.occurrence_model.predict_proba(ocean_data)[:, 1]
        occurrence = self.occurrence_model.predict(ocean_data)

        intensity = np.zeros_like(occurrence, dtype=float)
        storm_mask = occurrence == 1
        if storm_mask.sum() > 0:
            intensity[storm_mask] = self.intensity_model.predict(
                ocean_data[storm_mask]
            )

        return prob, occurrence, intensity

    def risk_level(self, probability, intensity=0.0):
        if probability < 0.25:
            return "Low"
        if probability < 0.45:
            return "Moderate"
        if probability < 0.70:
            return "Critical" if intensity >= 3.5 else "High"
        return "Extreme" if intensity >= 4.5 else ("Critical" if intensity >= 3.5 else "High")

    def generate_storm_alert(self, ocean_data, location="your area"):
        prob, occurrence, intensity = self.predict_storm(ocean_data)

        p = float(prob[0])
        occ = bool(occurrence[0])
        intens = float(intensity[0]) if occ else 0.0
        level = self.risk_level(p, intens)

        intens_str = f" (est. intensity {intens:.1f})" if occ else ""

        messages = {
            "Low": (f"Storm conditions near {location} are calm{intens_str}. "
                    f"No immediate threat ({p:.0%} probability)."),
            "Moderate": (f"⚠️  Elevated ocean activity near {location}{intens_str}. "
                        f"Monitor weather forecasts closely ({p:.0%} probability)."),
            "High": (f"🚨 Storm risk is HIGH near {location}{intens_str}. "
                    f"Secure vessels, prepare for surge and high winds ({p:.0%} probability)."),
            "Critical": (f"🚨🚨 CRITICAL STORM WARNING for {location}{intens_str}. "
                        f"Evacuate coastal areas, activate disaster protocols ({p:.0%} probability)."),
            "Extreme": (f"🚨🚨🚨 EXTREME STORM WARNING for {location}{intens_str}. "
                       f"Major evacuation underway, catastrophic damage expected ({p:.0%} probability)."),
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'storm_probability': round(p, 4),
            'predicted_occurrence': occ,
            'predicted_intensity': round(intens, 2),
            'risk_level': level,
            'alert_message': messages[level],
        }

    def plot_feature_importance(self):
        if not self.is_trained:
            raise RuntimeError("Train the models before plotting.")

        labels = {
            'wind_speed_kmh': 'Wind Speed (km/h)',
            'central_pressure_hpa': 'Central Pressure (hPa)',
            'sea_surface_temp_c': 'Sea Surface Temp (°C)',
            'coastal_slope_degrees': 'Coastal Slope (°)',
            'sea_elevation_m': 'Sea Elevation (m)',
            'latitude_degrees': 'Latitude (°)',
            'recent_storms': 'Recent Storms (60d)',
            'ocean_current_ms': 'Ocean Current (m/s)',
        }

        occ_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.occurrence_model.feature_importances_,
        }).sort_values('importance', ascending=False)
        occ_imp['label'] = occ_imp['feature'].map(labels)

        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.suptitle("What signals drive storm risk?",
                     fontsize=14, fontweight='bold', y=1.01)

        # Occurrence model
        colors_occ = plt.cm.Blues(np.linspace(0.35, 0.85, len(occ_imp)))
        bars = axes[0].barh(occ_imp['label'], occ_imp['importance'],
                            color=colors_occ, edgecolor='white', linewidth=0.4)
        axes[0].set_title('"Will a storm form?" — feature importance',
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

            colors_int = plt.cm.Purples(np.linspace(0.35, 0.85, len(int_imp)))
            bars2 = axes[1].barh(int_imp['label'], int_imp['importance'],
                                 color=colors_int, edgecolor='white', linewidth=0.4)
            axes[1].set_title('"How severe will it be?" — feature importance',
                              fontsize=11, pad=8)
            axes[1].set_xlabel('Influence on intensity estimate', fontsize=10)
            axes[1].invert_yaxis()
            for bar, val in zip(bars2, int_imp['importance']):
                axes[1].text(bar.get_width() + 0.001,
                             bar.get_y() + bar.get_height() / 2,
                             f'{val:.3f}', va='center', fontsize=8, color='#555')
        except Exception:
            axes[1].text(0.5, 0.5,
                         'Intensity model unavailable\n(too few storm samples)',
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=11, color='#888')
            axes[1].set_title('"How severe will it be?" — feature importance',
                              fontsize=11)

        plt.tight_layout()
        return fig


def run_demo():
    print("\n🌪️   Storm Prediction System — Live Demo")
    print("─" * 54)

    system = StormPredictionSystem()

    print("  → Loading oceanographic & atmospheric data…")
    data = system.generate_sample_data(2000)
    storms = data[data['storm_occurrence'] == 1]
    print(f"     {len(data):,} records — {len(storms):,} storm events "
          f"({len(storms)/len(data):.0%} of dataset)")
    if len(storms):
        print(f"     Avg intensity: {storms['storm_intensity'].mean():.1f}  "
              f"Max: {storms['storm_intensity'].max():.1f}")

    print("  → Training two models (occurrence + intensity)…")
    metrics, _ = system.train_models(data)
    print(f"     Occurrence accuracy: {metrics['occurrence_accuracy']:.2%}")
    print(f"     Intensity error (MAE): ±{metrics['intensity_mae']:.2f}")

    # Three realistic scenarios
    scenarios = [
        {
            "name": "Gulf of Mexico — hurricane season",
            "data": dict(wind_speed_kmh=75, central_pressure_hpa=960, sea_surface_temp_c=29,
                         coastal_slope_degrees=1.2, sea_elevation_m=200, latitude_degrees=25,
                         recent_storms=1, ocean_current_ms=1.2),
        },
        {
            "name": "North Atlantic — calm zone",
            "data": dict(wind_speed_kmh=15, central_pressure_hpa=1015, sea_surface_temp_c=18,
                         coastal_slope_degrees=6, sea_elevation_m=4000, latitude_degrees=45,
                         recent_storms=0, ocean_current_ms=-0.3),
        },
        {
            "name": "Western Pacific — typhoon region",
            "data": dict(wind_speed_kmh=110, central_pressure_hpa=920, sea_surface_temp_c=31,
                         coastal_slope_degrees=2.5, sea_elevation_m=500, latitude_degrees=15,
                         recent_storms=2, ocean_current_ms=2.0),
        },
    ]

    print("\n  → Running predictions…\n")
    for s in scenarios:
        alert = system.generate_storm_alert(s["data"], s["name"])
        intens_note = (f"  (predicted intensity {alert['predicted_intensity']:.1f})"
                       if alert['predicted_occurrence'] else "")
        print(f"  📍 {alert['location']}")
        print(f"     Risk:   {alert['risk_level']}{intens_note}  "
              f"({alert['storm_probability']:.0%} probability)")
        print(f"     Alert:  {alert['alert_message']}")
        print()

    print("  → Building feature importance charts…")
    fig = system.plot_feature_importance()
    plt.show()
    print("  Done! ✅\n")

    return system


if __name__ == "__main__":
    run_demo()
