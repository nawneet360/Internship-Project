
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FloodPredictionSystem:
 
    def __init__(self):
    
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = None

    def generate_sample_data(self, n_samples=1000):
        
        np.random.seed(42)
        rainfall      = np.random.gamma(2, 10, n_samples)     
        rainfall_7day = np.random.gamma(1.5, 8, n_samples)      
        temperature   = np.random.normal(20, 10, n_samples)     
        humidity      = np.random.uniform(30, 100, n_samples)   

        river_level       = np.random.normal(5, 2, n_samples)    
        elevation         = np.random.uniform(0, 500, n_samples) 
        slope             = np.random.uniform(0, 30, n_samples)  
        soil_permeability = np.random.uniform(0.1, 1.0, n_samples)  

    
        flood_risk = (
            0.30 * (rainfall / 50) +                       
            0.20 * (rainfall_7day / 40) +                  
            0.20 * np.maximum(0, (river_level - 3) / 10) +
            0.10 * (humidity / 100) +                      
            0.10 * np.maximum(0, (1 - elevation / 100)) +  
            0.05 * np.maximum(0, (1 - slope / 20)) +     
            0.05 * (1 - soil_permeability)                  
        )

        flood_risk = np.clip(
            flood_risk + np.random.normal(0, 0.1, n_samples), 0, 1
        )

        return pd.DataFrame({
            'rainfall_mm':        rainfall,
            'river_level_m':      river_level,
            'temperature_c':      temperature,
            'humidity_pct':       humidity,
            'elevation_m':        elevation,
            'slope_degrees':      slope,
            'soil_permeability':  soil_permeability,
            'rainfall_7day_avg':  rainfall_7day,
            'flood_risk':         flood_risk,
        })



    def train_model(self, data):
        X = data.drop('flood_risk', axis=1)
        y = data['flood_risk']
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_pred = self.model.predict(X_train)
        test_pred  = self.model.predict(X_test)

        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae':  mean_absolute_error(y_test,  test_pred),
            'train_r2':  r2_score(y_train, train_pred),
            'test_r2':   r2_score(y_test,  test_pred),
        }

        return metrics, (X_test, y_test, test_pred)



    def predict_flood_risk(self, weather_data):
       
        if not self.is_trained:
            raise RuntimeError(
                "Please train the model first by calling train_model()."
            )

        if isinstance(weather_data, dict):
            weather_data = pd.DataFrame([weather_data])

        return self.model.predict(weather_data)

    def risk_level(self, score):
        
        if score < 0.20:
            return "Low"
        elif score < 0.40:
            return "Moderate"
        elif score < 0.70:
            return "High"
        else:
            return "Critical"

    def generate_alert(self, weather_data, location="your area"):
        """
        Produce a complete, actionable flood alert.

        Returns a dictionary with everything a first responder or citizen
        needs to know: when, where, how bad, and what to do next.
        """
        score = self.predict_flood_risk(weather_data)[0]
        level = self.risk_level(score)

        messages = {
            "Low":      (f"Conditions in {location} look normal right now. "
                         f"No immediate flood concern (score: {score:.0%})."),
            "Moderate": (f"⚠️  Keep an eye on the weather in {location}. "
                         f"Conditions could worsen — stay informed (score: {score:.0%})."),
            "High":     (f"🚨 Flood risk is HIGH in {location}. "
                         f"Move valuables to upper floors, avoid low-lying roads "
                         f"(score: {score:.0%})."),
            "Critical": (f"🚨🚨 CRITICAL FLOOD WARNING for {location}. "
                         f"Evacuate immediately if instructed. Do not attempt to "
                         f"cross flooded roads (score: {score:.0%})."),
        }

        return {
            'timestamp':     datetime.now().isoformat(),
            'location':      location,
            'risk_score':    round(float(score), 4),
            'risk_level':    level,
            'message':       messages[level],
        }



    def plot_feature_importance(self):
       
        if not self.is_trained:
            raise RuntimeError("Train the model before plotting feature importance.")

        importance_df = pd.DataFrame({
            'feature':    self.feature_names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)

        labels = {
            'rainfall_mm':       'Today\'s Rainfall (mm)',
            'rainfall_7day_avg': '7-Day Avg Rainfall (mm)',
            'river_level_m':     'River Level (m)',
            'humidity_pct':      'Humidity (%)',
            'elevation_m':       'Elevation (m)',
            'slope_degrees':     'Ground Slope (°)',
            'soil_permeability': 'Soil Drainage',
            'temperature_c':     'Temperature (°C)',
        }
        importance_df['label'] = importance_df['feature'].map(labels)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn_r(
            np.linspace(0.2, 0.9, len(importance_df))
        )
        bars = ax.barh(
            importance_df['label'], importance_df['importance'],
            color=colors, edgecolor='white', linewidth=0.5
        )
        ax.set_xlabel('How much this factor influences the flood risk score',
                      fontsize=11)
        ax.set_title('What drives flood risk? — Feature importance',
                     fontsize=13, fontweight='bold', pad=12)
        ax.invert_yaxis()

        for bar, val in zip(bars, importance_df['importance']):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9, color='#444')

        plt.tight_layout()
        return fig




def run_demo():
    print("\n🌊  Flood Early Warning System — Live Demo")
    print("─" * 52)

    system = FloodPredictionSystem()

    print("  → Loading weather & terrain data…")
    data = system.generate_sample_data(1000)
    print(f"     {len(data):,} historical records ready")

    print("  → Training the model (this takes a few seconds)…")
    metrics, _ = system.train_model(data)
    print(f"     Accuracy (R²):  {metrics['test_r2']:.2%}")
    print(f"     Avg error (MAE): {metrics['test_mae']:.4f}")

    scenarios = [
        {
            "name": "Mumbai — monsoon conditions",
            "data": dict(rainfall_mm=62, river_level_m=8.2, temperature_c=28,
                         humidity_pct=97, elevation_m=8, slope_degrees=1,
                         soil_permeability=0.2, rainfall_7day_avg=48),
        },
        {
            "name": "Denver — dry summer day",
            "data": dict(rainfall_mm=3, river_level_m=3.1, temperature_c=32,
                         humidity_pct=38, elevation_m=1609, slope_degrees=12,
                         soil_permeability=0.75, rainfall_7day_avg=5),
        },
    ]

    print("\n  → Running predictions…\n")
    for s in scenarios:
        alert = system.generate_alert(s["data"], s["name"])
        print(f"  📍 {alert['location']}")
        print(f"     Risk:    {alert['risk_level']}  ({alert['risk_score']:.0%})")
        print(f"     Alert:   {alert['message']}")
        print()

    print("  → Building feature importance chart…")
    fig = system.plot_feature_importance()
    plt.show()
    print("  Done! ✅\n")

    return system


if __name__ == "__main__":
    run_demo()
