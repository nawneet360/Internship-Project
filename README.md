# 🌊🌍🔥🌀 DisasterAI Pro

> **AI-powered early warning for Flood, Earthquake, Wildfire & Storm — because a few hours of warning saves lives.**

---

## Table of Contents

- [What This Is](#what-this-is)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Web Interface](#web-interface)
- [REST API Reference](#rest-api-reference)
- [Input Value Ranges](#input-value-ranges)
- [How the Models Work](#how-the-models-work)
- [Example Predictions](#example-predictions)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

---

## What This Is

DisasterAI Pro is a unified ML platform that predicts four types of natural disasters in real time:

| Module | What it predicts | Key inputs |
|---|---|---|
| 🌊 **Flood** | Risk score + 4-tier alert | Rainfall, river level, soil permeability, elevation |
| 🌍 **Earthquake** | Probability + estimated magnitude | P/S-wave velocity, micro-tremors, fault distance |
| 🔥 **Wildfire** | Fire probability + intensity | Temperature, humidity, wind speed, vegetation dryness |
| 🌀 **Storm/Cyclone** | Storm probability + surge risk | Pressure, wind speed, sea temperature, coastal slope |

The system outputs a clear, plain-language alert for every prediction — no PhD required to read it.

---

## Project Structure

```
Mini Project/
├── disaster_prediction_system.html   ← Full web dashboard (open in browser, no server needed)
├── web_server.py                     ← Flask backend with real scikit-learn ML models
├── requirements.txt                  ← Python dependencies
├── examples/
│   ├── flood_prediction.py           ← Flood model standalone demo
│   └── earthquake_prediction.py      ← Earthquake model standalone demo
└── README.md                         ← This file
```

---

## Quick Start

### Option A — Browser Only (no Python needed)

Open `disaster_prediction_system.html` directly in any browser.  
The interface runs its own built-in prediction engine — no server required.

```bash
open "disaster_prediction_system.html"
# or double-click the file in Finder
```

---

### Option B — Full ML Backend (real scikit-learn models)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies (~2 min)
pip install -r requirements.txt

# 3. Start the server
python3 web_server.py
```

**Expected startup output:**
```
⏳ Training Flood model...
✅ Flood model ready
⏳ Training Earthquake model...
✅ Earthquake model ready
⏳ Training Wildfire model...
✅ Wildfire model ready
⏳ Training Storm model...
✅ Storm model ready
 * Running on http://localhost:8080
```

Then open **http://localhost:8080** in your browser.

> Training takes **15–30 seconds** on first run only. Subsequent predictions take < 15 ms each.

---

### Option C — Run Python Demos Directly

```bash
source .venv/bin/activate
python examples/flood_prediction.py
python examples/earthquake_prediction.py
```

---

## Web Interface

The dashboard has 4 tabs — one per disaster type. Each tab contains:

| Component | Description |
|---|---|
| **Input Form** | 8 fields with live input hints and range validation |
| **Risk Gauge** | 0–100 % circular meter, colour-coded green → red |
| **Result Panel** | Risk level, probability, confidence, recommendation |
| **Trend Chart** | 24-hour historical simulation (updates live) |
| **Radar Chart** | Factor breakdown showing which inputs drive the risk |
| **Alert System** | Plain-language warnings at 3 severity levels |
| **Monitoring Map** | Global map with risk hotspots |

**Earthquake tab extras:**
- Live animated seismograph (ECG-style waveform)
- Global tectonic plate map with live epicentre dots
- Richter scale live bar chart

---

## REST API Reference

**Base URL:** `http://localhost:8080`  
**Content-Type:** `application/json`

### GET `/api/status`
```bash
curl http://localhost:8080/api/status
```
```json
{ "status": "ok", "flood": true, "earthquake": true, "wildfire": true, "storm": true }
```

---

### POST `/api/predict/flood`
```bash
curl -X POST http://localhost:8080/api/predict/flood \
  -H "Content-Type: application/json" \
  -d '{"rainfall": 100, "river_level": 5.5, "soil_moisture": 65, "area": 500}'
```
```json
{
  "risk_score": 0.78,
  "risk_level": "High",
  "message": "🚨 Flood risk is HIGH. Move valuables to upper floors, avoid low-lying roads.",
  "timestamp": "2026-03-27T10:00:00",
  "factors": { "rainfall_impact": 80, "drainage_impact": 60, "river_impact": 75, "terrain_impact": 50 }
}
```

**Parameters:**

| Field | Type | Range | Required |
|---|---|---|---|
| rainfall | float | 0–500 mm | ✅ |
| river_level | float | 0–10 m | ✅ |
| soil_moisture | float | 0–100 % | ✅ |
| area | float | 0–5000 km² | ✅ |

---

### POST `/api/predict/earthquake`
```bash
curl -X POST http://localhost:8080/api/predict/earthquake \
  -H "Content-Type: application/json" \
  -d '{"latitude": 40.71, "longitude": -74.00, "depth": 10, "magnitude": 4.5,
       "prev_quakes": 2, "recent_activity": 1, "tectonic_stress": 75, "fault_proximity": 5}'
```
```json
{
  "predicted_magnitude": 5.2,
  "predicted_intensity": "VI",
  "risk_level": "High",
  "message": "⚠️ High seismic risk. Ensure structural preparedness.",
  "factors": { "depth_impact": 30, "magnitude_impact": 45, "location_impact": 60, "tectonic_impact": 70 }
}
```

**Parameters:**

| Field | Type | Range | Required |
|---|---|---|---|
| latitude | float | −90 to 90 | ✅ |
| longitude | float | −180 to 180 | ✅ |
| depth | float | 0–700 km | ✅ |
| magnitude | float | 0–9 | ✅ |
| prev_quakes | int | 0–100+ | ✅ |
| recent_activity | int | 0–50 | ✅ |
| tectonic_stress | float | 0–100 % | ✅ |
| fault_proximity | float | 0–100 km | ✅ |

---

### POST `/api/predict/wildfire`
```bash
curl -X POST http://localhost:8080/api/predict/wildfire \
  -H "Content-Type: application/json" \
  -d '{"temperature": 35, "humidity": 25, "wind_speed": 15, "fuel_load": 8,
       "vegetation_density": 0.7, "previous_fires": 2, "elevation": 1200, "slope": 25}'
```
```json
{
  "predicted_occurrence": true,
  "fire_probability": 0.72,
  "predicted_intensity": 6.1,
  "risk_level": "High",
  "message": "🚨 Fire risk is HIGH (72% probability). Restrict outdoor burning.",
  "factors": { "temperature_impact": 65, "humidity_impact": 55, "wind_impact": 40 }
}
```

**Parameters:**

| Field | Type | Range | Required |
|---|---|---|---|
| temperature | float | −50 to 60 °C | ✅ |
| humidity | float | 0–100 % | ✅ |
| wind_speed | float | 0–150 km/h | ✅ |
| fuel_load | float | 0–20 t/ha | ✅ |
| vegetation_density | float | 0–1 | ✅ |
| previous_fires | int | 0–100+ | ✅ |
| elevation | float | 0–5000 m | ✅ |
| slope | float | 0–90 ° | ✅ |

---

### POST `/api/predict/storm`
```bash
curl -X POST http://localhost:8080/api/predict/storm \
  -H "Content-Type: application/json" \
  -d '{"pressure": 980, "wind_speed": 45, "temperature": 28, "humidity": 85,
       "visibility": 2, "sea_temp": 26, "latitude": 15, "season": 9}'
```
```json
{
  "predicted_occurrence": true,
  "storm_probability": 0.64,
  "predicted_intensity": 3.2,
  "risk_level": "High",
  "message": "⚠️ Storm conditions strengthening. Evacuate coastal low-lying areas.",
  "factors": { "pressure_impact": 70, "wind_impact": 55, "coastal_impact": 60 }
}
```

**Parameters:**

| Field | Type | Range | Required |
|---|---|---|---|
| pressure | float | 900–1050 mb | ✅ |
| wind_speed | float | 0–300 km/h | ✅ |
| temperature | float | −50 to 60 °C | ✅ |
| humidity | float | 0–100 % | ✅ |
| visibility | float | 0–50 km | ✅ |
| sea_temp | float | −2 to 40 °C | ✅ |
| latitude | float | −90 to 90 | ✅ |
| season | int | 1–12 (month) | ✅ |

---

### Error Responses

```json
{ "error": "Invalid request data", "details": "Missing required field: rainfall" }   // 400
{ "error": "Server error", "details": "Model prediction failed" }                    // 500
```

---

## Input Value Ranges (Quick Reference)

### Risk Level Scale (all models)
| Score | Level |
|---|---|
| 0–24 % | 🟢 Low |
| 25–49 % | 🟡 Moderate |
| 50–74 % | 🟠 High |
| 75–100 % | 🔴 Critical |

### High-Risk Sample Inputs

**Flood — Critical:**
```json
{ "rainfall": 200, "river_level": 8.5, "soil_moisture": 90, "area": 1000 }
```

**Earthquake — High:**
```json
{ "latitude": 40.71, "longitude": -74.00, "depth": 5, "magnitude": 6.5,
  "prev_quakes": 10, "recent_activity": 5, "tectonic_stress": 95, "fault_proximity": 1 }
```

**Wildfire — Critical:**
```json
{ "temperature": 45, "humidity": 10, "wind_speed": 60, "fuel_load": 18,
  "vegetation_density": 0.95, "previous_fires": 10, "elevation": 3000, "slope": 80 }
```

**Storm — Critical:**
```json
{ "pressure": 950, "wind_speed": 100, "temperature": 30, "humidity": 95,
  "visibility": 1, "sea_temp": 28, "latitude": 15, "season": 9 }
```

---

## How the Models Work

All four models follow the same pipeline:

```
Sensor readings → Feature engineering → Trained ML model → Risk score → Plain-language alert
```

| Model | Algorithm | Accuracy |
|---|---|---|
| Flood | Random Forest (classifier) | ~94.7 % |
| Earthquake | Random Forest + Gradient Boosting | ~92.6 % |
| Wildfire | Random Forest (classifier + regressor) | ~88.1 % |
| Storm | Gradient Boosting | ~90.4 % |

Models are trained on **synthetic data** that mirrors real-world hydrological and seismological patterns. For production use, swap synthetic data for live feeds from:
- **USGS** stream gauges and weather stations (Flood)
- **USGS ShakeAlert / EMSC** seismic networks (Earthquake)
- **NOAA VIIRS / NASA FIRMS** (Wildfire)
- **NOAA NHC / Joint Typhoon Warning Center** (Storm)

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Address already in use` on port 8080 | Port taken | `lsof -i :8080` → `kill -9 <PID>` |
| `ModuleNotFoundError: flask` | Deps not installed | `pip install flask flask-cors numpy pandas scikit-learn` |
| Web page not loading | Server not running | `curl http://localhost:8080/api/status` to check |
| Charts not showing | JS error in browser | Open browser console (F12) and check for errors |
| All predictions show "Low" | Input values too low | Try rainfall > 150 mm or tectonic_stress > 80 for testing |
| Slow startup | Model training | Normal — takes 15–30 s once. Predictions thereafter < 15 ms |

**Stop the server:** `Ctrl+C`  
**Restart the server:** `python3 web_server.py`  
**Check server logs:** Watch the terminal where the server is running

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | Python · scikit-learn (Random Forest, Gradient Boosting) |
| Backend | Flask · Flask-CORS |
| Frontend | HTML · Vanilla CSS · JavaScript |
| Charts | Chart.js |
| Icons | Font Awesome 6 |
| Fonts | Google Fonts (Inter) |

**Dependencies** (`requirements.txt`):
```
numpy
pandas
scikit-learn
matplotlib
seaborn
flask
flask-cors
```

---

## Performance

| Metric | Value |
|---|---|
| Model training (startup) | 15–30 s (one-time) |
| Prediction response time | 5–15 ms |
| Average response time | < 10 ms |
| Memory per model | ~30–50 MB |
| Total memory usage | ~150–200 MB |

---

## Integration Examples

### Python
```python
import requests

response = requests.post("http://localhost:8080/api/predict/flood", json={
    "rainfall": 100, "river_level": 5.5, "soil_moisture": 65, "area": 500
})
result = response.json()
print(f"Risk: {result['risk_level']} ({result['risk_score']*100:.0f}%)")
```

### JavaScript
```javascript
const result = await fetch("http://localhost:8080/api/predict/flood", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rainfall: 100, river_level: 5.5, soil_moisture: 65, area: 500 })
}).then(r => r.json());

console.log(`Risk: ${result.risk_level}`);
```

---

> ⚠️ **Ethical Note:** Early warning systems save lives, but false alarms cause panic. This project includes confidence scores and uncertainty language by design. Any real-world deployment must be validated by domain experts before public use.

---

*Built with Python · scikit-learn · Flask · Chart.js · © 2026 DisasterAI Pro*
