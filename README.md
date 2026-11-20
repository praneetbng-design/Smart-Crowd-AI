# Smart Crowd AI: Real-Time Crowd Monitoring & Stampede Prevention System

A production-grade, dark-themed computer vision web application built with **Python**, **Streamlit**, **OpenCV**, **YOLOv8**, **Plotly**, **Folium**, and **SQLite**. Designed for high-density environments such as stadiums, transit terminals, pilgrimage plazas, and festival grounds to prevent crush hazards and stampedes.

---

## 🌟 Key Capabilities

1. **Crowd Monitoring Dashboard**:
   - Live KPI metric cards for monitored population, sector risk status, peak local density, and active alerts.
   - Dynamic Plotly time-series timeline chart tracking crowd count and density with warning/critical threshold lines.
   - Stampede Hazard Index gauge (0–100 scale).
   - Zone capacity saturation comparison bar charts.
   - Multi-venue selection (Central Mega Arena, Grand Transit Terminal, Pilgrimage Sacred Square, Festival Parklands).

2. **Image & Video AI Scanner**:
   - Object detection powered by **YOLOv8-Nano** (`person` class) with graceful fallback to **OpenCV HOG Pedestrian Classifier** and synthetic silhouette analysis.
   - Gaussian crowd density heatmap overlay with smooth color blending.
   - Bounding boxes with confidence ratings and pedestrian centroid tracking coordinates.
   - Video stream analyzer with configurable frame stride and temporal surge curves.
   - Preloaded sample datasets for instant 1-click testing without needing manual uploads.

3. **Interactive Multi-Sector Zone Map**:
   - Dynamic Folium dark-mode map displaying safe (green), warning (amber), critical (orange), and stampede risk (red) zones.
   - Global crowd density heatmap layer with color gradients.
   - Interactive popups with capacity statistics and Fruin Level of Service ratings.
   - Sector registration form to add custom coordinates and venues dynamically.

4. **Configurable Risk Scoring & Automatic Alerts**:
   - Risk scoring algorithm incorporating crowd density (people/m²), capacity saturation percentage, surge velocity (people entering per second), and spatial clustering.
   - Configurable sliders for Warning, Critical, and Stampede risk density thresholds.
   - Incident acknowledgment workflow.
   - 100% offline Web Audio API synthesized alert chimes for critical warnings.
   - Prescriptive operational directives for venue marshals (gate metering, emergency exit opening, medical dispatch).

5. **Persistent SQLite Audit & Detection Logs**:
   - SQLite database (`data/crowd_data.db`) storing all detections, risk ratings, and alert history.
   - Filter by source type (Image, Video, Live, Simulation) and safety status.
   - One-click CSV export button.
   - Database reset and cleanup controls.

6. **Live Surge Demo Simulator**:
   - Zero-hardware demonstration mode for testing when physical CCTV feeds or models are offline.
   - Scenarios: *Normal Free Flow*, *Metro Evening Rush Hour*, *Concert Gate Ingress Bottleneck*, *Severe Stampede & Chokepoint Crush*.
   - Explicit badges distinguishing real AI detections from simulated telemetry.

---

## 📐 Crowd Dynamics Theory: Fruin's Level of Service (LOS)

The risk engine incorporates John J. Fruin's standard *Pedestrian Planning and Design* crowd dynamics:
- **LOS A/B (< 1.0 p/m²)**: Free pedestrian flow; unimpeded reverse and crossing movements.
- **LOS C/D (1.0 – 2.0 p/m²)**: Normal flow with minor speed restriction; voluntary path selection.
- **LOS E (2.0 – 3.5 p/m²)**: Capacity limit; involuntary physical contact and reduced transit velocity.
- **LOS F (> 3.5 p/m²)**: Breakdown of crowd movement; compressive pressure waves, shockwaves, and high stampede risk.

---

## 📁 Project Structure

```
smart-crowd-ai/
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Dependency specifications
├── run.bat                     # Windows 1-click launcher script
├── README.md                   # System documentation
├── src/
│   ├── database.py             # SQLite schema, queries, and CSV export
│   ├── detector.py             # YOLOv8 & OpenCV HOG detection engine
│   ├── risk_engine.py          # Stampede risk formulas and advisories
│   ├── map_module.py           # Folium interactive map & heatmaps
│   ├── simulation.py           # Synthetic crowd generator & surge presets
│   └── ui_components.py        # Dark theme CSS, Plotly gauges, audio alarms
├── assets/                     # Pre-generated sample images and surge video
│   ├── sample_safe_crowd.jpg
│   ├── sample_warning_crowd.jpg
│   ├── sample_critical_surge.jpg
│   ├── sample_stampede_risk.jpg
│   └── sample_crowd_surge.mp4
└── data/
    └──                           # SQLite database is created locally at runtime
```

---

## 🚀 How to Run the Application

### Option 1: One-Click Batch Script (Windows)
First run `setup_env.bat` to install dependencies, then double-click `run.bat`.

### Option 2: Command Line
From the project directory, create the environment, install dependencies, and start the app:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open your browser to:
```
http://localhost:8501
```

---

## 🛡️ License & Privacy
100% free and open-source tools only. No paid APIs, no subscriptions, and all data is processed strictly locally on your machine.
