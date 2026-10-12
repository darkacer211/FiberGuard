# FiberGuard: AI-Driven Optical Telemetry & Fault Diagnosis Suite

![FiberGuard Banner](https://img.shields.io/badge/Optical_Standard-ITU--T_G.652.D_%2F_G.657-0284c7)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI_%2B_Uvicorn-009688)
![ML](https://img.shields.io/badge/ML_Engine-Scikit--Learn-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-91.1%25_Holdout-10b981)

**FiberGuard** is a carrier-grade, physics-accurate simulation and autonomous AI fault diagnostic platform for single-mode optical fiber communications (1550 nm C-Band). It integrates mathematical physical optics modeling, realistic Optical Time-Domain Reflectometry (OTDR) signal processing, multi-trace eye diagram analysis, and real machine learning classifiers (Random Forest, Gradient Boosting, k-NN, and Neural Networks) to instantly detect, classify, and localize physical optical link impairments.

---

## 🌟 Key Features

### 1. 🛰️ Interactive Link Operations & Real-Time AI Diagnostics
- **Live Optical Waveguide Visualizer:** Real-time canvas animation displaying optical pulse propagation through SMF-28 fiber, power-gradient attenuation, dispersion-induced temporal broadening, and backward Rayleigh/Fresnel reflections.
- **Twin High-Resolution Scopes:**
  - **OTDR Trace Oscilloscope:** Distance-resolved backscatter curve ($dB$ vs $km$) with interactive crosshair inspection, splice steps, Fresnel reflection spikes, noise floor, and AI localization flags.
  - **Receiver Eye Diagram Scope:** Simulated multi-trace PRBS waveform with intersymbol interference (ISI), Gaussian jitter, Amplified Spontaneous Emission (ASE) noise rails, and vertical eye-opening metrics.
- **Real-Time Telecom KPIs:** Received Power ($P_{rx}$ with sensitivity margin), Optical Signal-to-Noise Ratio (OSNR), Effective Q-Factor ($Q_{eff}$), Bit Error Rate (BER with scientific notation), and Pulse Broadening ($\Delta\tau$).
- **Itemized Power Budget Waterfall:** Complete optical loss accounting table with algebraic formula tracking.

### 2. 📚 Comprehensive Optical Physics & Theory Handbook
- In-depth, illustrated theoretical breakdown with mathematical formulations:
  - **Beer-Lambert Law & Rayleigh Scattering:** Exponential attenuation $\alpha \approx 0.20$ dB/km at 1550 nm.
  - **OTDR Reflectometry & Event Signatures:** Round-trip timing $z = c \cdot t / (2 n_g)$, non-reflective vs Fresnel reflective boundaries.
  - **Macro-Bending & Micro-Bending:** Critical angle violation, mode field leakage, ITU-T G.652.D vs G.657 bend-insensitive fiber.
  - **EDFA Amplification & ASE Noise:** 980 nm / 1480 nm pump degradation, population inversion, Noise Figure (NF), and OSNR equations.
  - **Chromatic Dispersion & Intersymbol Interference:** Pulse spreading $\Delta\tau = D \cdot L \cdot \Delta\lambda$ with $D = 17$ ps/(nm·km).
  - **Signal Integrity Math:** Q-Factor relation to SNR, eye opening penalty, and Complementary Error Function $BER = \frac{1}{2}\text{erfc}(Q/\sqrt{2})$.
- **Interactive Optical Link Calculator:** Test custom fiber length, launch power, attenuation, dispersion, and bit rate dynamically.

### 3. 🧠 Machine Learning Studio & Benchmarks
- Real scikit-learn models trained on rigorous physics-generated datasets with measurement noise:
  - **Random Forest Classifier** (100 Decision Trees, Max Depth 14)
  - **Gradient Boosting Classifier** (80 Boosted Trees)
  - **k-Nearest Neighbors** (Distance-weighted $k=9$)
  - **Multi-Layer Perceptron** (Neural Network with $(64, 32)$ architecture)
- Genuine evaluation on an independent **1,050 holdout test set**:
  - **91.1% Overall Accuracy** (vs. only 63.8% for legacy 3 dB power threshold alarms).
  - Full **7×7 Confusion Matrix Heatmap** with normalized percentage tooltips.
  - Per-class Precision, Recall, and F1-Scores for all 7 impairment categories.
  - Feature Importance rankings (Mean Decrease in Impurity).
  - On-demand server-side model retraining.

### 4. 🔬 Diagnostic Sandbox & Incident Lab
- **6 Real-World Pre-Configured Outage Incidents:**
  1. *Major Metro Conduit Trenching Sever* (Fiber Cut at 41.6 km)
  2. *Splice Tray Pinch in Manhole 14* (Macro-Bend 3.8 dB drop)
  3. *Contaminated Central Office Patch Cassette* (Dirty Connector 2.4 dB drop + 7.5 dB reflection)
  4. *Underground Cable Water Ingress* (Aged fiber 0.36 dB/km attenuation)
  5. *Repeater EDFA Pump Degradation* (Gain drop to 3 dB, NF surge to 14.5 dB)
  6. *Uncompensated 40G Chromatic Dispersion* (Pulse spread 2.3× bit period)
- **Manual Optical Telemetry Form:** Input field instrument readings directly to obtain an instant AI root cause diagnosis and field restoration action protocol.
- **Formal Incident Restoration Report:** Formatted printable and exportable field report.

### 5. 📊 Telemetry Dataset Studio
- Live synthetic telemetry dataset explorer.
- Direct CSV export (`/api/dataset/export.csv`) for use in Python, Jupyter, or MATLAB.

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10+
- Web browser (Chrome, Edge, Firefox, Safari)

### 1. Clone or Open Project
```bash
cd "c:\Users\athar\Optical communication"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Application
Simply run the one-click launcher:
```bash
python run.py
```
This will:
1. Load or train the cached machine learning models.
2. Start the Uvicorn FastAPI backend at `http://127.0.0.1:8000`.
3. Automatically launch your default web browser to the FiberGuard interface.

Alternatively, run with Uvicorn directly:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## ⚡ Deploy to Vercel

FiberGuard is configured for zero-config deployment on **Vercel** with support for Serverless Python (`api/index.py`) and static asset hosting (`public/`).

### Deploy via Vercel Dashboard (Easiest)
1. Go to [vercel.com](https://vercel.com) and log in with your GitHub account.
2. Click **"Add New..."** -> **"Project"**.
3. Select your repository: **`darkacer211/FiberGuard`**.
4. Leave the default settings (Framework Preset: **Other**, Root Directory: **`./`**).
5. Click **"Deploy"**.

### Deploy via Vercel CLI
```bash
npm i -g vercel
vercel
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Health check, active model, and system status |
| `/api/simulate` | `POST` | Runs optical physics simulation & OTDR trace synthesis |
| `/api/predict` | `POST` | AI fault classification, probabilities & distance localization |
| `/api/models` | `GET` | Benchmark metrics, confusion matrices & feature importances |
| `/api/models/select` | `POST` | Switches active inference model |
| `/api/models/train` | `POST` | Retrains model with custom sample size |
| `/api/dataset/sample` | `GET` | Fetches sample telemetry records for explorer |
| `/api/dataset/export.csv` | `GET` | Downloads full synthetic dataset as CSV |
| `/api/diagnose-custom` | `POST` | Field engineer manual instrument diagnosis |

---

## 🔬 Mathematical Physics Reference

### 1. Power Budget Equation
$$P_{rx} = P_{tx} - \alpha \cdot L - \sum L_{splice} - \sum L_{connector} - L_{fault} + G_{EDFA}$$

### 2. Optical Signal-to-Noise Ratio (OSNR)
$$OSNR \approx 58 + P_{in} - NF \quad [\text{dB, ref 0.1 nm}]$$

### 3. Temporal Pulse Broadening (GVD)
$$\Delta \tau = D \cdot L \cdot \Delta \lambda \quad [\text{ps}]$$

### 4. Q-Factor & Intersymbol Interference
$$Q = Q_0 \cdot \text{EyeOpening}(\sigma_{eff})$$

### 5. Bit Error Rate (BER)
$$BER = \frac{1}{2} \text{erfc}\left(\frac{Q}{\sqrt{2}}\right)$$

---

## 🏛️ Project Structure

```
Optical communication/
├── backend/
│   ├── __init__.py
│   ├── physics_engine.py       # Optical link propagation, dispersion, EDFA & OTDR physics
│   ├── feature_extractor.py    # Industrial OTDR digital signal processing & feature extraction
│   ├── ml_engine.py            # Scikit-learn pipelines, benchmarking & serialization
│   ├── models_cache/           # Cached pre-trained joblib weights and benchmark metrics
│   └── main.py                 # FastAPI REST API & static file server
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphism, cyber-optical styling & responsive design
│   ├── js/
│   │   ├── api.js              # REST client with automatic fallback handling
│   │   ├── client-physics.js   # Client-side physics & k-NN fallback engine
│   │   ├── link-view.js        # Waveguide canvas visualizer
│   │   ├── otdr-scope.js       # Interactive OTDR oscilloscope canvas
│   │   ├── eye-scope.js        # High-speed receiver eye diagram canvas
│   │   └── app.js              # Tab controller, KPIs, calculator & incident lab
│   └── index.html              # Multi-tab web interface
├── run.py                      # Application launcher script
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation & specifications
```

---

## 📜 License
MIT License - Developed for Advanced Optical Telecommunication Research & Educational Demonstration.
