"""
FiberGuard Backend - FastAPI Application
Provides RESTful APIs for optical link simulation, digital signal processing,
multi-model machine learning inference, and telemetry dataset access.
"""

import io
import csv
import math
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from backend.physics_engine import (
    OpticalLinkParams, simulate_optical_physics, generate_otdr_trace,
    FAULT_NAMES, FAULT_SHORT, FAULT_COLORS, FAULT_DESCRIPTIONS,
    ALPHA0, FRONT_CONN, END_CONN, SPLICE_LOSS, G0, NF0
)
from backend.feature_extractor import analyze_otdr_trace, extract_features, FEATURE_NAMES
from backend.ml_engine import ml_engine

app = FastAPI(
    title="FiberGuard API",
    description="AI-driven Fault Detection & Telemetry Analytics for Optical Fiber Communications",
    version="2.0.0"
)

# Startup event: ensure ML models are trained and cached
@app.on_event("startup")
async def startup_event():
    print("[FastAPI] Initializing FiberGuard Machine Learning models...")
    ml_engine.train_or_load_all()
    print("[FastAPI] Models ready!")


# Request & Response Schemas
class SimulationRequest(BaseModel):
    length_km: float = Field(80.0, ge=10.0, le=150.0)
    launch_power_dbm: float = Field(0.0, ge=-10.0, le=15.0)
    bit_rate_gbps: float = Field(10.0, ge=1.0, le=100.0)
    fault_type: int = Field(0, ge=0, le=6)
    fault_pos_ratio: float = Field(0.45, ge=0.05, le=0.95)
    severity: float = Field(0.5, ge=0.0, le=1.0)
    seed: Optional[int] = 42


class PredictionRequest(BaseModel):
    features: Optional[List[float]] = None
    simulation_params: Optional[SimulationRequest] = None
    model_name: Optional[str] = None


class SelectModelRequest(BaseModel):
    model_name: str


class TrainModelRequest(BaseModel):
    model_name: str
    n_train: int = 3500


class CustomTelemetryRequest(BaseModel):
    fiber_length_km: float = 80.0
    measured_prx_dbm: float = -15.0
    measured_osnr_db: float = 24.0
    measured_alpha_db_km: float = 0.20
    step_loss_db: float = 0.0
    reflection_spike_db: float = 0.0
    detected_fiber_end_km: float = 80.0
    measured_pulse_spread_ps: float = 27.2
    model_name: Optional[str] = None


@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "system": "FiberGuard Optical AI Suite",
        "version": "2.0.0",
        "active_model": ml_engine.active_model_name,
        "available_models": list(ml_engine.models.keys())
    }


@app.post("/api/simulate")
def simulate_link(req: SimulationRequest):
    """Runs complete optical physics simulation and OTDR trace generation."""
    rng = None
    if req.seed is not None:
        import numpy as np
        rng = np.random.default_rng(req.seed)

    params = OpticalLinkParams(
        length_km=req.length_km,
        launch_power_dbm=req.launch_power_dbm,
        bit_rate_gbps=req.bit_rate_gbps,
        fault_type=req.fault_type,
        fault_pos_ratio=req.fault_pos_ratio,
        severity=req.severity
    )

    telemetry = simulate_optical_physics(params, rng)
    otdr = generate_otdr_trace(telemetry, rng, num_points=512)
    analysis = analyze_otdr_trace(otdr['trace'], otdr['dx'], req.length_km)
    features = extract_features(telemetry, analysis)

    # Power budget breakdown table
    p_margin = telemetry.prx_dbm - telemetry.receiver_sensitivity_dbm
    budget_items = [
        {"item": "Launch Power (P_tx)", "value": f"{telemetry.params.launch_power_dbm:+.1f} dBm", "note": "Transmitter optical source", "status": "neutral"},
        {"item": "Tx Patch Connector", "value": f"{-FRONT_CONN:+.1f} dB", "note": "Front panel insertion loss", "status": "neutral"},
        {"item": "Fiber Attenuation Loss", "value": f"{-telemetry.fiber_loss_db:+.1f} dB", "note": f"{telemetry.attenuation_alpha:.2f} dB/km × {telemetry.fiber_length:.0f} km", "status": "neutral" if telemetry.attenuation_alpha <= 0.22 else "bad"},
        {"item": "Fusion Splice Losses", "value": f"{-telemetry.splice_loss_db:+.1f} dB", "note": f"{telemetry.num_splices} splices @ {SPLICE_LOSS:.1f} dB/splice", "status": "neutral"},
        {"item": "Rx Patch Connector", "value": f"{-END_CONN:+.1f} dB", "note": "Rear panel insertion loss", "status": "neutral"},
        {"item": "Fault Excess Loss", "value": "-∞ dB" if telemetry.is_cut else f"{-telemetry.extra_loss_db:+.1f} dB", "note": "Fiber severed" if telemetry.is_cut else ("Bend / dirt loss" if telemetry.extra_loss_db > 0 else "None"), "status": "bad" if (telemetry.extra_loss_db > 0 or telemetry.is_cut) else "good"},
        {"item": "EDFA Input Power (P_in)", "value": "None" if telemetry.is_cut else f"{telemetry.pin_edfa_dbm:+.1f} dBm", "note": "Subtotal before amplification", "status": "sum"},
        {"item": "EDFA Booster Gain", "value": f"{telemetry.edfa_gain_db:+.1f} dB", "note": "Degraded pump" if telemetry.params.fault_type == 5 else f"Nominal gain (+{G0:.0f} dB)", "status": "bad" if telemetry.params.fault_type == 5 else "neutral"},
        {"item": "Receiver Power (P_rx)", "value": "LOS" if telemetry.is_cut else f"{telemetry.prx_dbm:+.1f} dBm", "note": "Total incident light at photodiode", "status": "sum"},
        {"item": "Receiver Sensitivity", "value": f"{telemetry.receiver_sensitivity_dbm:+.1f} dBm", "note": f"PIN photodiode for {telemetry.params.bit_rate_gbps:.1f} Gb/s", "status": "neutral"},
        {"item": "Link Power Margin", "value": "Loss of Light" if telemetry.is_cut else f"{p_margin:+.1f} dB", "note": "P_rx - P_sens", "status": "good" if p_margin >= 3.0 else ("warn" if p_margin >= 0.0 else "bad")}
    ]

    return {
        "params": {
            "length_km": telemetry.fiber_length,
            "launch_power_dbm": telemetry.params.launch_power_dbm,
            "bit_rate_gbps": telemetry.params.bit_rate_gbps,
            "fault_type": telemetry.params.fault_type,
            "fault_name": FAULT_NAMES[telemetry.params.fault_type],
            "fault_pos_ratio": telemetry.params.fault_pos_ratio,
            "fault_km": telemetry.fault_km,
            "severity": telemetry.params.severity
        },
        "telemetry": {
            "prx_dbm": round(telemetry.prx_dbm, 2),
            "measured_prx_dbm": round(telemetry.measured_prx_dbm, 2),
            "osnr_db": round(telemetry.osnr_db, 2),
            "measured_osnr_db": round(telemetry.measured_osnr_db, 2),
            "q_effective": round(telemetry.q_effective, 2),
            "q_db": round(20.0 * math.log10(max(0.01, telemetry.q_effective)), 1) if telemetry.q_effective > 0.1 else None,
            "ber": float(telemetry.ber),
            "eye_opening_pct": round(telemetry.eye_opening * 100.0, 1),
            "pulse_spread_ps": round(telemetry.dispersion_ratio * telemetry.bit_period_ps, 1),
            "bit_period_ps": round(telemetry.bit_period_ps, 1),
            "margin_db": round(p_margin, 2),
            "attenuation_alpha": round(telemetry.attenuation_alpha, 3),
            "is_cut": telemetry.is_cut
        },
        "otdr": otdr,
        "otdr_analysis": analysis,
        "features": [round(f, 4) for f in features],
        "feature_names": FEATURE_NAMES,
        "power_budget": budget_items
    }


@app.post("/api/predict")
def predict_fault(req: PredictionRequest):
    """Executes ML fault diagnosis and localization."""
    features = req.features
    loc_km = None
    ground_truth_km = None
    ground_truth_type = None

    if features is None and req.simulation_params is not None:
        p = req.simulation_params
        sim_res = simulate_link(p)
        features = sim_res["features"]
        ground_truth_km = sim_res["params"]["fault_km"]
        ground_truth_type = sim_res["params"]["fault_type"]
        an = sim_res["otdr_analysis"]
        if p.fault_type == 1:
            loc_km = an["end_distance_km"]
        elif p.fault_type in [2, 3]:
            loc_km = an["event_pos_km"]

    if features is None:
        raise HTTPException(status_code=400, detail="Either 'features' or 'simulation_params' must be provided.")

    model_name = req.model_name or ml_engine.active_model_name
    result = ml_engine.predict(features, model_name=model_name)

    # Attach localization if applicable
    pred_class = result["predicted_class"]
    result["localized_km"] = loc_km
    result["ground_truth_km"] = ground_truth_km
    result["ground_truth_type"] = ground_truth_type

    # Distance localization error
    if loc_km is not None and ground_truth_km is not None and pred_class == ground_truth_type:
        result["localization_error_km"] = round(abs(loc_km - ground_truth_km), 2)
    else:
        result["localization_error_km"] = None

    # Compare with legacy threshold alarm
    dp = features[0] * 4.0
    result["legacy_threshold"] = {
        "alarm_triggered": bool(abs(dp) > 3.0),
        "delta_prx_db": round(dp, 2),
        "verdict": "ALARM: Power deviated > 3 dB" if abs(dp) > 3.0 else "SILENT: Power within 3 dB tolerance",
        "limitation": "Legacy alarms cannot determine the physical cause, fail to localize the fault along the fiber, and miss dispersion, EDFA ASE noise, and mild macro-bends."
    }

    return result


@app.get("/api/models")
def get_all_models():
    """Returns evaluation benchmarks across all trained machine learning architectures."""
    return {
        "active_model": ml_engine.active_model_name,
        "available_models": list(ml_engine.models.keys()),
        "metrics": ml_engine.metrics,
        "fault_names": FAULT_NAMES,
        "fault_short": FAULT_SHORT,
        "fault_colors": FAULT_COLORS
    }


@app.post("/api/models/select")
def select_model(req: SelectModelRequest):
    """Switches the active inference model."""
    if req.model_name not in ml_engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found. Available: {list(ml_engine.models.keys())}")
    ml_engine.active_model_name = req.model_name
    return {
        "status": "success",
        "active_model": ml_engine.active_model_name,
        "metrics": ml_engine.metrics.get(req.model_name)
    }


@app.post("/api/models/train")
def train_model(req: TrainModelRequest):
    """Retrains a specific model architecture with custom sample size."""
    if req.model_name not in ['Random Forest', 'Gradient Boosting', 'k-Nearest Neighbors', 'Multi-Layer Perceptron']:
        raise HTTPException(status_code=400, detail="Unsupported model name for training.")
    new_metrics = ml_engine.retrain_model(req.model_name, req.n_train)
    return {
        "status": "success",
        "model_name": req.model_name,
        "metrics": new_metrics
    }


@app.get("/api/dataset/sample")
def get_dataset_sample():
    """Returns representative telemetry records for the dataset inspector."""
    return {
        "count": len(ml_engine.cached_sample_records),
        "records": ml_engine.cached_sample_records,
        "features": FEATURE_NAMES
    }


@app.get("/api/dataset/export.csv")
def export_dataset_csv():
    """Generates and downloads a CSV of the synthetic optical link measurements."""
    output = io.StringIO()
    writer = csv.writer(output)

    header = [
        "Sample_ID", "Fault_Type_Code", "Fault_Name", "Fiber_Length_km", 
        "Launch_Power_dBm", "Bit_Rate_Gbps", "Severity_Percent", 
        "Rx_Power_dBm", "OSNR_dB", "Attenuation_Alpha_dB_km", 
        "OTDR_Step_Loss_dB", "Fresnel_Reflection_Spike_dB",
        "Feature_RxPowerDev", "Feature_OSNRDev", "Feature_AlphaDev",
        "Feature_StepLoss", "Feature_SpikeHeight", "Feature_CutDev", "Feature_DispRatio"
    ]
    writer.writerow(header)

    for rec in ml_engine.cached_sample_records:
        feats = rec.get("features", [0]*7)
        row = [
            rec["id"], rec["fault_type"], rec["fault_name"], rec["length_km"],
            rec["launch_power_dbm"], rec["bit_rate_gbps"], rec["severity_pct"],
            rec["prx_dbm"], rec["osnr_db"], rec["alpha_db_km"],
            rec["step_loss_db"], rec["fresnel_spike_db"]
        ] + feats
        writer.writerow(row)

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fiberguard_telemetry_dataset.csv"}
    )


@app.post("/api/diagnose-custom")
def diagnose_custom_telemetry(req: CustomTelemetryRequest):
    """
    Diagnostic sandbox for field optical engineers: converts manual instrument readings into AI fault diagnosis.
    """
    L = req.fiber_length_km
    # Expected design calculations for a nominal 10 Gb/s link
    num_splices = max(0, int(L // 20))
    pin_exp = 0.0 - FRONT_CONN - (ALPHA0 * L) - (num_splices * SPLICE_LOSS) - END_CONN
    prx_exp = pin_exp + G0
    osnr_exp = 58.0 + pin_exp - NF0
    t_ps = 100.0  # 10 Gbps
    ratio_exp = (D_DISP * L * DLAM0) / t_ps
    measured_ratio = req.measured_pulse_spread_ps / t_ps

    # Build normalized feature vector
    p_dev = np.clip(req.measured_prx_dbm - prx_exp, -45.0, 10.0) / 4.0
    osnr_dev = np.clip(req.measured_osnr_db - osnr_exp, -45.0, 10.0) / 5.0
    alpha_dev = (req.measured_alpha_db_km - ALPHA0) / 0.05
    step_loss = req.step_loss_db / 1.0
    spike_height = req.reflection_spike_db / 3.0
    end_ratio = min(1.2, req.detected_fiber_end_km / L)
    cut_dev = np.clip((1.0 - end_ratio) / 0.05, 0.0, 20.0)
    disp_log = math.log10(max(1e-3, measured_ratio) / max(1e-3, ratio_exp)) / 0.3

    features = [
        float(p_dev), float(osnr_dev), float(alpha_dev),
        float(step_loss), float(spike_height), float(cut_dev), float(disp_log)
    ]

    model_name = req.model_name or ml_engine.active_model_name
    pred = ml_engine.predict(features, model_name=model_name)

    # Legacy comparison
    dp = req.measured_prx_dbm - prx_exp
    alarm = abs(dp) > 3.0

    return {
        "inputs": req.dict(),
        "extracted_features": [round(f, 3) for f in features],
        "feature_names": FEATURE_NAMES,
        "prediction": pred,
        "legacy_alarm": {
            "triggered": alarm,
            "delta_prx_db": round(dp, 2),
            "message": "ALARM: Exceeds 3 dB tolerance" if alarm else "SILENT: Normal power window"
        }
    }


# Mount static directory for frontend
import os
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
