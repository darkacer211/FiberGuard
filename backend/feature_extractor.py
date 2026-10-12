"""
FiberGuard Backend - OTDR Digital Signal Processing & Feature Extraction
Implements industry-standard reflectometry processing algorithms:
linear regression slope estimation, moving-window step detection,
peak searching for reflective anomalies, and feature normalization for ML.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple
from backend.physics_engine import ALPHA0, NOISE_FLOOR, OpticalLinkTelemetry


def analyze_otdr_trace(trace: List[float], dx: float, fiber_length: float) -> Dict[str, Any]:
    """
    Analyzes raw OTDR backscatter trace data to extract physical fiber parameters:
    attenuation slope (alpha), localized step loss, Fresnel peak height, and cut location.
    """
    n = len(trace)
    tr = np.array(trace, dtype=np.float32)

    # 5-point moving average smoothing
    smoothed = np.convolve(tr, np.ones(5) / 5.0, mode='same')

    # Detect trace drop-off / fiber end
    end_idx = n - 1
    while end_idx > 5 and smoothed[end_idx] < (NOISE_FLOOR + 4.5):
        end_idx -= 1

    end_distance = max(0.0, float((end_idx - 2) * dx))
    end_ratio = min(1.2, end_distance / max(1.0, fiber_length))

    # Fiber Attenuation Slope (alpha) estimation via partitioned linear regression
    # Partition into segments to avoid single-event skewing, take median slope
    start_idx = 4
    last_idx = max(start_idx + 15, end_idx - 6)
    k_segments = 6
    seg_len = max(4, (last_idx - start_idx) // k_segments)
    slopes = []

    for k in range(k_segments):
        s_i = start_idx + k * seg_len
        e_i = min(s_i + seg_len, end_idx)
        if e_i - s_i > 4:
            x_vals = np.arange(s_i, e_i) * dx
            y_vals = tr[s_i:e_i]
            # Linear regression: y = m*x + c
            coeffs = np.polyfit(x_vals, y_vals, 1)
            slope = -coeffs[0]  # Attenuation in dB/km (positive)
            if 0.05 <= slope <= 1.5:
                slopes.append(slope)

    alpha_measured = float(np.median(slopes)) if len(slopes) > 0 else ALPHA0

    # Localized step loss & Fresnel reflection peak detection
    # Dual-window moving average differential across a gap
    w = 8
    gap = 3
    lo = start_idx + w + gap + 1
    hi = max(lo + 5, end_idx - 16)

    best_step = 0.0
    best_idx = -1
    step_array = np.full(n, -10.0, dtype=np.float32)

    if hi > lo:
        for i in range(lo, hi):
            left_mean = float(np.mean(tr[i - gap - w: i - gap]))
            right_mean = float(np.mean(tr[i + gap + 1: i + gap + 1 + w]))
            # Subtract baseline attenuation across the window span
            expected_drop = alpha_measured * (2 * gap + 1 + w) * dx
            step_val = (left_mean - right_mean) - expected_drop
            step_array[i] = step_val
            if step_val > best_step:
                best_step = step_val
                best_idx = i

    event_pos = 0.0
    detected_step = 0.0
    reflection_spike = 0.0

    if best_idx > 0 and best_step > 0.4:
        # Refine event center
        l_bound = best_idx
        r_bound = best_idx
        while l_bound - 1 >= lo and step_array[l_bound - 1] >= 0.85 * best_step:
            l_bound -= 1
        while r_bound + 1 < hi and step_array[r_bound + 1] >= 0.85 * best_step:
            r_bound += 1

        center_idx = (l_bound + r_bound) // 2
        event_pos = float(center_idx * dx)
        detected_step = float(best_step)

        # Search for Fresnel reflection peak at center
        left_baseline = float(np.mean(tr[center_idx - gap - w: center_idx - gap]))
        peak_val = float(np.max(tr[center_idx - gap: center_idx + gap + 1]))
        reflection_spike = max(0.0, peak_val - left_baseline)

    return {
        'alpha_measured': round(alpha_measured, 3),
        'detected_step_db': round(detected_step, 2),
        'event_pos_km': round(event_pos, 2),
        'reflection_spike_db': round(reflection_spike, 2),
        'end_distance_km': round(end_distance, 2),
        'end_ratio': round(end_ratio, 3)
    }


def extract_features(telemetry: OpticalLinkTelemetry, otdr_analysis: Dict[str, Any]) -> List[float]:
    """
    Extracts normalized 7-dimensional feature vector for machine learning:
    1. rx_power_dev: normalized deviation from expected Rx power
    2. osnr_dev: normalized deviation from expected OSNR
    3. alpha_dev: deviation from nominal fiber attenuation (0.20 dB/km)
    4. step_loss: localized non-reflective step drop (dB)
    5. spike_height: localized Fresnel reflection peak (dB)
    6. cut_ratio_dev: shortfall in fiber length ratio (flags cable breaks)
    7. dispersion_log_ratio: log ratio of pulse spread compared to design
    """
    # 1. Rx power deviation
    p_dev = np.clip(telemetry.measured_prx_dbm - telemetry.prx_expected_dbm, -45.0, 10.0) / 4.0

    # 2. OSNR deviation
    osnr_dev = np.clip(telemetry.measured_osnr_db - telemetry.osnr_expected_db, -45.0, 10.0) / 5.0

    # 3. Attenuation slope deviation
    alpha_dev = (otdr_analysis['alpha_measured'] - ALPHA0) / 0.05

    # 4. Step loss
    step_loss = otdr_analysis['detected_step_db'] / 1.0

    # 5. Fresnel spike
    spike_height = otdr_analysis['reflection_spike_db'] / 3.0

    # 6. End ratio shortfall
    cut_dev = np.clip((1.0 - otdr_analysis['end_ratio']) / 0.05, 0.0, 20.0)

    # 7. Dispersion pulse broadening ratio
    disp_ratio = math.log10(max(1e-3, telemetry.measured_dispersion_ratio) / max(1e-3, telemetry.dispersion_ratio_expected)) / 0.3

    return [
        float(p_dev),
        float(osnr_dev),
        float(alpha_dev),
        float(step_loss),
        float(spike_height),
        float(cut_dev),
        float(disp_ratio)
    ]


FEATURE_NAMES = [
    'Rx Power Deviation (dB)',
    'OSNR Deviation (dB)',
    'Alpha Slope Deviation (dB/km)',
    'OTDR Step Loss (dB)',
    'Fresnel Spike Height (dB)',
    'Fiber End Shortfall Ratio',
    'Dispersion Pulse Broadening Ratio'
]
