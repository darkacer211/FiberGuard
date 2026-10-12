"""
FiberGuard Backend - Physical Optics Simulation Engine
Models 1550 nm single-mode fiber (SMF-28) propagation, attenuation, 
dispersion, EDFA amplification, noise, OTDR traces, and eye diagrams.
"""

import math
import numpy as np
from scipy.special import erfc, erf
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# Physical & Telecom Constants for standard SMF-28 at 1550 nm (C-band)
D_DISP = 17.0        # Chromatic dispersion parameter [ps / (nm * km)]
DLAM0 = 0.02         # Nominal DFB laser linewidth [nm]
ALPHA0 = 0.20        # Nominal SMF attenuation [dB / km]
FRONT_CONN = 0.5     # Transmitter patch panel insertion loss [dB]
END_CONN = 0.5       # Receiver patch panel insertion loss [dB]
SPLICE_LOSS = 0.1    # Standard fusion splice insertion loss [dB]
G0 = 15.0            # Nominal EDFA amplifier gain [dB]
NF0 = 5.0            # Nominal EDFA noise figure [dB]
NOISE_FLOOR = -50.0  # OTDR Rayleigh backscatter noise floor [dB]

FAULT_NAMES = [
    'Normal Link',
    'Fiber Cut',
    'Macro-Bend',
    'Dirty Connector',
    'Aged Fiber',
    'EDFA Failure',
    'Chromatic Dispersion'
]

FAULT_SHORT = ['Normal', 'Cut', 'Bend', 'Dirty', 'Aged', 'EDFA', 'Disp']

FAULT_COLORS = [
    '#3ddc97',  # Normal: Emerald
    '#ff6b6b',  # Fiber Cut: Coral Red
    '#ffc857',  # Macro-Bend: Amber Yellow
    '#ff9a5c',  # Dirty Connector: Orange
    '#b69cff',  # Aged Fiber: Violet Purple
    '#5ee3ff',  # EDFA Failure: Cyan Blue
    '#ff7ccf'   # Dispersion: Magenta Pink
]

FAULT_DESCRIPTIONS = [
    {
        'title': 'Normal Operation',
        'signature': 'All monitored optical metrics reside well within design budget tolerances. Minimal 0.1 dB splice steps are buried within OTDR Rayleigh backscatter noise.',
        'cause': 'Fiber link is intact, connectors clean, EDFA operating nominally.',
        'action': 'Maintain routine telemetry logging and archive as baseline for predictive degradation analysis.',
        'severity_label': 'Operating Margin'
    },
    {
        'title': 'Physical Fiber Cut / Cable Sever',
        'signature': 'Catastrophic optical loss: prominent Fresnel reflection spike at break distance followed by backscatter plunge to the noise floor (-50 dB). Receiver optical power drops to < -60 dBm (Loss of Signal / LOS alarm).',
        'cause': 'Third-party construction excavation (backhoe dig-in), tensile shear, conduit collapse, or aerial cable rupture.',
        'action': 'Dispatch emergency fiber restoration crew to localized kilometer mark. Immediately switch traffic to redundant optical ring protection path.',
        'severity_label': 'Sever Completeness'
    },
    {
        'title': 'Localized Macro-Bending Loss',
        'signature': 'Sharp step loss in OTDR trace without any Fresnel reflection spike. Backscatter slope on both sides of the step remains parallel at 0.20 dB/km. High bend sensitivity at 1550 nm/1625 nm.',
        'cause': 'Fiber bent tighter than critical radius (< 30 mm for G.652.D) inside a splice tray, fiber distribution panel, or crushed duct.',
        'action': 'Inspect technician splice enclosure at identified distance. Release physical pinch or replace with ITU-T G.657 bend-insensitive fiber.',
        'severity_label': 'Bend Tightness'
    },
    {
        'title': 'Contaminated / Dirty Connector End-Face',
        'signature': 'Combined step insertion loss coupled with a significant Fresnel reflection peak. Caused by refractive index mismatch across microscopic air gap or oil/dust residue.',
        'cause': 'Particulate dust, finger oil, scratched ferrule, or improperly mated LC/SC optical connector.',
        'action': 'Inspect connector ferrule with fiber inspection scope. Clean using lint-free optic wipes and high-purity isopropyl alcohol, then re-seat.',
        'severity_label': 'Contamination Level'
    },
    {
        'title': 'Distributed Fiber Aging / Water Ingress',
        'signature': 'Uniformly elevated attenuation slope (> 0.25 - 0.40 dB/km) extending across the entire fiber span. No localized step events, but total span insertion loss severely exceeds design budget.',
        'cause': 'Hydroxyl (OH-) ion diffusion from underground water ingress, hydrogen darkening, radiation damage, or micro-bending along aged cable sheath.',
        'action': 'Perform OTDR dual-wavelength sweep (1310/1550 nm) to confirm OH- absorption peak. Plan cable segment overhaul or re-budget EDFA amplification.',
        'severity_label': 'Cable Aging Index'
    },
    {
        'title': 'EDFA Optical Amplifier Pump Degradation',
        'signature': 'OTDR trace appears completely pristine because backscatter only measures passive glass. However, receiver power plunges and OSNR degrades precipitously due to high ASE noise.',
        'cause': '980 nm / 1480 nm semiconductor pump laser diode degradation, thermoelectric cooler (TEC) failure, or loss of population inversion in erbium fiber.',
        'action': 'Check amplifier pump bias current, optical gain setpoint, and temperature telemetry. Replace degraded EDFA optical module.',
        'severity_label': 'Pump Degradation'
    },
    {
        'title': 'Excessive Chromatic Dispersion / Pulse Broadening',
        'signature': 'Total optical power and OTDR backscatter slope are 100% nominal. However, pulse broadening ratio Δτ / T_bit exceeds 1.0, closing the receiver eye diagram due to Intersymbol Interference (ISI).',
        'cause': 'Uncompensated chromatic dispersion over long SMF-28 span, laser chirp, wide DFB linewidth, or faulty electronic dispersion compensation (EDC).',
        'action': 'Install Dispersion Compensating Fiber (DCF) module, Fiber Bragg Grating (FBG), or upgrade transceiver to coherent DSP with electronic dispersion compensation.',
        'severity_label': 'Source Linewidth / Chirp'
    }
]


def gaussian_pulse(u: float, sigma: float) -> float:
    """Calculates smoothed rectangular pulse convolved with Gaussian dispersion filter."""
    if sigma <= 0.001:
        return 1.0 if abs(u) <= 0.5 else 0.0
    denom = math.sqrt(2.0) * sigma
    return float(0.5 * (erf((u + 0.5) / denom) - erf((u - 0.5) / denom)))


def eye_opening_ratio(sigma: float) -> float:
    """Calculates vertical eye opening factor considering worst-case intersymbol interference."""
    center = gaussian_pulse(0.0, sigma)
    isi = 0.0
    for k in range(-3, 4):
        if k != 0:
            isi += gaussian_pulse(float(k), sigma)
    return max(0.0, center - isi)


@dataclass
class OpticalLinkParams:
    length_km: float = 80.0       # Span length [km]
    launch_power_dbm: float = 0.0 # Transmitter power [dBm]
    bit_rate_gbps: float = 10.0   # Bit rate [Gb/s]
    fault_type: int = 0           # 0: Normal, 1: Cut, 2: Bend, 3: Dirty, 4: Aged, 5: EDFA, 6: Disp
    fault_pos_ratio: float = 0.45 # Location along fiber (0.0 to 1.0)
    severity: float = 0.5         # Fault severity (0.0 to 1.0)


@dataclass
class OpticalLinkTelemetry:
    params: OpticalLinkParams
    fiber_length: float
    fault_km: float
    bit_period_ps: float
    attenuation_alpha: float
    extra_loss_db: float
    reflection_spike_db: float
    edfa_gain_db: float
    noise_figure_db: float
    dispersion_ratio: float
    is_cut: bool
    num_splices: int
    fiber_loss_db: float
    splice_loss_db: float
    pin_edfa_dbm: float
    prx_dbm: float
    osnr_db: float
    pin_expected_dbm: float
    prx_expected_dbm: float
    osnr_expected_db: float
    dispersion_ratio_expected: float
    receiver_sensitivity_dbm: float
    sigma_dispersion: float
    eye_opening: float
    q_noise_free: float
    q_effective: float
    ber: float
    events: List[Dict[str, float]]
    measured_prx_dbm: float
    measured_osnr_db: float
    measured_dispersion_ratio: float


def simulate_optical_physics(params: OpticalLinkParams, rng: np.random.Generator = None) -> OpticalLinkTelemetry:
    """
    Simulates complete physics of a single-mode optical fiber transmission link.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = params.length_km
    f = params.fault_type
    sev = params.severity
    rate = params.bit_rate_gbps
    T_ps = 1000.0 / rate  # Bit period in picoseconds
    fault_km = params.fault_pos_ratio * L
    num_splices = max(0, int(L // 20))
    splice_loss_total = num_splices * SPLICE_LOSS

    alpha = ALPHA0
    extra_loss = 0.0
    reflection_spike = 0.0
    edfa_gain = G0
    edfa_nf = NF0
    pulse_ratio = (D_DISP * L * DLAM0) / T_ps
    is_cut = (f == 1)

    # Injected fault physics
    if f == 4:  # Aged fiber / high attenuation
        alpha = 0.25 + 0.15 * sev
    elif f == 2:  # Macro-bend
        extra_loss = 0.8 + 3.8 * sev
    elif f == 3:  # Dirty connector
        extra_loss = 0.7 + 2.5 * sev
        reflection_spike = 3.0 + 6.5 * sev
    elif f == 5:  # EDFA pump failure
        edfa_gain = G0 - (10.0 + 13.0 * sev)
        edfa_nf = 10.0 + 7.0 * sev
    elif f == 6:  # Excessive chromatic dispersion
        pulse_ratio = 1.0 + 1.5 * sev

    fiber_loss = alpha * L

    if is_cut:
        pin_edfa = -65.0
        prx = -65.0
        osnr = -5.0
    else:
        pin_edfa = params.launch_power_dbm - FRONT_CONN - fiber_loss - splice_loss_total - extra_loss - END_CONN
        prx = pin_edfa + edfa_gain
        osnr = 58.0 + pin_edfa - edfa_nf

    # Expected values for nominal link of identical design
    pin_exp = params.launch_power_dbm - FRONT_CONN - (ALPHA0 * L) - splice_loss_total - END_CONN
    prx_exp = pin_exp + G0
    osnr_exp = 58.0 + pin_exp - NF0
    ratio_exp = (D_DISP * L * DLAM0) / T_ps

    # Receiver physics
    rx_sens = -18.0 if rate > 5.0 else -24.0
    sigma_disp = math.sqrt(0.12**2 + (pulse_ratio / 2.355)**2)
    eye_open = eye_opening_ratio(sigma_disp)

    # Electrical receiver bandwidth and noise calculations
    be = 0.75 * rate
    q_opt = math.sqrt(max(0.001, (10.0**(osnr / 10.0)) * 12.5 / (2.0 * be)))
    q_th = 6.0 * (10.0**((prx - rx_sens) / 10.0))
    q0 = 1.0 / math.sqrt(max(1e-6, 1.0 / (q_opt**2 + 1e-6) + 1.0 / (q_th**2 + 1e-6)))
    q_eff = q0 * max(eye_open, 0.001)

    ber = float(0.5 * erfc(q_eff / math.sqrt(2.0)))
    if is_cut or prx < -50.0:
        ber = 0.5

    # Discrete events along fiber
    events = [{'x': 0.0, 'loss': 0.0, 'reflection': 10.0, 'type': 'Tx Launch'}]
    for k in range(1, num_splices + 1):
        events.append({
            'x': round(k * L / (num_splices + 1), 2),
            'loss': SPLICE_LOSS,
            'reflection': 0.0,
            'type': f'Fusion Splice #{k}'
        })

    if f == 2:
        events.append({'x': round(fault_km, 2), 'loss': extra_loss, 'reflection': 0.0, 'type': 'Macro-Bend'})
    elif f == 3:
        events.append({'x': round(fault_km, 2), 'loss': extra_loss, 'reflection': reflection_spike, 'type': 'Dirty Connector'})

    if is_cut:
        events.append({'x': round(fault_km, 2), 'loss': 50.0, 'reflection': 9.5, 'type': 'Fiber Cut'})
    else:
        events.append({'x': round(L, 2), 'loss': 0.0, 'reflection': 6.0, 'type': 'Rx End Connector'})

    # Noisy instrument telemetry readings
    measured_prx = prx + float(rng.normal(0, 0.28))
    measured_osnr = osnr + float(rng.normal(0, 0.45))
    measured_ratio = max(0.01, pulse_ratio * (1.0 + float(rng.normal(0, 0.07))))

    return OpticalLinkTelemetry(
        params=params,
        fiber_length=L,
        fault_km=fault_km,
        bit_period_ps=T_ps,
        attenuation_alpha=alpha,
        extra_loss_db=extra_loss,
        reflection_spike_db=reflection_spike,
        edfa_gain_db=edfa_gain,
        noise_figure_db=edfa_nf,
        dispersion_ratio=pulse_ratio,
        is_cut=is_cut,
        num_splices=num_splices,
        fiber_loss_db=fiber_loss,
        splice_loss_db=splice_loss_total,
        pin_edfa_dbm=pin_edfa,
        prx_dbm=prx,
        osnr_db=osnr,
        pin_expected_dbm=pin_exp,
        prx_expected_dbm=prx_exp,
        osnr_expected_db=osnr_exp,
        dispersion_ratio_expected=ratio_exp,
        receiver_sensitivity_dbm=rx_sens,
        sigma_dispersion=sigma_disp,
        eye_opening=eye_open,
        q_noise_free=q0,
        q_effective=q_eff,
        ber=ber,
        events=events,
        measured_prx_dbm=measured_prx,
        measured_osnr_db=measured_osnr,
        measured_dispersion_ratio=measured_ratio
    )


def generate_otdr_trace(telemetry: OpticalLinkTelemetry, rng: np.random.Generator = None, num_points: int = 512) -> Dict[str, Any]:
    """
    Synthesizes a realistic OTDR Rayleigh backscatter curve with noise, splice drops, and Fresnel peaks.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = telemetry.fiber_length
    l_max = L * 1.15
    dx = l_max / (num_points - 1)
    end_km = telemetry.fault_km if telemetry.is_cut else L
    trace = np.zeros(num_points, dtype=np.float32)

    for i in range(num_points):
        x = i * dx
        if x > end_km + dx * 0.5:
            # Past break point: noise floor
            val = NOISE_FLOOR + float(rng.normal(0, 0.75))
        else:
            # Rayleigh backscatter slope
            level = -telemetry.attenuation_alpha * x
            for ev in telemetry.events:
                if ev['loss'] > 0 and ev['x'] < x:
                    level -= ev['loss']
            level = max(level, NOISE_FLOOR)
            val = level + float(rng.normal(0, 0.03 + 0.0035 * abs(level)))

        # Add Fresnel reflection Gaussian spikes
        for ev in telemetry.events:
            if ev['reflection'] > 0:
                dist_norm = (x - ev['x']) / (1.15 * dx)
                val += ev['reflection'] * math.exp(-dist_norm * dist_norm)

        trace[i] = val

    distances = [round(i * dx, 3) for i in range(num_points)]
    return {
        'trace': [round(float(v), 2) for v in trace],
        'distances': distances,
        'dx': dx,
        'l_max': l_max,
        'end_km': end_km
    }
