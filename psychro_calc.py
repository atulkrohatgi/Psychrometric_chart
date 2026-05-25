import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)
PRESSURE = 101325.0  # Pa, standard sea level


def _make_state(dbt, w):
    """Build complete state dict from DBT + W using individual Get* calls."""
    w   = max(w, 1e-7)   # psychrolib rejects W = 0
    wbt = psychrolib.GetTWetBulbFromHumRatio(dbt, w, PRESSURE)
    dpt = psychrolib.GetTDewPointFromHumRatio(dbt, w, PRESSURE)
    rh  = psychrolib.GetRelHumFromHumRatio(dbt, w, PRESSURE)
    vp  = psychrolib.GetVapPresFromHumRatio(w, PRESSURE)
    h   = psychrolib.GetMoistAirEnthalpy(dbt, w)          # J/kg dry air
    v   = psychrolib.GetMoistAirVolume(dbt, w, PRESSURE)  # m³/kg dry air
    mu  = psychrolib.GetDegreeOfSaturation(dbt, w, PRESSURE)
    return {
        "DBT": round(dbt, 2),
        "WBT": round(wbt, 2),
        "DPT": round(dpt, 2),
        "RH":  round(rh * 100, 1),
        "W":   round(w, 6),
        "h":   round(h / 1000, 3),   # J/kg → kJ/kg
        "v":   round(v, 4),
        "Pv":  round(vp / 1000, 4),  # Pa  → kPa
        "mu":  round(mu, 4),
    }


def from_dbt_wbt(dbt, wbt):
    w = psychrolib.GetHumRatioFromTWetBulb(dbt, wbt, PRESSURE)
    return _make_state(dbt, w)


def from_dbt_rh(dbt, rh_pct):
    w = psychrolib.GetHumRatioFromRelHum(dbt, rh_pct / 100.0, PRESSURE)
    return _make_state(dbt, w)


def from_dbt_dpt(dbt, dpt):
    w = psychrolib.GetHumRatioFromTDewPoint(dpt, PRESSURE)
    return _make_state(dbt, w)


def from_dbt_w(dbt, w):
    return _make_state(dbt, w)


def from_dbt_v(dbt, v):
    # v = Ra * T * (1 + 1.6078 * W) / P  →  solve for W
    Ra = 287.042
    T  = dbt + 273.15
    w  = max(1e-7, ((v * PRESSURE) / (Ra * T) - 1.0) / 1.6078)
    return _make_state(dbt, w)


def from_dbt_h(dbt, h_kj):
    """State from dry-bulb temperature (°C) and specific enthalpy (kJ/kg dry air)."""
    denom = 2501.0 + 1.86 * dbt
    if denom <= 0:
        raise ValueError(f"Cannot derive W from h={h_kj} kJ/kg at DBT={dbt} °C")
    w = (h_kj - 1.006 * dbt) / denom
    return _make_state(dbt, max(w, 1e-7))
