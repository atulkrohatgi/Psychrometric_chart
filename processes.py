from psychro_calc import from_dbt_w, from_dbt_rh, from_dbt_wbt


# ── 1 & 2  Sensible Heating / Cooling ────────────────────────────────────────

def sensible_heating(s1, dbt2):
    s2 = from_dbt_w(dbt2, s1["W"])
    return s2, {
        "Heat Added (kJ/kg dry air)": round(s2["h"] - s1["h"], 3),
        "ΔT (°C)": round(dbt2 - s1["DBT"], 2),
        "W constant (kg/kg)": s1["W"],
    }


def sensible_cooling(s1, dbt2):
    s2 = from_dbt_w(dbt2, s1["W"])
    return s2, {
        "Heat Removed (kJ/kg dry air)": round(s1["h"] - s2["h"], 3),
        "ΔT (°C)": round(s1["DBT"] - dbt2, 2),
        "W constant (kg/kg)": s1["W"],
    }


# ── 3 & 4  Humidification / Dehumidification (isothermal, DBT constant) ──────

def humidification(s1, w2=None, rh2=None):
    s2 = from_dbt_w(s1["DBT"], w2) if w2 is not None else from_dbt_rh(s1["DBT"], rh2)
    return s2, {
        "Moisture Added (kg/kg dry air)": round(s2["W"] - s1["W"], 6),
        "Enthalpy Change (kJ/kg)": round(s2["h"] - s1["h"], 3),
        "DBT constant (°C)": s1["DBT"],
    }


def dehumidification(s1, w2=None, rh2=None):
    s2 = from_dbt_w(s1["DBT"], w2) if w2 is not None else from_dbt_rh(s1["DBT"], rh2)
    return s2, {
        "Moisture Removed (kg/kg dry air)": round(s1["W"] - s2["W"], 6),
        "Enthalpy Change (kJ/kg)": round(s1["h"] - s2["h"], 3),
        "DBT constant (°C)": s1["DBT"],
    }


# ── 5  Cooling & Dehumidification ─────────────────────────────────────────────

def cooling_dehumidification(s1, dbt2, w2=None, rh2=None):
    s2 = from_dbt_w(dbt2, w2) if w2 is not None else from_dbt_rh(dbt2, rh2)
    total_heat   = round(s1["h"] - s2["h"], 3)
    sensible     = round(1.006 * (s1["DBT"] - s2["DBT"]), 3)
    delta_w      = round(s1["W"] - s2["W"], 6)
    latent       = round(2501 * delta_w, 3)
    return s2, {
        "Total Heat Removed (kJ/kg)": total_heat,
        "Sensible Heat Removed (kJ/kg)": sensible,
        "Latent Heat Removed (kJ/kg)": latent,
        "Moisture Removed (kg/kg)": delta_w,
    }


# ── 6  Heating & Humidification ───────────────────────────────────────────────

def heating_humidification(s1, dbt2, w2=None, rh2=None):
    s2 = from_dbt_w(dbt2, w2) if w2 is not None else from_dbt_rh(dbt2, rh2)
    return s2, {
        "Total Heat Added (kJ/kg)": round(s2["h"] - s1["h"], 3),
        "Moisture Added (kg/kg)": round(s2["W"] - s1["W"], 6),
        "ΔT (°C)": round(s2["DBT"] - s1["DBT"], 2),
    }


# ── 7  Evaporative / Adiabatic Cooling ───────────────────────────────────────

def evaporative_cooling(s1, dbt2):
    # WBT stays constant; process moves along constant-WBT line
    s2 = from_dbt_wbt(dbt2, s1["WBT"])
    return s2, {
        "Temperature Drop (°C)": round(s1["DBT"] - s2["DBT"], 2),
        "Moisture Added (kg/kg)": round(s2["W"] - s1["W"], 6),
        "WBT constant (°C)": s1["WBT"],
        "Enthalpy Change (kJ/kg)": round(s2["h"] - s1["h"], 3),
    }


# ── 8  Adiabatic Mixing ───────────────────────────────────────────────────────

def adiabatic_mixing(s1, s2, m1, m2):
    """
    m1, m2 – mass flow rates of dry air (any consistent unit).
    Mixed state lies on the straight line between s1 and s2 on the chart,
    dividing it in the ratio m2 : m1 from s1.
    """
    total = m1 + m2
    w_mix = (m1 * s1["W"] + m2 * s2["W"]) / total
    h_mix = (m1 * s1["h"] + m2 * s2["h"]) / total   # kJ/kg

    # DBT from enthalpy and humidity ratio (approximate ASHRAE formula)
    # h = 1.006*T + W*(2501 + 1.86*T)  →  T = (h - 2501*W) / (1.006 + 1.86*W)
    dbt_mix = (h_mix - 2501.0 * w_mix) / (1.006 + 1.86 * w_mix)

    s_mix = from_dbt_w(round(dbt_mix, 3), round(w_mix, 6))
    return s_mix, {
        "Mixed DBT (°C)": round(dbt_mix, 2),
        "Mixed W (kg/kg)": round(w_mix, 6),
        "Mixed h (kJ/kg)": round(h_mix, 3),
        "Mass flow ratio m₁ : m₂": f"{m1} : {m2}",
    }
