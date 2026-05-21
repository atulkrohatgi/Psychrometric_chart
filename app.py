import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from psychro_calc import from_dbt_dpt, from_dbt_rh, from_dbt_v, from_dbt_w, from_dbt_wbt
from processes import (
    adiabatic_mixing,
    cooling_dehumidification,
    dehumidification,
    evaporative_cooling,
    heating_humidification,
    humidification,
    sensible_cooling,
    sensible_heating,
)
from psychro_chart import draw_psychro_chart

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Psychrometric Calculator",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🌡️ Psychrometric Calculator")
st.caption("SI Units  |  Standard Atmospheric Pressure: 101.325 kPa  |  ASHRAE formulas")

# ── session state initialisation ─────────────────────────────────────────────
# Results are stored here so that clicking Download does not clear the page.
for _k in ("sp_result", "sp_png", "proc_result", "proc_png", "ahu_png"):
    if _k not in st.session_state:
        st.session_state[_k] = None
if "ahu_chain" not in st.session_state:
    st.session_state["ahu_chain"] = []   # must be a list, not None


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_float(raw, label):
    try:
        return float(raw.strip()), None
    except (ValueError, AttributeError):
        return None, f"**{label}**: '{raw}' is not a valid number."


def validated_input(label, default, key, unit="", placeholder=""):
    hint = f"e.g. {default}"
    raw = st.text_input(f"{label}  {unit}", value=str(default), key=key,
                        placeholder=hint if placeholder == "" else placeholder)
    return _parse_float(raw, label)


def _fig_to_png(fig, dpi=180):
    """Save a matplotlib figure to PNG bytes and return them."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _download_button(png_bytes, filename):
    st.download_button(
        label="⬇️  Download Chart (PNG)",
        data=png_bytes,
        file_name=filename,
        mime="image/png",
    )


# Bounds: (min, max, friendly_name, advice)
BOUNDS = {
    "DBT": (-10.0,  50.0, "Dry Bulb Temperature",
            "Typical HVAC range: −10 °C to 50 °C."),
    "WBT": (-10.0,  50.0, "Wet Bulb Temperature",
            "WBT must be ≤ DBT (wet bulb is always ≤ dry bulb)."),
    "DPT": (-30.0,  50.0, "Dew Point Temperature",
            "DPT must be ≤ DBT (dew point is always ≤ dry bulb)."),
    "RH":  (  1.0, 100.0, "Relative Humidity",
            "RH must be between 1 % and 100 %."),
    "W":   (  0.0,  0.030, "Humidity Ratio",
            "W must be between 0 and 0.030 kg/kg dry air."),
    "v":   (  0.75,  0.96, "Specific Volume",
            "v must be between 0.75 and 0.96 m³/kg dry air."),
}


def check_bounds(value, key):
    lo, hi, name, advice = BOUNDS[key]
    if value < lo or value > hi:
        return (f"⚠️ **{name}** = {value} is outside the valid range "
                f"[{lo} … {hi}].  {advice}")
    return None


def check_secondary(dbt, param_name, val):
    errors = []
    if param_name == "Wet Bulb Temperature (WBT)" and val > dbt:
        errors.append("⚠️ **WBT cannot be greater than DBT.** "
                      "Wet bulb temperature is always ≤ dry bulb temperature.")
    if param_name == "Dew Point Temperature (DPT)" and val > dbt:
        errors.append("⚠️ **DPT cannot be greater than DBT.** "
                      "Dew point temperature is always ≤ dry bulb temperature.")
    return errors


PROPERTY_LABELS = {
    "DBT": ("Dry Bulb Temperature",   "°C"),
    "WBT": ("Wet Bulb Temperature",   "°C"),
    "DPT": ("Dew Point Temperature",  "°C"),
    "RH":  ("Relative Humidity",      "%"),
    "W":   ("Humidity Ratio",         "kg/kg dry air"),
    "h":   ("Enthalpy",               "kJ/kg dry air"),
    "v":   ("Specific Volume",        "m³/kg dry air"),
    "Pv":  ("Vapour Pressure",        "kPa"),
    "mu":  ("Degree of Saturation",   "—"),
}


def state_table(state):
    rows = []
    for key, (name, unit) in PROPERTY_LABELS.items():
        rows.append({"Property": name, "Symbol": key,
                     "Value": state[key], "Unit": unit})
    return pd.DataFrame(rows)


def calc_initial_state(dbt, param_name, param_val):
    if param_name == "Wet Bulb Temperature (WBT)":
        return from_dbt_wbt(dbt, param_val)
    if param_name == "Dew Point Temperature (DPT)":
        return from_dbt_dpt(dbt, param_val)
    if param_name == "Relative Humidity (RH)":
        return from_dbt_rh(dbt, param_val)
    if param_name == "Humidity Ratio (W)":
        return from_dbt_w(dbt, param_val)
    return from_dbt_v(dbt, param_val)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — State Point Calculator
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  State Point Calculator", "⚙️  Process Analysis", "🏭  AHU Chain"])

with tab1:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Inputs")

        dbt_val, dbt_err = validated_input(
            "Dry Bulb Temperature  DBT", 25.0, "sp_dbt", "(°C)")

        second = st.selectbox("Second known parameter", [
            "Wet Bulb Temperature (WBT)",
            "Dew Point Temperature (DPT)",
            "Relative Humidity (RH)",
            "Humidity Ratio (W)",
            "Specific Volume (v)",
        ], key="sp_second")

        defaults = {"Wet Bulb Temperature (WBT)": 18.0,
                    "Dew Point Temperature (DPT)": 12.0,
                    "Relative Humidity (RH)": 50.0,
                    "Humidity Ratio (W)": 0.010,
                    "Specific Volume (v)": 0.855}
        units    = {"Wet Bulb Temperature (WBT)": "(°C)",
                    "Dew Point Temperature (DPT)": "(°C)",
                    "Relative Humidity (RH)": "(%)",
                    "Humidity Ratio (W)": "(kg/kg)",
                    "Specific Volume (v)": "(m³/kg)"}

        sec_val, sec_err = validated_input(
            second, defaults[second], "sp_sec", units[second])

        go = st.button("Calculate", type="primary", key="sp_go")

    # ── Run calculation only when button clicked ──────────────────────────────
    if go:
        errors = []
        if dbt_err:  errors.append(dbt_err)
        if sec_err:  errors.append(sec_err)

        if dbt_val is not None:
            e = check_bounds(dbt_val, "DBT")
            if e: errors.append(e)

        bound_keys = {"Wet Bulb Temperature (WBT)": "WBT",
                      "Dew Point Temperature (DPT)": "DPT",
                      "Relative Humidity (RH)": "RH",
                      "Humidity Ratio (W)": "W",
                      "Specific Volume (v)": "v"}
        if sec_val is not None:
            e = check_bounds(sec_val, bound_keys[second])
            if e: errors.append(e)

        if dbt_val is not None and sec_val is not None:
            errors += check_secondary(dbt_val, second, sec_val)

        if errors:
            for e in errors:
                st.warning(e)
            st.session_state["sp_result"] = None
            st.session_state["sp_png"]    = None
        else:
            try:
                state = calc_initial_state(dbt_val, second, sec_val)
                fig = draw_psychro_chart(
                    states=[{"DBT": state["DBT"], "W": state["W"],
                             "label": "State 1"}],
                    title="Psychrometric Chart — State Point",
                )
                png = _fig_to_png(fig)
                plt.close(fig)
                st.session_state["sp_result"] = state
                st.session_state["sp_png"]    = png
            except Exception as e:
                st.error(f"Calculation error: {e}")
                st.session_state["sp_result"] = None
                st.session_state["sp_png"]    = None

    # ── Display results from session state (survives Download re-run) ─────────
    with right:
        if st.session_state["sp_result"] is not None:
            st.subheader("Calculated State Point")
            st.dataframe(state_table(st.session_state["sp_result"]),
                         use_container_width=True, hide_index=True)
            st.image(st.session_state["sp_png"], use_container_width=True)
            _download_button(st.session_state["sp_png"], "state_point_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Process Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Initial Air State")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        p_dbt_val, p_dbt_err = validated_input(
            "Dry Bulb Temperature  DBT₁", 30.0, "p_dbt", "(°C)")

        p_second = st.selectbox("Second known parameter", [
            "Relative Humidity (RH)",
            "Wet Bulb Temperature (WBT)",
            "Dew Point Temperature (DPT)",
            "Humidity Ratio (W)",
        ], key="p_second")

        p_defaults = {"Relative Humidity (RH)": 60.0,
                      "Wet Bulb Temperature (WBT)": 22.0,
                      "Dew Point Temperature (DPT)": 15.0,
                      "Humidity Ratio (W)": 0.016}
        p_units    = {"Relative Humidity (RH)": "(%)",
                      "Wet Bulb Temperature (WBT)": "(°C)",
                      "Dew Point Temperature (DPT)": "(°C)",
                      "Humidity Ratio (W)": "(kg/kg)"}

        p_val_val, p_val_err = validated_input(
            p_second, p_defaults[p_second], "p_val", p_units[p_second])

    with c2:
        process = st.selectbox("Select Psychrometric Process", [
            "Sensible Heating",
            "Sensible Cooling",
            "Humidification",
            "Dehumidification",
            "Cooling & Dehumidification",
            "Heating & Humidification",
            "Evaporative / Adiabatic Cooling",
            "Adiabatic Mixing",
        ], key="process")

        st.markdown("**End Condition**")

        if process in ("Sensible Heating", "Sensible Cooling"):
            default_end = 40.0 if process == "Sensible Heating" else 18.0
            end_dbt_val, end_dbt_err = validated_input(
                "Final DBT₂", default_end, "end_dbt", "(°C)")

        elif process in ("Humidification", "Dehumidification"):
            hum_by = st.radio("Specify end state by",
                              ["Relative Humidity (RH)", "Humidity Ratio (W)"],
                              key="hum_by")
            if hum_by == "Relative Humidity (RH)":
                end_rh2_val, end_rh2_err = validated_input(
                    "Final RH₂",
                    85.0 if process == "Humidification" else 30.0,
                    "end_rh2", "(%)")
                end_w2_val = end_w2_err = None
            else:
                end_w2_val, end_w2_err = validated_input(
                    "Final W₂", 0.020, "end_w2", "(kg/kg)")
                end_rh2_val = end_rh2_err = None

        elif process in ("Cooling & Dehumidification", "Heating & Humidification"):
            default_dbt2 = 14.0 if "Cooling" in process else 45.0
            end_dbt_val, end_dbt_err = validated_input(
                "Final DBT₂", default_dbt2, "end_dbt2", "(°C)")
            hum_by2 = st.radio("Specify final humidity by",
                               ["Relative Humidity (RH)", "Humidity Ratio (W)"],
                               key="hum_by2")
            if hum_by2 == "Relative Humidity (RH)":
                end_rh2_val, end_rh2_err = validated_input(
                    "Final RH₂", 90.0, "end_rh2b", "(%)")
                end_w2_val = end_w2_err = None
            else:
                end_w2_val, end_w2_err = validated_input(
                    "Final W₂", 0.010, "end_w2b", "(kg/kg)")
                end_rh2_val = end_rh2_err = None

        elif process == "Evaporative / Adiabatic Cooling":
            end_dbt_val, end_dbt_err = validated_input(
                "Final DBT₂", 22.0, "end_dbt_ev", "(°C)")

        elif process == "Adiabatic Mixing":
            st.markdown("*Second air stream*")
            mix_dbt2_val, mix_dbt2_err = validated_input(
                "DBT₂  stream 2", 15.0, "mix_dbt2", "(°C)")
            mix_rh2_val, mix_rh2_err   = validated_input(
                "RH₂   stream 2", 80.0, "mix_rh2",  "(%)")
            m1_val, m1_err = validated_input("Mass flow  m₁", 2.0, "m1", "(kg/s)")
            m2_val, m2_err = validated_input("Mass flow  m₂", 1.0, "m2", "(kg/s)")

    run = st.button("Run Process", type="primary", key="run")

    # ── Run calculation only when button clicked ──────────────────────────────
    if run:
        errors = []
        if p_dbt_err:  errors.append(p_dbt_err)
        if p_val_err:  errors.append(p_val_err)

        if p_dbt_val is not None:
            e = check_bounds(p_dbt_val, "DBT")
            if e: errors.append(e)

        p_bound_keys = {"Relative Humidity (RH)": "RH",
                        "Wet Bulb Temperature (WBT)": "WBT",
                        "Dew Point Temperature (DPT)": "DPT",
                        "Humidity Ratio (W)": "W"}
        if p_val_val is not None:
            e = check_bounds(p_val_val, p_bound_keys[p_second])
            if e: errors.append(e)

        if p_dbt_val is not None and p_val_val is not None:
            errors += check_secondary(p_dbt_val, p_second, p_val_val)

        if process in ("Sensible Heating", "Sensible Cooling",
                       "Evaporative / Adiabatic Cooling"):
            if end_dbt_err: errors.append(end_dbt_err)
            if end_dbt_val is not None:
                e = check_bounds(end_dbt_val, "DBT")
                if e: errors.append(e)
            if (process == "Sensible Heating" and p_dbt_val is not None
                    and end_dbt_val is not None and end_dbt_val <= p_dbt_val):
                errors.append("⚠️ Final DBT₂ must be **greater than** DBT₁ for Sensible Heating.")
            if (process == "Sensible Cooling" and p_dbt_val is not None
                    and end_dbt_val is not None and end_dbt_val >= p_dbt_val):
                errors.append("⚠️ Final DBT₂ must be **less than** DBT₁ for Sensible Cooling.")
            if (process == "Evaporative / Adiabatic Cooling" and p_dbt_val is not None
                    and end_dbt_val is not None and end_dbt_val >= p_dbt_val):
                errors.append("⚠️ Final DBT₂ must be **less than** DBT₁ for Evaporative Cooling.")

        elif process in ("Humidification", "Dehumidification"):
            for v, e in [(end_rh2_val, end_rh2_err), (end_w2_val, end_w2_err)]:
                if e: errors.append(e)
            if end_rh2_val is not None:
                eb = check_bounds(end_rh2_val, "RH")
                if eb: errors.append(eb)
            if end_w2_val is not None:
                eb = check_bounds(end_w2_val, "W")
                if eb: errors.append(eb)

        elif process in ("Cooling & Dehumidification", "Heating & Humidification"):
            if end_dbt_err: errors.append(end_dbt_err)
            if end_dbt_val is not None:
                e = check_bounds(end_dbt_val, "DBT")
                if e: errors.append(e)
            for v, e in [(end_rh2_val, end_rh2_err), (end_w2_val, end_w2_err)]:
                if e: errors.append(e)
            if end_rh2_val is not None:
                eb = check_bounds(end_rh2_val, "RH")
                if eb: errors.append(eb)
            if end_w2_val is not None:
                eb = check_bounds(end_w2_val, "W")
                if eb: errors.append(eb)
            if ("Cooling" in process and p_dbt_val is not None
                    and end_dbt_val is not None and end_dbt_val >= p_dbt_val):
                errors.append("⚠️ Final DBT₂ must be **less than** DBT₁ for Cooling & Dehumidification.")
            if ("Heating" in process and p_dbt_val is not None
                    and end_dbt_val is not None and end_dbt_val <= p_dbt_val):
                errors.append("⚠️ Final DBT₂ must be **greater than** DBT₁ for Heating & Humidification.")

        elif process == "Adiabatic Mixing":
            for v, e in [(mix_dbt2_val, mix_dbt2_err), (mix_rh2_val, mix_rh2_err),
                         (m1_val, m1_err), (m2_val, m2_err)]:
                if e: errors.append(e)
            if mix_dbt2_val is not None:
                eb = check_bounds(mix_dbt2_val, "DBT")
                if eb: errors.append(eb)
            if mix_rh2_val is not None:
                eb = check_bounds(mix_rh2_val, "RH")
                if eb: errors.append(eb)
            if m1_val is not None and m1_val <= 0:
                errors.append("⚠️ Mass flow m₁ must be > 0.")
            if m2_val is not None and m2_val <= 0:
                errors.append("⚠️ Mass flow m₂ must be > 0.")

        if errors:
            for e in errors:
                st.warning(e)
            st.session_state["proc_result"] = None
            st.session_state["proc_png"]    = None
        else:
            try:
                s1 = calc_initial_state(p_dbt_val, p_second, p_val_val)

                if process == "Sensible Heating":
                    s2, res = sensible_heating(s1, end_dbt_val)
                    pairs = [(s1, s2, f"Sensible Heating (+{res['Heat Added (kJ/kg dry air)']} kJ/kg)")]
                elif process == "Sensible Cooling":
                    s2, res = sensible_cooling(s1, end_dbt_val)
                    pairs = [(s1, s2, f"Sensible Cooling (−{res['Heat Removed (kJ/kg dry air)']} kJ/kg)")]
                elif process == "Humidification":
                    s2, res = humidification(s1, w2=end_w2_val, rh2=end_rh2_val)
                    pairs = [(s1, s2, "Humidification")]
                elif process == "Dehumidification":
                    s2, res = dehumidification(s1, w2=end_w2_val, rh2=end_rh2_val)
                    pairs = [(s1, s2, "Dehumidification")]
                elif process == "Cooling & Dehumidification":
                    s2, res = cooling_dehumidification(
                        s1, end_dbt_val, w2=end_w2_val, rh2=end_rh2_val)
                    pairs = [(s1, s2, "Cooling & Dehumidification")]
                elif process == "Heating & Humidification":
                    s2, res = heating_humidification(
                        s1, end_dbt_val, w2=end_w2_val, rh2=end_rh2_val)
                    pairs = [(s1, s2, "Heating & Humidification")]
                elif process == "Evaporative / Adiabatic Cooling":
                    s2, res = evaporative_cooling(s1, end_dbt_val)
                    pairs = [(s1, s2, "Evaporative Cooling")]
                elif process == "Adiabatic Mixing":
                    s_stream2 = from_dbt_rh(mix_dbt2_val, mix_rh2_val)
                    s2, res   = adiabatic_mixing(s1, s_stream2, m1_val, m2_val)
                    pairs     = [(s1, s2, "Mix →"), (s_stream2, s2, "Mix →")]

                fig = draw_psychro_chart(
                    states=(
                        [{"DBT": s1["DBT"],        "W": s1["W"],        "label": "Stream 1"},
                         {"DBT": s_stream2["DBT"], "W": s_stream2["W"], "label": "Stream 2"},
                         {"DBT": s2["DBT"],        "W": s2["W"],        "label": "Mixed"}]
                        if process == "Adiabatic Mixing" else
                        [{"DBT": s1["DBT"], "W": s1["W"], "label": "State 1"},
                         {"DBT": s2["DBT"], "W": s2["W"], "label": "State 2"}]
                    ),
                    process_pairs=pairs,
                    title=f"Psychrometric Chart — {process}",
                )
                png = _fig_to_png(fig)
                plt.close(fig)

                st.session_state["proc_result"] = {
                    "process": process,
                    "s1": s1,
                    "s2": s2,
                    "res": res,
                    "is_mixing": process == "Adiabatic Mixing",
                    "s_stream2": s_stream2 if process == "Adiabatic Mixing" else None,
                }
                st.session_state["proc_png"] = png

            except Exception as e:
                st.error(f"Process error: {e}")
                st.session_state["proc_result"] = None
                st.session_state["proc_png"]    = None

    # ── Display results from session state (survives Download re-run) ─────────
    pr = st.session_state.get("proc_result")
    if pr is not None:
        if pr["is_mixing"]:
            ca, cb, cc, cd = st.columns(4)
            with ca:
                st.markdown("**Stream 1**")
                st.dataframe(state_table(pr["s1"]),
                             use_container_width=True, hide_index=True)
            with cb:
                st.markdown("**Stream 2**")
                st.dataframe(state_table(pr["s_stream2"]),
                             use_container_width=True, hide_index=True)
            with cc:
                st.markdown("**Mixed State**")
                st.dataframe(state_table(pr["s2"]),
                             use_container_width=True, hide_index=True)
            with cd:
                st.markdown("**Process Results**")
                for k, v in pr["res"].items():
                    st.metric(k, v)
        else:
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**Initial State (1)**")
                st.dataframe(state_table(pr["s1"]),
                             use_container_width=True, hide_index=True)
            with cb:
                st.markdown("**Final State (2)**")
                st.dataframe(state_table(pr["s2"]),
                             use_container_width=True, hide_index=True)
            with cc:
                st.markdown("**Process Results**")
                for k, v in pr["res"].items():
                    st.metric(k, v)

        proc_png = st.session_state.get("proc_png")
        if proc_png:
            st.image(proc_png, use_container_width=True)
            _download_button(
                proc_png,
                filename=f"psychro_{pr['process'].lower().replace(' ', '_').replace('/', '_')}.png",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — AHU Multi-Process Chain
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🏭 AHU Multi-Process Chain")
    st.caption(
        "Build a sequence of psychrometric processes as they occur in an Air Handling Unit. "
        "Each step's output automatically becomes the next step's input. "
        "All processes are plotted together on a single chart."
    )

    # ── typical AHU hint ──────────────────────────────────────────────────────
    with st.expander("💡 Typical AHU sequence (click to expand)", expanded=False):
        st.markdown(
            "| Step | Process | Description |\n"
            "|------|---------|-------------|\n"
            "| 0 | — | Outdoor / supply air (initial state) |\n"
            "| 1 | Adiabatic Mixing | Mix outdoor air with return air |\n"
            "| 2 | Sensible Heating | Pre-heat the mixed air |\n"
            "| 3 | Cooling & Dehumidification | Cool and remove moisture |\n"
            "| 4 | Sensible Heating | Re-heat to supply temperature |\n"
            "| 5 | Humidification | Add moisture to target RH |\n"
        )

    # ── helpers scoped to this tab ────────────────────────────────────────────
    _SEC_UNITS = {
        "Relative Humidity (RH)": "(%)",
        "Wet Bulb Temperature (WBT)": "(°C)",
        "Dew Point Temperature (DPT)": "(°C)",
        "Humidity Ratio (W)": "(kg/kg)",
    }
    _SEC_DEFS = {
        "Relative Humidity (RH)": 40.0,
        "Wet Bulb Temperature (WBT)": 28.0,
        "Dew Point Temperature (DPT)": 20.0,
        "Humidity Ratio (W)": 0.014,
    }
    _SEC_BK = {
        "Relative Humidity (RH)": "RH",
        "Wet Bulb Temperature (WBT)": "WBT",
        "Dew Point Temperature (DPT)": "DPT",
        "Humidity Ratio (W)": "W",
    }

    # ── SECTION 1: INITIAL STATE ──────────────────────────────────────────────
    chain = st.session_state["ahu_chain"]
    with st.expander(
        "🔵 Step 0 — Set Initial Air State",
        expanded=(len(chain) == 0),
    ):
        _ia, _ib, _ic = st.columns([1.3, 1.3, 0.8])
        with _ia:
            _ahu_dbt0, _ahu_dbt0_err = validated_input(
                "Dry Bulb Temp DBT₀", 35.0, "ahu_dbt0", "(°C)")
        with _ib:
            _ahu_sec0 = st.selectbox(
                "Second parameter", list(_SEC_UNITS.keys()), key="ahu_sec0")
            _ahu_val0, _ahu_val0_err = validated_input(
                _ahu_sec0, _SEC_DEFS[_ahu_sec0],
                "ahu_val0", _SEC_UNITS[_ahu_sec0])
        with _ic:
            st.write(""); st.write("")
            _init_clicked = st.button(
                "▶ Set Starting Point", key="ahu_init_btn", type="primary")

        if _init_clicked:
            _errs = []
            if _ahu_dbt0_err: _errs.append(_ahu_dbt0_err)
            if _ahu_val0_err: _errs.append(_ahu_val0_err)
            if _ahu_dbt0 is not None:
                _e = check_bounds(_ahu_dbt0, "DBT")
                if _e: _errs.append(_e)
            if _ahu_val0 is not None:
                _e = check_bounds(_ahu_val0, _SEC_BK[_ahu_sec0])
                if _e: _errs.append(_e)
            if _ahu_dbt0 is not None and _ahu_val0 is not None:
                _errs += check_secondary(_ahu_dbt0, _ahu_sec0, _ahu_val0)
            if _errs:
                for _e in _errs: st.warning(_e)
            else:
                try:
                    _s0 = calc_initial_state(_ahu_dbt0, _ahu_sec0, _ahu_val0)
                    st.session_state["ahu_chain"] = [{
                        "state": _s0,
                        "process": "Initial State",
                        "result": {},
                        "mix_stream2": None,
                    }]
                    st.session_state["ahu_png"] = None
                    st.rerun()
                except Exception as _ex:
                    st.error(f"Error setting initial state: {_ex}")

    # re-read after possible rerun
    chain = st.session_state["ahu_chain"]

    if len(chain) > 0:
        # ── CHAIN SUMMARY TABLE ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Process Chain Summary")
        _summary_rows = []
        for _i, _step in enumerate(chain):
            _s = _step["state"]
            _summary_rows.append({
                "Step": _i,
                "Process": _step["process"],
                "DBT (°C)": _s["DBT"],
                "WBT (°C)": _s["WBT"],
                "RH (%)": _s["RH"],
                "W (kg/kg)": _s["W"],
                "h (kJ/kg)": _s["h"],
                "v (m³/kg)": _s["v"],
            })
        st.dataframe(
            pd.DataFrame(_summary_rows),
            use_container_width=True, hide_index=True,
        )

        # ── CONTROL BUTTONS ───────────────────────────────────────────────────
        _bc1, _bc2, _ = st.columns([1, 1, 5])
        with _bc1:
            if st.button("↩ Undo Last Step", key="ahu_undo"):
                st.session_state["ahu_chain"].pop()
                st.session_state["ahu_png"] = None
                st.rerun()
        with _bc2:
            if st.button("🗑 Clear All", key="ahu_clear"):
                st.session_state["ahu_chain"] = []
                st.session_state["ahu_png"] = None
                st.rerun()

        # ── ADD NEXT STEP ─────────────────────────────────────────────────────
        st.markdown("---")
        _cur = chain[-1]["state"]
        st.markdown(
            f"#### ➕ Add Next Step &nbsp;&nbsp; "
            f"<span style='font-size:0.88rem;color:#555;'>"
            f"Current → DBT {_cur['DBT']}°C | WBT {_cur['WBT']}°C | "
            f"RH {_cur['RH']}% | W {_cur['W']} kg/kg | h {_cur['h']} kJ/kg"
            f"</span>",
            unsafe_allow_html=True,
        )

        _ac1, _ac2 = st.columns(2)
        with _ac1:
            _ahu_proc = st.selectbox("Process type", [
                "Sensible Heating",
                "Sensible Cooling",
                "Humidification",
                "Dehumidification",
                "Cooling & Dehumidification",
                "Heating & Humidification",
                "Evaporative Cooling",
                "Adiabatic Mixing",
            ], key="ahu_proc_sel")
            _ahu_lbl = st.text_input(
                "Step label (shown on chart)", value=_ahu_proc, key="ahu_step_lbl")

        # Process-specific parameter inputs
        _ahu_edbt = _ahu_edbt_err = None
        _ahu_erh  = _ahu_erh_err  = None
        _ahu_ew   = _ahu_ew_err   = None
        _ahu_mdbt2 = _ahu_mdbt2_err = None
        _ahu_mrh2  = _ahu_mrh2_err  = None
        _ahu_m1    = _ahu_m1_err    = None
        _ahu_m2    = _ahu_m2_err    = None

        with _ac2:
            if _ahu_proc in ("Sensible Heating", "Sensible Cooling", "Evaporative Cooling"):
                _def2 = (round(_cur["DBT"] + 10, 1) if "Heating" in _ahu_proc
                         else round(_cur["DBT"] - 8, 1))
                _ahu_edbt, _ahu_edbt_err = validated_input(
                    "Target DBT₂", _def2, "ahu_edbt", "(°C)")

            elif _ahu_proc in ("Humidification", "Dehumidification"):
                _ahu_hum_by = st.radio(
                    "Specify end state by",
                    ["RH (%)", "W (kg/kg)"], key="ahu_hum_by", horizontal=True)
                if _ahu_hum_by == "RH (%)":
                    _ahu_erh, _ahu_erh_err = validated_input(
                        "Final RH₂",
                        85.0 if "Humidi" in _ahu_proc else 30.0,
                        "ahu_erh", "(%)")
                else:
                    _ahu_ew, _ahu_ew_err = validated_input(
                        "Final W₂", 0.015, "ahu_ew", "(kg/kg)")

            elif _ahu_proc in ("Cooling & Dehumidification",
                               "Heating & Humidification"):
                _def_dbt2 = 14.0 if "Cooling" in _ahu_proc else 45.0
                _ahu_edbt, _ahu_edbt_err = validated_input(
                    "Final DBT₂", _def_dbt2, "ahu_edbt2", "(°C)")
                _ahu_hum_by2 = st.radio(
                    "Final humidity by",
                    ["RH (%)", "W (kg/kg)"], key="ahu_hum_by2", horizontal=True)
                if _ahu_hum_by2 == "RH (%)":
                    _ahu_erh, _ahu_erh_err = validated_input(
                        "Final RH₂", 90.0, "ahu_erh2", "(%)")
                else:
                    _ahu_ew, _ahu_ew_err = validated_input(
                        "Final W₂", 0.008, "ahu_ew2", "(kg/kg)")

            elif _ahu_proc == "Adiabatic Mixing":
                st.caption("Stream 1 = current state.  Define the 2nd stream:")
                _ahu_mdbt2, _ahu_mdbt2_err = validated_input(
                    "DBT₂ of 2nd stream", 22.0, "ahu_mdbt2", "(°C)")
                _ahu_mrh2, _ahu_mrh2_err = validated_input(
                    "RH₂  of 2nd stream", 60.0, "ahu_mrh2",  "(%)")
                _ahu_m1, _ahu_m1_err = validated_input(
                    "Mass flow m₁ (current stream)", 3.0, "ahu_m1", "(kg/s)")
                _ahu_m2, _ahu_m2_err = validated_input(
                    "Mass flow m₂ (2nd stream)",     1.0, "ahu_m2", "(kg/s)")

        if st.button("➕ Add Step to Chain", key="ahu_add_step", type="primary"):
            _step_errs = []

            if _ahu_proc in ("Sensible Heating", "Sensible Cooling",
                             "Evaporative Cooling"):
                if _ahu_edbt_err: _step_errs.append(_ahu_edbt_err)
                if _ahu_edbt is not None:
                    _e = check_bounds(_ahu_edbt, "DBT")
                    if _e: _step_errs.append(_e)
                if (_ahu_proc == "Sensible Heating" and _ahu_edbt is not None
                        and _ahu_edbt <= _cur["DBT"]):
                    _step_errs.append("⚠️ Target DBT₂ must be > current DBT for Sensible Heating.")
                if (_ahu_proc == "Sensible Cooling" and _ahu_edbt is not None
                        and _ahu_edbt >= _cur["DBT"]):
                    _step_errs.append("⚠️ Target DBT₂ must be < current DBT for Sensible Cooling.")
                if (_ahu_proc == "Evaporative Cooling" and _ahu_edbt is not None
                        and _ahu_edbt >= _cur["DBT"]):
                    _step_errs.append("⚠️ Target DBT₂ must be < current DBT for Evaporative Cooling.")

            elif _ahu_proc in ("Humidification", "Dehumidification"):
                if _ahu_erh_err: _step_errs.append(_ahu_erh_err)
                if _ahu_ew_err:  _step_errs.append(_ahu_ew_err)
                if _ahu_erh is not None:
                    _e = check_bounds(_ahu_erh, "RH")
                    if _e: _step_errs.append(_e)
                if _ahu_ew is not None:
                    _e = check_bounds(_ahu_ew, "W")
                    if _e: _step_errs.append(_e)

            elif _ahu_proc in ("Cooling & Dehumidification",
                               "Heating & Humidification"):
                if _ahu_edbt_err: _step_errs.append(_ahu_edbt_err)
                if _ahu_erh_err:  _step_errs.append(_ahu_erh_err)
                if _ahu_ew_err:   _step_errs.append(_ahu_ew_err)
                if _ahu_edbt is not None:
                    _e = check_bounds(_ahu_edbt, "DBT")
                    if _e: _step_errs.append(_e)
                if _ahu_erh is not None:
                    _e = check_bounds(_ahu_erh, "RH")
                    if _e: _step_errs.append(_e)
                if _ahu_ew is not None:
                    _e = check_bounds(_ahu_ew, "W")
                    if _e: _step_errs.append(_e)
                if ("Cooling" in _ahu_proc and _ahu_edbt is not None
                        and _ahu_edbt >= _cur["DBT"]):
                    _step_errs.append("⚠️ DBT₂ must be < current DBT for Cooling & Dehumidification.")
                if ("Heating" in _ahu_proc and _ahu_edbt is not None
                        and _ahu_edbt <= _cur["DBT"]):
                    _step_errs.append("⚠️ DBT₂ must be > current DBT for Heating & Humidification.")

            elif _ahu_proc == "Adiabatic Mixing":
                for _v, _e in [
                    (_ahu_mdbt2, _ahu_mdbt2_err), (_ahu_mrh2, _ahu_mrh2_err),
                    (_ahu_m1,    _ahu_m1_err),    (_ahu_m2,   _ahu_m2_err),
                ]:
                    if _e: _step_errs.append(_e)
                if _ahu_mdbt2 is not None:
                    _e = check_bounds(_ahu_mdbt2, "DBT")
                    if _e: _step_errs.append(_e)
                if _ahu_mrh2 is not None:
                    _e = check_bounds(_ahu_mrh2, "RH")
                    if _e: _step_errs.append(_e)
                if _ahu_m1 is not None and _ahu_m1 <= 0:
                    _step_errs.append("⚠️ m₁ must be > 0.")
                if _ahu_m2 is not None and _ahu_m2 <= 0:
                    _step_errs.append("⚠️ m₂ must be > 0.")

            if _step_errs:
                for _e in _step_errs: st.warning(_e)
            else:
                try:
                    _mix_s2 = None
                    if _ahu_proc == "Sensible Heating":
                        _s_out, _res = sensible_heating(_cur, _ahu_edbt)
                    elif _ahu_proc == "Sensible Cooling":
                        _s_out, _res = sensible_cooling(_cur, _ahu_edbt)
                    elif _ahu_proc == "Humidification":
                        _s_out, _res = humidification(_cur, w2=_ahu_ew, rh2=_ahu_erh)
                    elif _ahu_proc == "Dehumidification":
                        _s_out, _res = dehumidification(_cur, w2=_ahu_ew, rh2=_ahu_erh)
                    elif _ahu_proc == "Cooling & Dehumidification":
                        _s_out, _res = cooling_dehumidification(
                            _cur, _ahu_edbt, w2=_ahu_ew, rh2=_ahu_erh)
                    elif _ahu_proc == "Heating & Humidification":
                        _s_out, _res = heating_humidification(
                            _cur, _ahu_edbt, w2=_ahu_ew, rh2=_ahu_erh)
                    elif _ahu_proc == "Evaporative Cooling":
                        _s_out, _res = evaporative_cooling(_cur, _ahu_edbt)
                    elif _ahu_proc == "Adiabatic Mixing":
                        _mix_s2 = from_dbt_rh(_ahu_mdbt2, _ahu_mrh2)
                        _s_out, _res = adiabatic_mixing(
                            _cur, _mix_s2, _ahu_m1, _ahu_m2)

                    st.session_state["ahu_chain"].append({
                        "state":       _s_out,
                        "process":     _ahu_lbl or _ahu_proc,
                        "result":      _res,
                        "mix_stream2": _mix_s2,
                    })
                    st.session_state["ahu_png"] = None
                    st.rerun()
                except Exception as _ex:
                    st.error(f"Step calculation error: {_ex}")

        # ── CHART ─────────────────────────────────────────────────────────────
        if len(chain) >= 2:
            st.markdown("---")
            st.markdown("#### 📊 Multi-Process Psychrometric Chart")

            if st.button("Draw Full Chain Chart", key="ahu_draw_chart",
                         type="primary"):
                _chart_states = []
                _chart_pairs  = []
                _mix_stream_idx = 0   # counter for unique mix-stream labels

                # State 0 — initial
                _chart_states.append({
                    "DBT":   chain[0]["state"]["DBT"],
                    "W":     chain[0]["state"]["W"],
                    "label": "S0: Initial",
                })

                for _i in range(1, len(chain)):
                    _step   = chain[_i]
                    _s_from = chain[_i - 1]["state"]
                    _s_to   = _step["state"]
                    _lbl    = _step["process"]

                    # For adiabatic mixing: add 2nd stream as extra state + arrow
                    if _step["mix_stream2"] is not None:
                        _mix_stream_idx += 1
                        _s2 = _step["mix_stream2"]
                        _chart_states.append({
                            "DBT":   _s2["DBT"],
                            "W":     _s2["W"],
                            "label": f"Stream 2\n(mix {_mix_stream_idx})",
                        })
                        _chart_pairs.append((_s2, _s_to, ""))

                    _chart_states.append({
                        "DBT":   _s_to["DBT"],
                        "W":     _s_to["W"],
                        "label": f"S{_i}: {_lbl}",
                    })
                    _chart_pairs.append((_s_from, _s_to, _lbl))

                _fig = draw_psychro_chart(
                    states=_chart_states,
                    process_pairs=_chart_pairs,
                    title="AHU — Multi-Process Psychrometric Chain",
                )
                _png = _fig_to_png(_fig)
                plt.close(_fig)
                st.session_state["ahu_png"] = _png
                st.rerun()

            if st.session_state["ahu_png"]:
                st.image(st.session_state["ahu_png"], use_container_width=True)
                _download_button(
                    st.session_state["ahu_png"], "ahu_chain_chart.png")


# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Psychrometric Calculator · SI Units · P = 101.325 kPa · ASHRAE formulas via psychrolib")
