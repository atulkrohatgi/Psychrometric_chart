import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc
import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)
PRESSURE = 101325.0

DBT_MIN, DBT_MAX = -10, 50
W_MIN,   W_MAX   =   0, 0.030

# Reference/alignment point (ASHRAE Chart 1 convention: 24 °C, 50 % RH)
_T_REF = 24.0
_W_REF = 0.00930   # ≈ GetHumRatioFromRelHum(24, 0.5, 101325)

# ── helpers ───────────────────────────────────────────────────────────────────

def _w_sat(dbt):
    try:
        return psychrolib.GetHumRatioFromRelHum(dbt, 1.0, PRESSURE)
    except Exception:
        return None


def _w_rh(dbt, rh_frac):
    try:
        return psychrolib.GetHumRatioFromRelHum(dbt, rh_frac, PRESSURE)
    except Exception:
        return None


def _w_wbt(dbt, wbt):
    if dbt < wbt:
        return None
    try:
        return psychrolib.GetHumRatioFromTWetBulb(dbt, wbt, PRESSURE)
    except Exception:
        return None


# ── SHF helper ────────────────────────────────────────────────────────────────

def _shf_data_slope(shf):
    """Return dW/dT slope (kg/kg per °C) for a given SHF. None → vertical."""
    if abs(shf) < 1e-9:          # SHF = 0 → pure latent → vertical
        return None
    return 1.006 * (1.0 - shf) / (2501.0 * shf)


# ── SHF protractor (inset axes) ───────────────────────────────────────────────

def _draw_shf_protractor(fig, ax):
    """
    Clean vertical SHF scale on the right edge of the chart, parallel to Y-axis.
    Uses SHF value directly as the y-coordinate (linear, evenly spaced).
    Positive SHF: dark blue.  Negative SHF: red.
    Also marks the alignment circle on the main chart body.
    """
    fig.canvas.draw()   # ensure layout is finalised before adding the inset

    # ── SHF values ────────────────────────────────────────────────────────────
    shf_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                  0.0, -0.1, -0.2, -0.3, -0.5]
    SHF_TOP =  1.05    # inset y-range top
    SHF_BOT = -0.58    # inset y-range bottom

    # ── inset axes: right of main axes, full height ───────────────────────────
    axs = ax.inset_axes([1.22, 0.0, 0.09, 1.0])
    axs.set_xlim(0, 1)
    axs.set_ylim(SHF_BOT, SHF_TOP)
    axs.axis("off")
    axs.patch.set_facecolor("#eef2f9")
    axs.patch.set_alpha(0.95)

    # ── thin border ───────────────────────────────────────────────────────────
    for spine in ["top", "bottom", "left", "right"]:
        axs.spines[spine].set_visible(False)

    # ── vertical scale line ───────────────────────────────────────────────────
    lx = 0.18                   # x of the scale bar (axes-fraction of inset)
    axs.plot([lx, lx], [-0.5, 1.0], color="#2c3e50", lw=1.8, zorder=3,
             solid_capstyle="round")

    # ── dashed reference at SHF = 0 ──────────────────────────────────────────
    axs.plot([0.0, lx + 0.30], [0.0, 0.0],
             color="#7f8c8d", lw=0.9, ls="--", zorder=2)

    # ── ticks and labels ──────────────────────────────────────────────────────
    MAJOR = {1.0, 0.5, 0.0}      # bolder ticks and labels

    for shf in shf_values:
        is_neg  = shf < 0
        is_maj  = shf in MAJOR
        clr     = "#c0392b" if is_neg else "#1a252f"
        tick_w  = lx + (0.22 if is_maj else 0.15)   # tick end x
        lw_tick = 1.4 if is_maj else 0.9
        fs      = 8.5 if is_maj else 7.0
        fw      = "bold" if is_maj else "normal"

        # tick mark (pointing right from the scale bar)
        axs.plot([lx, tick_w], [shf, shf], color=clr, lw=lw_tick, zorder=4)

        # label
        if shf == 0.0:
            txt = "0"
        elif is_neg:
            txt = f"−{abs(shf):.1f}"   # proper minus sign
        else:
            txt = f"{shf:.1f}"

        axs.text(tick_w + 0.06, shf, txt,
                 ha="left", va="center",
                 fontsize=fs, fontweight=fw, color=clr, zorder=5)

    # ── section divider at SHF = 0 (thin horizontal line) ────────────────────
    axs.axhline(0.0, color="#aab4be", lw=0.5, zorder=1, xmin=0, xmax=0.8)

    # ── title block ───────────────────────────────────────────────────────────
    axs.text(0.50, 1.04,  "SHF",
             ha="center", va="bottom",
             fontsize=10, fontweight="bold", color="#1a252f", zorder=6)
    axs.text(0.50, 0.99, "Sensible Heat\n/ Total Heat",
             ha="center", va="top",
             fontsize=5.5, color="#5d6d7e", linespacing=1.4, zorder=6)

    # ── "positive" / "negative" micro-labels ─────────────────────────────────
    axs.text(lx - 0.04, 0.55,  "▲ positive",
             ha="right", va="center", fontsize=5, color="#1a252f",
             rotation=90, alpha=0.55)
    axs.text(lx - 0.04, -0.25, "▼ negative",
             ha="right", va="center", fontsize=5, color="#c0392b",
             rotation=90, alpha=0.55)

    # ── alignment circle on main chart ────────────────────────────────────────
    ax.plot(_T_REF, _W_REF, "o",
            color="black", ms=9, zorder=10,
            markeredgecolor="black", markeredgewidth=1.2)
    ax.plot(_T_REF, _W_REF, "o", color="white", ms=5, zorder=11)
    ax.annotate(
        "Alignment Circle\n(24°C, 50%RH)",
        xy=(_T_REF, _W_REF),
        xytext=(18, -30), textcoords="offset points",
        fontsize=6.5, color="#2c3e50", ha="center",
        arrowprops=dict(arrowstyle="-", color="#7f8c8d", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88,
                  ec="#7f8c8d", lw=0.8),
        zorder=12,
    )


# ── main chart builder ────────────────────────────────────────────────────────

def draw_psychro_chart(states=None, process_pairs=None, title="Psychrometric Chart"):
    """
    states        : list of {"DBT": float, "W": float, "label": str}
    process_pairs : list of (state1_dict, state2_dict, label_str)
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f5f8fc")

    dbt_arr = np.linspace(DBT_MIN, DBT_MAX, 400)

    # ── constant RH lines (10 %…90 %) ────────────────────────────────────────
    for rh in np.arange(0.1, 1.0, 0.1):
        xs, ys = [], []
        for t in dbt_arr:
            w = _w_rh(t, rh)
            if w is not None and W_MIN <= w <= W_MAX:
                xs.append(t); ys.append(w)
        if xs:
            ax.plot(xs, ys, color="#1a7dd4", lw=1.0, alpha=0.75)
            ax.text(xs[-1] + 0.4, ys[-1],
                    f"{int(round(rh*100))}%",
                    fontsize=6.5, color="#1055a8", va="center", alpha=1.0)

    # ── saturation curve (100 % RH) ───────────────────────────────────────────
    xs_sat, ys_sat = [], []
    for t in dbt_arr:
        w = _w_sat(t)
        if w is not None and W_MIN <= w <= W_MAX:
            xs_sat.append(t); ys_sat.append(w)
    ax.plot(xs_sat, ys_sat, color="#0047ab", lw=2.8, zorder=3,
            label="Saturation (100 % RH)")

    # ── constant WBT lines (every 5 °C) ──────────────────────────────────────
    for wbt in range(-5, 35, 5):
        xs, ys = [], []
        for t in np.linspace(max(wbt, DBT_MIN), DBT_MAX, 300):
            w = _w_wbt(t, wbt)
            if w is not None and W_MIN <= w <= W_MAX:
                xs.append(t); ys.append(w)
        if xs:
            ax.plot(xs, ys, color="#00a550", lw=0.9, alpha=0.75, ls="--")
            ax.text(xs[0] - 0.5, ys[0], f"WBT\n{wbt}°C",
                    fontsize=5.0, color="#007a38", ha="right", va="center",
                    alpha=1.0)

    # ── constant enthalpy lines (every 10 kJ/kg) ─────────────────────────────
    for h_kj in range(-10, 130, 10):
        xs, ys = [], []
        for t in dbt_arr:
            denom = 2501 + 1.86 * t
            if denom <= 0:
                continue
            w = (h_kj - 1.006 * t) / denom
            if W_MIN <= w <= W_MAX:
                xs.append(t); ys.append(w)
        if xs:
            ax.plot(xs, ys, color="#e8000d", lw=0.8, alpha=0.70, ls="-.")
            ax.text(xs[0] - 0.3, ys[0], f"h={h_kj}",
                    fontsize=5.0, color="#b50009", ha="right", va="center",
                    alpha=1.0)

    # ── constant specific volume lines ────────────────────────────────────────
    Ra = 287.042
    for v_val in [0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]:
        xs, ys = [], []
        for t in dbt_arr:
            T_k = t + 273.15
            w = (v_val * PRESSURE / (Ra * T_k) - 1.0) / 1.6078
            if W_MIN <= w <= W_MAX:
                xs.append(t); ys.append(w)
        if len(xs) > 2:
            ax.plot(xs, ys, color="#9b30d9", lw=0.85, alpha=0.75, ls=(0, (3, 5)))
            # Label near the saturation curve end (lowest T end)
            ax.text(xs[0] - 0.3, ys[0], f"v={v_val}",
                    fontsize=5.0, color="#7a1db8", ha="right", va="center",
                    alpha=1.0)

    # ── state points ──────────────────────────────────────────────────────────
    point_colors = ["#e74c3c", "#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#16a085"]
    label_offsets = [(14, 12), (-14, -18), (14, -18), (-14, 12)]
    if states:
        for i, st in enumerate(states):
            col = point_colors[i % len(point_colors)]
            ox, oy = label_offsets[i % len(label_offsets)]
            ax.plot(st["DBT"], st["W"], "o", color=col, ms=9,
                    zorder=6, markeredgecolor="white", markeredgewidth=1.2)
            ax.annotate(
                st.get("label", f"State {i+1}"),
                xy=(st["DBT"], st["W"]),
                xytext=(ox, oy), textcoords="offset points",
                fontsize=9, color=col, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8,
                                shrinkA=0, shrinkB=4),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.92,
                          ec=col, lw=1.2),
                zorder=7,
            )

    # ── process arrows ────────────────────────────────────────────────────────
    if process_pairs:
        for s_from, s_to, label in process_pairs:
            ax.annotate(
                "",
                xy=(s_to["DBT"], s_to["W"]),
                xytext=(s_from["DBT"], s_from["W"]),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                lw=2.0, mutation_scale=18),
                zorder=5,
            )
            mid_t = (s_from["DBT"] + s_to["DBT"]) / 2
            mid_w = (s_from["W"]   + s_to["W"])   / 2
            dx = s_to["DBT"] - s_from["DBT"]
            dw = s_to["W"]   - s_from["W"]
            length = (dx**2 + dw**2) ** 0.5 or 1e-6
            perp_t  =  dw / length * 1.5
            perp_w  = -dx / length * 0.002
            if perp_w < 0:
                perp_t, perp_w = -perp_t, -perp_w
            ax.text(
                mid_t + perp_t, mid_w + perp_w, label,
                fontsize=8, color="#2c3e50", fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7",
                          alpha=0.92, ec="#f39c12", lw=1.2),
                zorder=8,
            )

    # ── axes & legend ─────────────────────────────────────────────────────────
    ax.set_xlim(DBT_MIN - 1, DBT_MAX + 2)
    ax.set_ylim(W_MIN, W_MAX + 0.001)
    ax.set_xlabel("Dry Bulb Temperature  (°C)", fontsize=12, labelpad=8)
    ax.set_ylabel("Humidity Ratio  W  (kg/kg dry air)", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, color="gray", alpha=0.18, linewidth=0.5)
    ax.tick_params(labelsize=9)

    # Secondary y-axis in g/kg
    ax2 = ax.twinx()
    ax2.set_ylim(W_MIN * 1000, (W_MAX + 0.001) * 1000)
    ax2.set_ylabel("Humidity Ratio  (g/kg dry air)", fontsize=10, labelpad=8)
    ax2.tick_params(labelsize=8)

    # Compact legend for line types
    legend_items = [
        plt.Line2D([0], [0], color="#0047ab", lw=2.8,  label="Saturation (100%RH)"),
        plt.Line2D([0], [0], color="#1a7dd4", lw=1.0,  label="Const. RH"),
        plt.Line2D([0], [0], color="#00a550", lw=0.9,  ls="--", label="Const. WBT"),
        plt.Line2D([0], [0], color="#e8000d", lw=0.8,  ls="-.", label="Const. Enthalpy"),
        plt.Line2D([0], [0], color="#9b30d9", lw=0.85, ls=(0,(3,5)), label="Const. Sp. Volume"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=6.5,
              framealpha=0.85, edgecolor="#bdc3c7")

    # Reserve right margin for the SHF vertical strip (ax.inset_axes needs space)
    fig.subplots_adjust(left=0.08, right=0.74, bottom=0.09, top=0.93)

    # ── SHF vertical scale + alignment circle ──────────────────────────────────
    _draw_shf_protractor(fig, ax)

    return fig
