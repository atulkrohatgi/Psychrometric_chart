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
    Draw the SHF scale as a vertical strip on the extreme right of the chart,
    parallel to the W (y) axis.  Each SHF value is positioned at a y-coordinate
    proportional to its visual process-line angle on the chart, so the scale is
    graphically consistent with the slopes seen on the psychrometric chart.
    Also marks the alignment circle on the main chart body.
    """
    # ── 1. get actual axis bbox for scale-factor computation ─────────────────
    fig.canvas.draw()
    renderer  = fig.canvas.get_renderer()
    bbox      = ax.get_window_extent(renderer=renderer)

    ax_w_px = bbox.width
    ax_h_px = bbox.height
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    T_range = x_hi - x_lo
    W_range = y_hi - y_lo

    # scale converts a data slope dW/dT → visual tan(angle)
    scale = (ax_h_px * T_range) / (ax_w_px * W_range)

    # ── 2. SHF → visual angle → y-position on the vertical scale ─────────────
    # angle=0° (SHF=1, pure sensible) maps to y = _W_REF (alignment circle)
    # angle=90° (SHF=0, pure latent)  maps to y = W_MAX
    # negative angles map below _W_REF toward W_MIN

    def _angle(shf):
        m = _shf_data_slope(shf)
        if m is None:
            return np.pi / 2
        return np.arctan(m * scale)

    def _y_pos(shf):
        a = _angle(shf)
        if a >= 0:
            # 0° → _W_REF,  90° → W_MAX
            return _W_REF + (a / (np.pi / 2)) * (W_MAX - _W_REF)
        else:
            # negative angle → below _W_REF toward W_MIN
            max_neg = -np.pi / 2
            return _W_REF + (a / max_neg) * (_W_REF - W_MIN)

    # All SHF values to show (positive scale on top, negative below ref line)
    shf_pos  = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    shf_neg  = [-0.1, -0.2, -0.3, -0.5, -1.0]
    shf_all  = shf_pos + shf_neg

    # ── 3. create the vertical scale strip ───────────────────────────────────
    # ax.inset_axes uses axes-fraction units; x=1.22 places it well outside
    # the main axes (and the twin g/kg axis), within the reserved right margin.
    # Width 0.09 in axes-fraction ≈ 0.9 inch on a 14" figure — enough for labels.
    axs = ax.inset_axes([1.22, 0.0, 0.09, 1.0])

    # Match the W-axis limits exactly so y-positions align with the main chart
    axs.set_ylim(y_lo, y_hi)
    axs.set_xlim(0, 1)
    axs.axis("off")
    axs.patch.set_facecolor("#eaf0fb")
    axs.patch.set_alpha(0.92)

    # ── 4. draw the vertical scale line ───────────────────────────────────────
    line_x = 0.20   # x position of the scale bar within the strip
    axs.plot([line_x, line_x], [y_lo, y_hi],
             color="#2c3e50", lw=1.6, zorder=3)

    # Horizontal dashed reference line at alignment-circle height
    axs.axhline(_W_REF, color="#7f8c8d", lw=0.8, ls="--", zorder=2)

    # ── 5. tick marks and labels ───────────────────────────────────────────────
    for shf in shf_all:
        yp = _y_pos(shf)
        if not (y_lo <= yp <= y_hi):
            continue                      # skip if outside visible range

        is_neg    = shf < 0
        tick_len  = 0.18                  # tick pointing RIGHT from scale bar
        clr       = "#c0392b" if is_neg else "#2c3e50"
        lw_tick   = 0.9 if shf not in (1.0, 0.5, 0.0) else 1.3

        # Major ticks at 0, 0.5, 1.0 are slightly longer
        tlen = tick_len * (1.4 if shf in (0.0, 0.5, 1.0) else 1.0)
        axs.plot([line_x, line_x + tlen], [yp, yp],
                 color=clr, lw=lw_tick, zorder=4)

        # Label to the right of the tick
        if shf == 0.0:
            txt = "0"
        elif shf < 0:
            txt = f"−{abs(shf):.1f}"
        else:
            txt = f"{shf:.1f}"

        axs.text(line_x + tlen + 0.05, yp, txt,
                 ha="left", va="center",
                 fontsize=7.5 if shf in (0.0, 0.5, 1.0) else 6.5,
                 fontweight="bold" if shf in (0.0, 0.5, 1.0) else "normal",
                 color=clr, zorder=5)

        # Light horizontal guide line across strip
        axs.axhline(yp, color=clr, lw=0.25, alpha=0.35, zorder=1)

    # ── 6. title ──────────────────────────────────────────────────────────────
    axs.text(0.50, y_hi + (y_hi - y_lo) * 0.012,
             "SHF", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#1a252f", zorder=6)
    axs.text(0.50, y_hi - (y_hi - y_lo) * 0.01,
             "Sensible Heat\n/ Total Heat",
             ha="center", va="top",
             fontsize=5.5, color="#5d6d7e", zorder=6)

    # ── 7. alignment circle on the main chart ─────────────────────────────────
    ax.plot(_T_REF, _W_REF, "o",
            color="black", ms=9, zorder=10,
            markeredgecolor="black", markeredgewidth=1.2)
    ax.plot(_T_REF, _W_REF, "o",
            color="white", ms=5, zorder=11)
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
            ax.plot(xs, ys, color="#4a90d9", lw=0.8, alpha=0.45)
            ax.text(xs[-1] + 0.4, ys[-1],
                    f"{int(round(rh*100))}%",
                    fontsize=6.5, color="#2060a0", va="center", alpha=0.85)

    # ── saturation curve (100 % RH) ───────────────────────────────────────────
    xs_sat, ys_sat = [], []
    for t in dbt_arr:
        w = _w_sat(t)
        if w is not None and W_MIN <= w <= W_MAX:
            xs_sat.append(t); ys_sat.append(w)
    ax.plot(xs_sat, ys_sat, color="#1a5fa8", lw=2.2, zorder=3,
            label="Saturation (100 % RH)")

    # ── constant WBT lines (every 5 °C) ──────────────────────────────────────
    for wbt in range(-5, 35, 5):
        xs, ys = [], []
        for t in np.linspace(max(wbt, DBT_MIN), DBT_MAX, 300):
            w = _w_wbt(t, wbt)
            if w is not None and W_MIN <= w <= W_MAX:
                xs.append(t); ys.append(w)
        if xs:
            ax.plot(xs, ys, color="#2ca05a", lw=0.7, alpha=0.35, ls="--")
            ax.text(xs[0] - 0.5, ys[0], f"WBT\n{wbt}°C",
                    fontsize=5.0, color="#1a7a3c", ha="right", va="center",
                    alpha=0.80)

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
            ax.plot(xs, ys, color="#c0392b", lw=0.6, alpha=0.25, ls="-.")
            ax.text(xs[0] - 0.3, ys[0], f"h={h_kj}",
                    fontsize=5.0, color="#922b21", ha="right", va="center",
                    alpha=0.75)

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
            ax.plot(xs, ys, color="#8e44ad", lw=0.65, alpha=0.35, ls=(0, (3, 5)))
            # Label near the saturation curve end (lowest T end)
            ax.text(xs[0] - 0.3, ys[0], f"v={v_val}",
                    fontsize=5.0, color="#6c3483", ha="right", va="center",
                    alpha=0.80)

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
        plt.Line2D([0], [0], color="#1a5fa8", lw=2.2,  label="Saturation (100%RH)"),
        plt.Line2D([0], [0], color="#4a90d9", lw=0.9,  label="Const. RH"),
        plt.Line2D([0], [0], color="#2ca05a", lw=0.8,  ls="--", label="Const. WBT"),
        plt.Line2D([0], [0], color="#c0392b", lw=0.7,  ls="-.", label="Const. Enthalpy"),
        plt.Line2D([0], [0], color="#8e44ad", lw=0.7,  ls=(0,(3,5)), label="Const. Sp. Volume"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=6.5,
              framealpha=0.85, edgecolor="#bdc3c7")

    # Reserve right margin for the SHF vertical strip (ax.inset_axes needs space)
    fig.subplots_adjust(left=0.08, right=0.74, bottom=0.09, top=0.93)

    # ── SHF vertical scale + alignment circle ──────────────────────────────────
    _draw_shf_protractor(fig, ax)

    return fig
