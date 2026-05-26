import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
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


# ── SHF protractor ────────────────────────────────────────────────────────────

def _draw_shf_protractor(fig, ax):
    """
    Geometrically correct SHF protractor drawn in main-chart coordinates
    (clip_on=False so it lives just right of DBT_MAX).

    The scale bar is placed at x_s °C.  For each SHF value the tick is at:
        W_tick = W_ref + slope * (x_s - T_ref)
        slope  = 1.006 * (1 - SHF) / (2501 * SHF)

    Only SHF values whose tick falls within [W_REF, W_MAX - margin] are drawn
    as individual ticks; values that would clip above W_MAX are summarised with
    an upward arrow at the scale-bar top, eliminating all vertical overlap.
    """
    # ── geometry ──────────────────────────────────────────────────────────────
    x_s     = DBT_MAX + 4          # scale x-position (°C, right of chart)
    delta_x = x_s - _T_REF        # horiz distance from alignment circle

    def _w_tick(shf):
        if abs(shf) < 1e-9:
            return W_MAX
        slope = 1.006 * (1.0 - shf) / (2501.0 * shf)
        return _W_REF + slope * delta_x

    # ── determine which SHF ticks fit inside the chart (with 0.001 safety gap)
    ALL_SHF   = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    THRESHOLD = W_MAX - 0.001          # strict upper limit for visible ticks
    visible   = [s for s in ALL_SHF if _w_tick(s) <= THRESHOLD]
    hidden    = [s for s in ALL_SHF if _w_tick(s) >  THRESHOLD]

    # Top of scale bar sits just above the highest visible tick
    w_bar_top = _w_tick(visible[-1]) + 0.0015 if visible else W_MAX

    # ── scale bar ─────────────────────────────────────────────────────────────
    ax.plot([x_s, x_s], [_W_REF - 0.0005, w_bar_top],
            color="#2c3e50", lw=1.8, clip_on=False, zorder=20,
            solid_capstyle="round")

    # ── ticks & labels for visible SHF values ────────────────────────────────
    MAJOR = {1.0, 0.5}
    for shf in visible:
        w      = _w_tick(shf)
        is_maj = shf in MAJOR
        clr    = "#1a252f"
        t_len  = 0.75 if is_maj else 0.50
        lw_t   = 1.4  if is_maj else 0.9
        fs     = 8.5  if is_maj else 7.5
        fw     = "bold" if is_maj else "normal"

        ax.plot([x_s, x_s + t_len], [w, w],
                color=clr, lw=lw_t, clip_on=False, zorder=21)
        ax.text(x_s + t_len + 0.3, w, f"{shf:.1f}",
                fontsize=fs, color=clr, fontweight=fw,
                ha="left", va="center", clip_on=False, zorder=22)

    # ── cap: upward arrow + list of hidden (high-latent) SHF values ──────────
    if hidden:
        hidden_str = ", ".join(f"{s:.1f}" for s in hidden) + " →↑"
        ax.annotate(
            "",
            xy   =(x_s, w_bar_top + 0.0010),
            xytext=(x_s, w_bar_top - 0.0005),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                            lw=1.2, mutation_scale=10),
            clip_on=False, zorder=22,
        )
        ax.text(x_s + 0.4, w_bar_top + 0.0012, hidden_str,
                fontsize=6.5, color="#5d6d7e", style="italic",
                ha="left", va="bottom", clip_on=False, zorder=22)

    # ── horizontal reference: alignment circle → SHF = 1.0 ───────────────────
    ax.plot([_T_REF, x_s], [_W_REF, _W_REF],
            color="#aab4be", lw=0.9, ls="--", clip_on=False, zorder=8)

    # ── title block: positioned well above the scale bar top ─────────────────
    title_y    = w_bar_top + 0.0045
    subtitle_y = w_bar_top + 0.0030
    ax.text(x_s + 0.3, title_y, "SHF",
            fontsize=10, color="#1a252f", fontweight="bold",
            ha="left", va="bottom", clip_on=False, zorder=22)
    ax.text(x_s + 0.3, subtitle_y, "Sensible Heat / Total Heat",
            fontsize=5.5, color="#5d6d7e",
            ha="left", va="bottom", clip_on=False, zorder=22)

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

    # ── constant enthalpy lines (every 10 kJ/kg) with visible scale ──────────
    _lbl_kw = dict(fontsize=8, color="#b50009", fontweight="bold",
                   clip_on=False,
                   bbox=dict(boxstyle="round,pad=0.18", fc="white",
                             alpha=0.88, ec="#e8000d", lw=0.6))
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
            ax.plot(xs, ys, color="#e8000d", lw=0.9, alpha=0.75, ls="-.")

            # Lines that enter from the LEFT boundary → label on left scale
            if xs[0] <= DBT_MIN + 0.5:
                ax.text(xs[0] - 0.7, ys[0], f"{h_kj}",
                        ha="right", va="center", **_lbl_kw)

            # Lines that enter from the TOP boundary (W_MAX) → label along top
            if ys[0] >= W_MAX - 0.001:
                ax.text(xs[0], W_MAX + 0.00035, f"{h_kj}",
                        ha="center", va="bottom", **_lbl_kw)

    # Enthalpy scale axis labels
    ax.text(DBT_MIN - 3.8, W_MAX * 0.55, "h\n(kJ/kg)",
            fontsize=8.5, color="#b50009", fontweight="bold",
            ha="center", va="center", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0",
                      alpha=0.9, ec="#e8000d", lw=0.8))
    ax.text((DBT_MIN + DBT_MAX) * 0.38, W_MAX + 0.00085, "h  (kJ/kg)",
            fontsize=8.5, color="#b50009", fontweight="bold",
            ha="center", va="bottom", clip_on=False)

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
    label_offsets = [(38, 14), (-38, -18), (38, -18), (-38, 14)]
    if states:
        for i, st in enumerate(states):
            col  = point_colors[i % len(point_colors)]
            ox, oy = label_offsets[i % len(label_offsets)]
            # Fix 1: clamp W to saturation curve — no point can plot above it
            w_sat_val = _w_sat(st["DBT"]) or W_MAX
            w_plot    = min(st["W"], w_sat_val)
            ax.plot(st["DBT"], w_plot, "o", color=col, ms=9,
                    zorder=6, markeredgecolor="white", markeredgewidth=1.2)
            ax.annotate(
                st.get("label", f"State {i+1}"),
                xy=(st["DBT"], w_plot),
                xytext=(ox, oy), textcoords="offset points",
                fontsize=9, color=col, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8,
                                shrinkA=0, shrinkB=4),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.92,
                          ec=col, lw=1.2),
                zorder=7,
            )

    # ── process arrows ────────────────────────────────────────────────────────
    # Pre-compute axes geometry for pixel-accurate perpendicular label offsets
    fig_w, fig_h   = fig.get_size_inches()
    ax_pos         = ax.get_position()
    plot_w_pts     = ax_pos.width  * fig_w * 72   # axes width  in pts
    plot_h_pts     = ax_pos.height * fig_h * 72   # axes height in pts
    x_data_range   = ax.get_xlim()[1] - ax.get_xlim()[0]
    w_data_range   = ax.get_ylim()[1] - ax.get_ylim()[0]
    LABEL_OFFSET_PTS = 32          # perpendicular distance from line to label

    def _proc_label(mid_t, mid_w, dx, dw, label):
        """Place a process label perpendicular to the line, always above it."""
        # Convert direction to display (pts) space
        dx_pts = dx / x_data_range * plot_w_pts
        dw_pts = dw / w_data_range * plot_h_pts
        len_pts = (dx_pts**2 + dw_pts**2) ** 0.5 or 1.0
        # 90° counter-clockwise perpendicular
        ox =  -dw_pts / len_pts * LABEL_OFFSET_PTS
        oy =   dx_pts / len_pts * LABEL_OFFSET_PTS
        # Ensure label sits above the line (positive display-y = upward)
        if oy < 0:
            ox, oy = -ox, -oy
        ax.annotate(
            label,
            xy=(mid_t, mid_w),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=8, color="#2c3e50", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7",
                      alpha=0.92, ec="#f39c12", lw=1.2),
            zorder=8,
        )

    def _arrow(x1, y1, x2, y2, col="#2c3e50", lw=2.0):
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=col,
                            lw=lw, mutation_scale=18),
            zorder=5)

    if process_pairs:
        for item in process_pairs:
            s_from, s_to, label = item[0], item[1], item[2]

            # Fix 1: clamp endpoints to saturation
            w_sat_from = _w_sat(s_from["DBT"]) or W_MAX
            w_sat_to   = _w_sat(s_to["DBT"])   or W_MAX
            t1, w1 = s_from["DBT"], min(s_from["W"], w_sat_from)
            t2, w2 = s_to["DBT"],   min(s_to["W"],   w_sat_to)

            # Fix 2: Heating & Humidification → two distinct sub-arrows
            if "Heating" in label and "Humidif" in label:
                # Step A — sensible heating: horizontal at constant W
                t_mid = t2;  w_mid = w1
                _arrow(t1, w1, t_mid, w_mid, col="#c0392b")   # red → heating
                _arrow(t_mid, w_mid, t2, w2,  col="#2980b9")   # blue → humidification
                # Label on the humidification segment (vertical leg)
                _proc_label(
                    (t_mid + t2) / 2, (w_mid + w2) / 2,
                    t2 - t_mid, w2 - w_mid, label
                )
            else:
                # Normal single arrow
                _arrow(t1, w1, t2, w2)
                dx = t2 - t1
                dw = w2 - w1
                _proc_label((t1 + t2) / 2, (w1 + w2) / 2, dx, dw, label)

    # ── axes & legend ─────────────────────────────────────────────────────────
    ax.set_xlim(DBT_MIN - 1, DBT_MAX + 2)
    ax.set_ylim(W_MIN, W_MAX + 0.001)
    ax.set_xlabel("Dry Bulb Temperature  (°C)", fontsize=12, labelpad=8)
    ax.set_ylabel("Humidity Ratio  W  (kg/kg dry air)", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Major ticks: every 5 °C on X, every 0.005 kg/kg on Y
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.005))

    # Minor ticks: every 1 °C on X, every 0.001 kg/kg on Y
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))

    # Major grid
    ax.grid(True, which="major", color="gray", alpha=0.22, linewidth=0.6)
    # Minor grid (lighter)
    ax.grid(True, which="minor", color="gray", alpha=0.10, linewidth=0.3)

    # Major tick style
    ax.tick_params(which="major", labelsize=9, length=5, width=0.8)
    # Minor tick style (no labels, shorter)
    ax.tick_params(which="minor", labelsize=0, length=3, width=0.5)

    # Secondary y-axis in g/kg
    ax2 = ax.twinx()
    ax2.set_ylim(W_MIN * 1000, (W_MAX + 0.001) * 1000)
    ax2.set_ylabel("Humidity Ratio  (g/kg dry air)", fontsize=10, labelpad=70)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.tick_params(which="major", labelsize=8, length=5, width=0.8)
    ax2.tick_params(which="minor", labelsize=0, length=3, width=0.5)

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

    # Right margin leaves room for the SHF scale drawn with clip_on=False
    fig.subplots_adjust(left=0.08, right=0.74, bottom=0.09, top=0.93)

    # ── SHF vertical scale + alignment circle ──────────────────────────────────
    _draw_shf_protractor(fig, ax)

    return fig
