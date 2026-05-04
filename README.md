# Psychrometric Calculator

A web-based psychrometric calculator built with Streamlit. Computes moist air properties and analyses 8 psychrometric processes, with results plotted on a full ASHRAE-style psychrometric chart.

## Features

### State Point Calculator
Enter dry-bulb temperature (DBT) plus **any one** of the following:
- Wet-bulb temperature (WBT)
- Dew-point temperature (DPT)
- Relative humidity (RH %)
- Humidity ratio / absolute humidity (W, kg/kg)
- Specific volume (v, m³/kg)

Outputs all 9 psychrometric properties:

| Symbol | Property | Unit |
|--------|----------|------|
| DBT | Dry-bulb temperature | °C |
| WBT | Wet-bulb temperature | °C |
| DPT | Dew-point temperature | °C |
| RH | Relative humidity | % |
| W | Humidity ratio | kg/kg dry air |
| h | Specific enthalpy | kJ/kg dry air |
| v | Specific volume | m³/kg dry air |
| Pv | Vapour pressure | kPa |
| μ | Degree of saturation | — |

### Process Analysis
Eight psychrometric processes with initial/final state tables and key process results:

1. **Sensible Heating** — DBT increases, W constant
2. **Sensible Cooling** — DBT decreases, W constant
3. **Humidification** — W increases at constant DBT
4. **Dehumidification** — W decreases at constant DBT
5. **Cooling & Dehumidification** — DBT and W both decrease
6. **Heating & Humidification** — DBT and W both increase
7. **Evaporative Cooling** — adiabatic saturation (DBT decreases, W increases along WBT line)
8. **Adiabatic Mixing** — two air streams mixed at a given mass-flow ratio

### Psychrometric Chart
Full ASHRAE-style SI chart plotted with matplotlib:

- **Saturation curve** (100 % RH boundary)
- **Constant RH lines** — 10 %, 20 %, … 90 %
- **Constant WBT lines** — 5 °C to 30 °C
- **Constant enthalpy lines** — 20 to 90 kJ/kg
- **Constant specific volume lines** — 0.78 to 0.94 m³/kg
- **SHF (Sensible Heat Factor) scale** — vertical linear strip on the right-hand side
- **Alignment circle** — reference point at 24 °C / 50 % RH per ASHRAE convention
- **State point markers** with labelled property callouts
- **Process arrows** with direction indicators and perpendicular process labels

## Tech Stack

| Component | Library |
|-----------|---------|
| UI / web server | [Streamlit](https://streamlit.io) ≥ 1.32 |
| Psychrometric calculations | [psychrolib](https://github.com/psychrometrics/psychrolib) ≥ 2.5 |
| Chart rendering | [matplotlib](https://matplotlib.org) ≥ 3.8 |
| Numerical utilities | [NumPy](https://numpy.org) ≥ 1.26 |
| Results tables | [pandas](https://pandas.pydata.org) ≥ 2.1 |

**Language:** Python 3.11+

## Units & Pressure

All calculations use **SI units** at standard mean sea-level pressure:

```
P = 101 325 Pa  (101.325 kPa)
```

Temperatures in °C, specific enthalpy in kJ/kg dry air, specific volume in m³/kg dry air.

## Input Validation

The app checks every input before calculating and shows a warning for out-of-range values:

| Input | Valid range |
|-------|-------------|
| DBT | −10 °C to 50 °C |
| WBT | must be ≤ DBT |
| DPT | must be ≤ DBT |
| RH | 1 % to 100 % |
| W | 0 to 0.030 kg/kg |
| v | 0.75 to 0.96 m³/kg |

Process inputs are also direction-checked (e.g., heating requires the target DBT to be higher than the initial DBT).

## Project Structure

```
application_cladue/
├── app.py            # Streamlit UI — two tabs, all inputs & validation
├── psychro_calc.py   # Core psychrometric calculations (psychrolib wrappers)
├── processes.py      # 8 process functions
├── psychro_chart.py  # Matplotlib chart with all background lines & SHF scale
├── requirements.txt  # Python dependencies
└── render.yaml       # Render deployment configuration
```

## Local Setup

**Prerequisites:** Python 3.11 or later, pip

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd application_cladue

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

## Deployment to Render

The repository includes a `render.yaml` that configures the service automatically.

1. Push the project to a GitHub (or GitLab) repository.
2. Log in to [render.com](https://render.com) and click **New → Web Service**.
3. Connect your repository — Render detects `render.yaml` and pre-fills all settings.
4. Click **Create Web Service**. Render installs dependencies and starts the app.

The service is configured as:

```yaml
runtime: python
buildCommand: pip install -r requirements.txt
startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

## Usage Guide

### Tab 1 — State Point Calculator

1. Select your **secondary input type** from the dropdown (WBT, DPT, RH, W, or v).
2. Type the **DBT** value and the chosen secondary value into the text fields.
3. Click **Calculate**. A results table with all 9 properties and the state point plotted on the psychrometric chart are displayed below.

### Tab 2 — Process Analysis

1. **Define State 1** using the same DBT + secondary input method as Tab 1.
2. Choose the **process type** from the dropdown.
3. Enter the required process parameter(s) — each process shows only the fields it needs (e.g., target DBT for sensible heating, target W or RH for humidification, mass-flow ratio for adiabatic mixing).
4. Click **Calculate Process**. The app shows:
   - Initial and final state tables
   - Key process results (heat added/removed, moisture added/removed, SHF where applicable)
   - The process plotted as an arrow on the psychrometric chart

### Reading the SHF Scale

The vertical strip on the right side of the chart is the **Sensible Heat Factor protractor**:

- The **alignment circle** on the main chart marks the reference point (24 °C, 50 % RH).
- Draw a line from the alignment circle to the SHF value on the vertical scale; this line is parallel to the actual process line on the chart.
- Values above the dashed zero-line (SHF > 0) indicate net sensible gain; values below (SHF < 0) indicate net latent gain with sensible loss.

## License

MIT
