# 🔋 EV Battery Health & Degradation Dataset (50K Records)

Predict State of Health, Remaining Useful Life, and failure risk for electric
vehicle battery packs — a physics-informed synthetic dataset spanning four
cell chemistries, 2,600 batteries, and 50,000 longitudinal observations.

## 🔋 Overview

EV adoption is accelerating worldwide, and the battery pack — the single most
expensive component of an electric vehicle — degrades unevenly depending on
chemistry, climate, charging habits, and driving patterns. This creates two
large and growing markets: **predictive maintenance** (warning fleet
operators and drivers before failure) and the **battery second-life economy**
(reusing retired EV packs for stationary storage once they fall below ~80%
State of Health).

This dataset simulates realistic longitudinal health telemetry for EV battery
packs across their service life, grounded in published cycle-life and
calendar-aging behavior for four common chemistries (LFP, NMC, NCA, LMO). It
is designed to support regression, classification, time-series, and
unsupervised learning use cases out of the box.

## 📊 Content

| File | Rows | Description |
|---|---|---|
| `ev_battery_health.csv` | 50,000 | Main dataset: one row per battery observation |
| `data_dictionary.csv` | 35 | Column name, dtype, unit, description, example value |
| `sample_eda_notebook.ipynb` | - | Starter notebook: EDA, visualization, baseline models |

The main file contains **35 columns** across 2,600 unique batteries
(`battery_id`), each with ~10–40 longitudinal observations
(`observation_id`) captured over its service life.

## 🗂️ Column Descriptions

| Column | Type | Unit | Description |
|---|---|---|---|
| battery_id | string | - | Unique battery pack identifier |
| observation_id | string | - | Unique observation/reading identifier |
| chemistry | categorical | - | LFP / NMC / NCA / LMO |
| manufacturer | categorical | - | Synthetic pack manufacturer |
| vehicle_model | categorical | - | EV model/platform |
| pack_capacity_kwh | float | kWh | Nominal rated capacity |
| manufacture_date | date | - | Pack manufacture date |
| observation_date | date | - | Date of this reading |
| battery_age_days | int | days | Age at observation |
| cycle_count | int | cycles | Cumulative equivalent full cycles |
| climate_zone | categorical | - | Operating climate |
| avg_daily_distance_km | float | km | Average daily driving distance |
| avg_charging_power_kw | float | kW | Average charging power |
| fast_charge_ratio | float | 0–1 | Share of DC fast-charge sessions |
| avg_depth_of_discharge_pct | float | % | Average depth of discharge |
| avg_state_of_charge_pct | float | % | Average state of charge |
| charge_cycles_per_month | float | /month | Charge frequency |
| total_energy_throughput_kwh | float | kWh | Cumulative energy throughput |
| avg_ambient_temp_c | float | °C | Average ambient temperature |
| humidity_pct | float | % | Average humidity |
| altitude_m | float | m | Operating altitude |
| internal_resistance_mohm | float | mΩ | Internal resistance (rises with age) |
| voltage_avg_v | float | V | Average pack voltage |
| current_avg_a | float | A | Average current draw |
| charge_time_avg_min | float | min | Average charge session duration |
| peak_temp_during_fast_charge_c | float | °C | Peak fast-charge temperature |
| capacity_fade_pct | float | % | Cumulative fade (100 − SOH) |
| service_count | int | count | Maintenance events to date |
| last_service_date | date | - | Most recent service date |
| bms_firmware_version | categorical | - | BMS firmware version |
| recall_flag | binary | 0/1 | Subject to manufacturer recall |
| **state_of_health_pct** | float **[TARGET]** | % | Current capacity vs. nominal |
| **remaining_useful_life_cycles** | int **[TARGET]** | cycles | Cycles remaining until 80% SOH |
| **failure_risk_label** | categorical **[TARGET]** | - | Low / Medium / High risk |
| **degradation_category** | categorical **[TARGET]** | - | Normal / Accelerated / Critical |

Full dictionary with example values: `data_dictionary.csv`.

## 🎯 Suggested Use Cases

- **Regression** — predict `state_of_health_pct` or `remaining_useful_life_cycles`
- **Classification** — predict `failure_risk_label` or `degradation_category` (imbalanced classes — good for testing resampling/weighting techniques)
- **Anomaly detection** — flag sensor outliers in resistance, temperature, or charge time
- **Time-series forecasting** — model SOH trajectory per `battery_id` over `cycle_count`/`battery_age_days`
- **Clustering** — segment battery cohorts by usage persona or degradation pattern
- **Feature importance / SHAP** — identify which usage and environmental factors drive degradation fastest

## ⚙️ Methodology

Data is **fully synthetic**, generated with Python (pandas/numpy, seed=42) using a
physics-informed degradation model:

- Each of the four chemistries has distinct base fade rate, temperature
  sensitivity, depth-of-discharge sensitivity, and fast-charge penalty
  parameters, loosely grounded in published cycle-life ranges (e.g. LFP
  chemistries typically retaining high capacity past 3,000+ cycles under
  moderate conditions, while NMC/NCA/LMO chemistries fade faster but offer
  higher energy density).
- Each battery has persistent "random effect" traits (manufacturing quality
  factor, driver usage intensity, baseline DOD, baseline fast-charge ratio)
  so that observations from the same battery are correlated over time rather
  than independently random — mimicking real fleet telemetry.
- State of Health follows a non-linear fade curve combining cycle-based and
  calendar-based aging, stressed by high temperature, high DOD, and frequent
  fast charging.
- Electrical measurements (internal resistance, etc.) are derived from SOH
  and chemistry rather than generated independently, preserving realistic
  correlations.
- 3–8% missingness was injected per feature column (never on IDs or targets).
- 1–2% outliers were injected into key sensor columns to simulate telemetry
  noise/faults.
- Failure risk classes are intentionally imbalanced (~85% Low / ~9% Medium /
  ~6% High), matching real-world fleet failure-rate skew.

## ⚠️ Limitations

This is a **synthetic dataset**, not real fleet telemetry. While the
degradation physics, chemistry parameters, and noise characteristics are
grounded in publicly available battery research, no data here comes from an
actual vehicle or battery testing lab. Do not use this dataset to make real
engineering, safety, or warranty decisions. It is intended for **educational,
benchmarking, and methodology-development purposes**.

## 🙏 Acknowledgements

Inspired by the growing body of public battery-aging research (e.g. NASA
Prognostics Center of Excellence battery datasets, academic LFP/NMC
cycle-life studies) and the real-world need for accessible, large-scale
predictive-maintenance datasets in the EV space.

## 📜 License

 **CC0: Public Domain**

## 🧪 Reproducibility


**Train/test split guidance:** split by `battery_id`, not by row, to avoid
data leakage — a held-out battery's observations should never appear in both
train and test sets.
