# Multi-Outcome Power Simulation Framework for GenAI Computing Education Trials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edu-sandbox/cer-genai-multioutcome-power/blob/main/quickstart_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Anonymized companion repository for double-blind peer review of the manuscript:

> *Sample Sizes for Multi-Outcome Trials of Generative AI in Introductory Programming: A Simulation Study of Power, Equivalence, and Software Failure*

A permanent archival release (Zenodo DOI) and full authorship metadata (`CITATION.cff`) will be added after acceptance.

---

## Overview

This repository provides a seeded Monte Carlo simulation and prospective power-planning framework structured under the ADEMP protocol for multi-outcome classroom experiments evaluating generative AI (GenAI) in introductory programming:

- **Coupled 40-minute debugging session DGM** with matched-pair randomization across four tasks (two explicit defects, two hidden semantic defects), cumulative session-clock right-censoring, multidimensional cognitive load subscales (IL, EL, GL), and unassisted transfer.
- **56-cell factorial grid (Blocks A–D, $R = 1{,}000$)** evaluating Holm-adjusted marginal and joint confirmatory power, two-sided FWER, paired and unpaired TOST equivalence ($\Delta_0 = \pm 0.25$), censoring stress, scale correlations, and Intention-to-Treat (ITT) vs. paired Per-Protocol (PP) attrition.
- **Cross-DGM invariance suite** across five calibrated survival-side alternatives: Weibull latency (`W`), aptitude-treatment interaction (`A`), speed–ability correlation (`F`), differential abandonment (`C`), and heavy-tailed Student-$t_4$ latency (`T`).
- **Cluster-robust Cox failure guards** diagnosing and resolving undefined (`NaN`) sandwich standard errors in `statsmodels.PHReg` (`v1` naive vs. `v2` delete-one-student cluster jackknife fallback vs. `v3` pre-event censoring removal).
- **Reusable `--quickstart` planner** and zero-install Google Colab notebook (`quickstart_colab.ipynb`) converting four user-supplied pilot effect sizes ($\text{HR}$, $d_{\text{EL}}$, $d_{\text{GL}}$, $d_{\text{trans}}$) into calibrated sample-size curves with Monte Carlo standard errors (MCSE) and automated Fréchet–Boole verification.

---

## Quick Start (Browser — No Local Setup)

Click the **Open in Colab** badge at the top of this page to launch `quickstart_colab.ipynb`. The notebook clones this repository, runs the built-in self-tests, and computes endpoint-specific and joint power curves from four aggregate pilot effect sizes in roughly 1–2 minutes.

### Privacy & Data Safety Notice

All datasets generated and archived in this repository are **100% synthetic**. The Google Colab notebook processes only four aggregate numerical inputs and simulates synthetic cohorts inside the cloud runtime. **Do not upload real student telemetry, grades, or identifiable classroom records to Google Colab.** For empirical analyses of actual classroom data, clone this repository and execute the analysis scripts strictly locally.

---

## Local Installation & Tested Environment

The reported manuscript results were executed under the following reference environment:

- **Python:** `3.12.16`
- **Packages:** `numpy==2.5.5`, `pandas==3.1.1`, `scipy==1.18.1`, `statsmodels==0.15.0`, `matplotlib==3.11.2`

```bash
git clone [https://github.com/edu-sandbox/cer-genai-multioutcome-power.git](https://github.com/edu-sandbox/cer-genai-multioutcome-power.git)
cd cer-genai-multioutcome-power

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on software version specificity:** The `NaN` cluster-robust sandwich standard-error failure in `statsmodels.PHReg` documented in the manuscript was observed under `statsmodels==0.15.0` with Efron's tie method and student-level clustering.

---

## Repository Structure

### Core Python Codebase
- `sim_core.py` — Baseline multi-outcome classroom DGM (matched-pair randomization, 40-min session clock, cognitive load, transfer).
- `analysis_core.py` — Primary confirmatory pipeline (clustered Cox `PHReg`, paired Wilcoxon, baseline-adjusted ANCOVA, Holm FWER adjustment, TOST).
- `scenarios_simulation.py` — Scenario runner, Holm matrix utilities, paired `per_protocol()` exclusion, and `v2` delete-one-student cluster jackknife (`cox_jackknife`).
- `p5_drop_early.py` — `v3` Cox wrapper (`h1a_v3`) removing pre-event right-censored records prior to sandwich estimation.
- `p6_time_ratio.py` — Secondary baseline-adjusted log-time sensitivity model.
- `grid_runner.py` — 56-cell factorial simulation grid (Blocks A–D).
- `variant_runner_safe.py` — Self-contained runner for alternative DGMs (`W/A/F/C/T`), HR bisection calibration, Fréchet–Boole invariant auditor, cross-DGM invariance builder, and `--quickstart` planner.

### Companion Notebook & Archived Artifacts
- `quickstart_colab.ipynb` — Four-cell Google Colab companion notebook for `--quickstart`.
- Aggregated CSV tables: `grid_summary.csv`, `blockD_itt_vs_pp.csv`, `invariance_lam0.60.csv`, `invariance_lam1.00.csv`, `quickstart_grid.csv`, `quickstart_result.csv`, `TableA1_full_56_cells.csv`.
- Publication figures (300 DPI): `Fig1_power_surfaces.png` through `Fig5_quickstart_demo.png`.

---

## Reproducibility & Usage Commands

### 1. Run Built-In Self-Tests and Boole-Inequality Audits
Verify mathematical invariants (Fréchet–Boole bounds, Weibull and Student-$t_4$ transforms, bisection bracket checks, and left-tail log-time floor) and audit any summary CSV table:

```bash
python variant_runner_safe.py --selftest
python variant_runner_safe.py --check-csv grid_summary.csv
python variant_runner_safe.py --check-csv quickstart_grid.csv
```

### 2. Run Prospective Sample-Size Planning (`--quickstart`)
Reproduce Table 6 and Figure 5 using your own (or the manuscript's benchmark) four pilot effect sizes:

```bash
python variant_runner_safe.py --quickstart \
    --pilot-hr 1.80 \
    --pilot-d-el -0.40 \
    --pilot-d-gl -0.30 \
    --pilot-d-trans -0.50 \
    --power 0.80 \
    --reps 500 \
    --workers 4
```

Outputs are saved to `variant_output_safe/quickstart_result.csv` and `variant_output_safe/quickstart_grid.csv`. To check sensitivity under alternative survival distributions, add `--variant W` (Weibull) or `--variant T` (Student-$t_4$).

### 3. Reproduce the Baseline 56-Cell Factorial Grid (Blocks A–D)
```bash
python grid_runner.py --run
```

### 4. Reproduce Cross-DGM Invariance (Table 5 & Figure 4)
Verify neutrality of zero-perturbation variants (`A`, `F`, `C`), calibrate $\kappa$ to the baseline pseudo-true HR at $\lambda = 0.6$, run each variant, and compile invariance tables:

```bash
# Neutrality verification against base generator
python variant_runner_safe.py --check-neutral

# Calibrate and run each alternative DGM (repeat for A, C, F, T, W)
for V in A C F T W; do
    python variant_runner_safe.py --calibrate --variant "$V" --tag "${V}_default"
    python variant_runner_safe.py --run --variant "$V" --reps 1000 --workers 4 --tag "${V}_default"
done

# Build cross-DGM invariance summary tables (lam = 0.6 and lam = 1.0)
python variant_runner_safe.py --invariance --lam 0.6 --power 0.80
python variant_runner_safe.py --invariance --lam 1.0 --power 0.80
```

Add `--zero-solver off` to disable the floor-level control-group `zero_solver` option across all non-null grid cells.

---

## License

Released under the [MIT License](LICENSE).
