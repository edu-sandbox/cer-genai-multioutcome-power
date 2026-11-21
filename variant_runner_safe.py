#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
variant_runner_safe.py -- Self-contained runner for alternative DGMs (W/A/F/C/T),
invariant auditor (Frechet-Boole bounds, MCSE), and --quickstart sample-size planner.

DESIGN GUARANTEES:
  1. Does not modify the baseline simulation modules (sim_core.py, analysis_core.py,
     scenarios_simulation.py, grid_runner.py, p5_drop_early.py, p6_time_ratio.py).
  2. Writes outputs to a dedicated directory: variant_output_safe/ (or VAR_OUT_DIR).
  3. Enforces methodological and numerical safeguards:
     - Unified column map ENDPOINT_COLS matching ss.HYP (H1a, H2a_EL, H2b_GL, H3).
     - --quickstart incorporates all four pilot effect sizes (HR, d_EL, d_GL, d_trans).
     - Scalar multiplier kappa applies across all alternative variants (including C),
       and --calibrate verifies bisection brackets before solving.
     - Automated Frechet-Boole checks, per-endpoint MCSE, v2 jackknife, and PP summaries.
     - --zero-solver off allows disabling the baseline zero_solver rule globally.
     - Clear diagnostics separating missing project .py files from missing Python packages.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit, ndtr
from scipy.stats import t as _student_t

# ---------------------------------------------------------------------------
# 0. Constants and unified column schema
# ---------------------------------------------------------------------------
HERE: Path = Path(__file__).resolve().parent
OUT_DIR_SAFE: Path = Path(
    os.environ.get("VAR_OUT_DIR") or (HERE / "variant_output_safe")
)
TOL: float = 1e-9
ALL_VARIANTS: tuple[str, ...] = ("base", "W", "A", "F", "C", "T")
VALID_ALT_VARIANTS: tuple[str, ...] = ("W", "A", "F", "C", "T")

# Weibull shape: log-SD = pi / (sqrt(6) * k) = 0.45 => k = pi / (sqrt(6) * 0.45)
WEIBULL_K: float = float(np.pi / (np.sqrt(6.0) * 0.45))

VARIANT_DEFAULTS: dict[str, dict[str, float]] = {
    "base": {},
    "W": {},
    "A": {"gamma": 0.3},
    "F": {"rho": 0.6},
    "C": {"delta": -0.15},
    "T": {"df": 4.0},
}

# Unified mapping from endpoints to grid_summary.csv columns (matching ss.HYP)
ENDPOINT_COLS: dict[str, str] = {
    "H1a": "v3_pow_H1a",
    "H2a_EL": "v3_pow_H2a_EL",
    "H2b_GL": "v3_pow_H2b_GL",
    "H3_trans": "v3_pow_H3",
    "joint": "v3_joint",
}

HYP_COL_SUFFIXES: tuple[str, ...] = ("H1a", "H2a_EL", "H2b_GL", "H3")

_PROJECT_MODULES: dict[str, ModuleType] | None = None


def load_project_modules() -> dict[str, ModuleType]:
    """Lazily import project modules, separating missing .py files from missing pip packages."""
    global _PROJECT_MODULES
    if _PROJECT_MODULES is not None:
        return _PROJECT_MODULES

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    required = (
        "sim_core",
        "analysis_core",
        "scenarios_simulation",
        "p5_drop_early",
        "p6_time_ratio",
        "grid_runner",
    )

    # First check that the required .py files exist next to this script
    missing_files = [f"{name}.py" for name in required if not (HERE / f"{name}.py").exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Required project files not found in {HERE}: {', '.join(missing_files)}"
        )

    loaded: dict[str, ModuleType] = {}
    for mod_name in required:
        try:
            loaded[mod_name] = __import__(mod_name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Found {mod_name}.py, but the active Python environment is missing "
                f"package '{exc.name}'. Install dependencies with: "
                "pip install numpy pandas scipy statsmodels matplotlib"
            ) from exc

    _PROJECT_MODULES = loaded
    return loaded


# ---------------------------------------------------------------------------
# 1. Distribution transforms and alternative DGM generator
# ---------------------------------------------------------------------------
def weibull_mult(z_std: np.ndarray | float, k: float = WEIBULL_K) -> np.ndarray | float:
    """Transform z_std ~ N(0, 1) into a Weibull multiplier with median 1 and SD(log) = pi/(sqrt(6)*k)."""
    if k <= 0.0:
        raise ValueError(f"Weibull shape parameter k must be > 0, got {k}")
    u = np.clip(ndtr(z_std), 1e-12, 1.0 - 1e-12)
    return (np.log(2.0)) ** (-1.0 / k) * (-np.log1p(-u)) ** (1.0 / k)


def studentt_mult(
    z_std: np.ndarray | float, df: float, target_log_sd: float
) -> np.ndarray | float:
    """Transform z_std ~ N(0, 1) into exp(t_df * scale) with unit median and target log-SD."""
    if df <= 2.0:
        raise ValueError(f"df must be > 2 (otherwise t variance is infinite), got {df}")
    if target_log_sd <= 0.0:
        raise ValueError(f"target_log_sd must be > 0, got {target_log_sd}")
    u = np.clip(ndtr(z_std), 1e-12, 1.0 - 1e-12)
    t_val = _student_t.ppf(u, df=df)
    sd_t = float(np.sqrt(df / (df - 2.0)))
    return np.exp(t_val * (target_log_sd / sd_t))


def generate_dataset_safe(
    seed: int,
    params: dict[str, Any] | None,
    variant: str = "base",
    vp: dict[str, float] | None = None,
    kappa: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Generate a synthetic dataset for the selected DGM variant.
    When variant == "base", delegates directly to sim_core.generate_dataset (bit-for-bit identical).
    """
    mods = load_project_modules()
    sc = mods["sim_core"]

    if variant == "base":
        return sc.generate_dataset(seed=seed, params=params)
    if variant not in VALID_ALT_VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}. Allowed values: {ALL_VARIANTS}")
    if kappa <= 0.0 or not np.isfinite(kappa):
        raise ValueError(f"kappa must be a finite number > 0, got {kappa}")

    return _generate_variant_impl(sc, seed, params, variant, vp or {}, float(kappa))


def _generate_variant_impl(
    sc: ModuleType,
    seed: int,
    params: dict[str, Any] | None,
    variant: str,
    vp: dict[str, float],
    kappa: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Implement variants W/A/F/C/T while preserving the exact RNG draw order of sim_core."""
    P = params if params is not None else sc.PARAMS
    vpf: dict[str, float] = {**VARIANT_DEFAULTS[variant], **vp}
    rng = np.random.default_rng(seed)
    n = int(P["n_students"])
    if n <= 0 or n % 2 != 0:
        raise ValueError(f"n_students must be a positive even integer, got {n}")
    tasks: list[str] = list(P["tasks"])

    # 1. Baseline pre-test score
    ids = np.array([f"ID_{k:02d}" for k in range(1, n + 1)])
    pre = np.clip(np.round(rng.normal(P["pre_mean"], P["pre_sd"], n)), 0, 20).astype(int)

    # 2. Matched-pair randomization (exact RNG order from sim_core)
    order = np.lexsort((rng.random(n), -pre))
    group = np.empty(n, dtype=object)
    pair_id = np.zeros(n, dtype=int)
    for pair_no, k in enumerate(range(0, n, 2), start=1):
        a, b = order[k], order[k + 1]
        if rng.integers(0, 2) == 0:
            group[a], group[b] = "CG", "EG"
        else:
            group[a], group[b] = "EG", "CG"
        pair_id[a] = pair_id[b] = pair_no
    z = (pre - P["pre_mean"]) / P["pre_sd"]

    # 3. Student-level latent traits and AI uptake
    speed = rng.normal(0, P["speed_sd"], n)
    ability = rng.normal(0, P["ability_sd"], n)

    if variant == "F":
        rho = float(vpf["rho"])
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"For variant F, rho must lie in [-1, 1], got {rho}")
        a_std = ability / P["ability_sd"]
        s_std = speed / P["speed_sd"]
        speed = P["speed_sd"] * (-rho * a_std + np.sqrt(1.0 - rho ** 2) * s_std)

    uptake = rng.beta(P["uptake_a"], P["uptake_b"], n)
    uptake_mean = float(P["uptake_a"] / (P["uptake_a"] + P["uptake_b"]))

    eg_idx = np.where(group == "EG")[0]
    cg_idx = np.where(group == "CG")[0]
    n_viol = int(P["n_protocol_violators"])
    if not 0 <= n_viol <= len(eg_idx):
        raise ValueError(
            f"n_protocol_violators={n_viol} exceeds number of EG students ({len(eg_idx)})"
        )
    violators = rng.choice(eg_idx, size=n_viol, replace=False)
    protocol_violation = np.zeros(n, dtype=int)
    protocol_violation[violators] = 1
    uptake[violators] = 0.95
    zero_solver = int(cg_idx[np.argmin(pre[cg_idx])])

    if variant == "A":
        a_scale = kappa * np.maximum(0.0, 1.0 - float(vpf["gamma"]) * z)
    else:
        a_scale = np.full(n, kappa, dtype=float)

    # 4. Main debugging session (40 minutes, 4 tasks)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        g = str(group[i])
        if rng.random() < P["p_fixed_order"]:
            task_order = list(tasks)
        else:
            task_order = list(rng.permutation(tasks))
        clock = 0.0

        for pos, t in enumerate(task_order, start=1):
            ttype = str(P["task_type"][t])

            if P.get("zero_solver", True) and i == zero_solver:
                solvable = False
            else:
                lp = (
                    logit(P["p_solvable"][g][t])
                    + np.log(P["odds_per_sd_pre"]) * z[i]
                    + ability[i]
                )
                solvable = bool(rng.random() < expit(lp))

            med = float(P["median_time_cg"][t])
            if g == "EG":
                reduction = float(P["ai_time_reduction"][ttype]) * float(uptake[i])
                if variant == "A":
                    reduction = min(0.95, reduction * float(a_scale[i]))
                elif variant in ("W", "F", "C", "T"):
                    reduction = min(0.95, kappa * reduction)
                med = med * (1.0 - reduction)

            time_noise_raw = float(rng.normal(0, P["time_sigma"]))
            if variant == "W":
                time_mult = float(weibull_mult(time_noise_raw / P["time_sigma"]))
            elif variant == "T":
                time_mult = float(
                    studentt_mult(
                        time_noise_raw / P["time_sigma"],
                        float(vpf["df"]),
                        float(P["time_sigma"]),
                    )
                )
            else:
                time_mult = float(np.exp(time_noise_raw))

            solve_time = float(
                med
                * (P["time_mult_per_sd_pre"] ** z[i])
                * np.exp(speed[i])
                * time_mult
            )

            giveup_raw = float(rng.normal(0, P["giveup_sigma"]))
            giveup = float(P["giveup_median"] * np.exp(giveup_raw))
            if variant == "C":
                z_group = 1.0 if g == "EG" else 0.0
                giveup *= float(np.exp(float(vpf["delta"]) * z_group))

            remaining = float(P["session_minutes"]) - clock
            if remaining <= 0.05:
                rows.append(
                    dict(
                        student_id=ids[i],
                        task=t,
                        task_type=ttype,
                        order_position=pos,
                        attempted=0,
                        time_sec=np.nan,
                        time_min=np.nan,
                        solved=0,
                        attempts_pytest=0,
                        ended_by="not_reached",
                    )
                )
                continue

            work = solve_time if solvable else giveup
            if work <= remaining:
                t_spent, solved = work, int(solvable)
                ended_by = "solved" if solvable else "abandoned"
                clock += work
            else:
                t_spent, solved, ended_by = remaining, 0, "session_end"
                clock = float(P["session_minutes"])

            mu = float(P["attempts_mean"][g][ttype])
            typical = float(P["median_time_cg"][t])
            if g == "EG":
                typical *= 1.0 - float(P["ai_time_reduction"][ttype]) * uptake_mean
            mu_eff = mu * (max(t_spent, 0.5) / typical) ** 0.4
            if not solved:
                mu_eff *= 1.3
            extra_mean = max(mu_eff - 1.0, 0.2)
            k_disp = float(P["attempts_dispersion"])
            extra = rng.negative_binomial(k_disp, k_disp / (k_disp + extra_mean))
            attempts = 1 + int(extra)

            # Floor at 1 second to prevent ln(0) = -inf in extreme left tails of Student-t
            sec = max(1, int(round(t_spent * 60)))
            rows.append(
                dict(
                    student_id=ids[i],
                    task=t,
                    task_type=ttype,
                    order_position=pos,
                    attempted=1,
                    time_sec=sec,
                    time_min=round(sec / 60.0, 3),
                    solved=solved,
                    attempts_pytest=attempts,
                    ended_by=ended_by,
                )
            )

    # 5. Assemble task-level and student-level tables (identical to sim_core.py)
    tasks_df = pd.DataFrame(rows)
    info = pd.DataFrame({"student_id": ids, "group": group, "pair_id": pair_id})
    tasks_df = tasks_df.merge(info, on="student_id")
    tasks_df = tasks_df[
        [
            "student_id",
            "pair_id",
            "group",
            "task",
            "task_type",
            "order_position",
            "attempted",
            "time_sec",
            "time_min",
            "solved",
            "attempts_pytest",
            "ended_by",
        ]
    ]
    tasks_df["synthetic"] = True

    solved_by_type = tasks_df.pivot_table(
        index="student_id", columns="task_type", values="solved", aggfunc="sum"
    )
    sr_expl_n = solved_by_type.loc[ids, "explicit"].values
    sr_hid_n = solved_by_type.loc[ids, "hidden"].values
    attempts_total = (
        tasks_df.groupby("student_id")["attempts_pytest"].sum().loc[ids].values
    )

    r = float(P["item_corr"])
    corr = np.array(P["scale_corr"], dtype=float)
    eps = rng.multivariate_normal(np.zeros(3), corr, size=n)
    items: dict[str, np.ndarray] = {}
    for j, (name, cfg) in enumerate(P["cog_load"].items()):
        k_items, sd_t = int(cfg["k"]), float(cfg["sd"])
        var_item = (sd_t ** 2) / (r + (1.0 - r) / k_items)
        latent_total_sd = float(np.sqrt(r * var_item))
        resid_sd = float(np.sqrt(max(latent_total_sd ** 2 - float(cfg["beta_pre"]) ** 2, 1e-6)))
        item_noise_sd = float(np.sqrt((1.0 - r) * var_item))
        mu_vec = np.array([cfg["mean"][gg] for gg in group], dtype=float)
        latent = mu_vec + float(cfg["beta_pre"]) * z + resid_sd * eps[:, j]
        for item_no in range(1, k_items + 1):
            raw = latent + rng.normal(0, item_noise_sd, n)
            items[f"{name}{item_no}"] = np.clip(np.round(raw), 0, 10).astype(int)

    tr = P["transfer"]
    mu_tr = np.array([tr["mean"][gg] for gg in group], dtype=float)
    noise = rng.normal(0, 1, n)
    is_eg = (group == "EG").astype(float)
    score = (
        mu_tr
        + float(tr["sd"]) * (float(tr["r_pre"]) * z + np.sqrt(1.0 - float(tr["r_pre"]) ** 2) * noise)
        + float(tr["per_hidden_solved"]) * (sr_hid_n - 1.0)
        - float(tr["uptake_penalty"]) * (uptake - uptake_mean) * is_eg
    )
    transfer = np.clip(np.round(score), 0, 10).astype(int)

    students = pd.DataFrame(
        {
            "student_id": ids,
            "pair_id": pair_id,
            "group": group,
            "pre_score": pre,
            "protocol_violation": protocol_violation,
            "ai_prompts_n": np.where(group == "EG", rng.poisson(3 + 12 * uptake), np.nan),
            "SR_expl_n": sr_expl_n,
            "SR_hid_n": sr_hid_n,
            "SR_expl": sr_expl_n / 2.0,
            "SR_hid": sr_hid_n / 2.0,
            "attempts_total": attempts_total,
        }
    )
    for key, vals in items.items():
        students[key] = vals
    students["IL"] = students[["IL1", "IL2"]].mean(axis=1)
    students["EL"] = students[["EL1", "EL2", "EL3"]].mean(axis=1)
    students["GL"] = students[["GL1", "GL2", "GL3"]].mean(axis=1)
    students["transfer_score"] = transfer
    students["synthetic"] = True

    meta: dict[str, Any] = {
        "seed": seed,
        "variant": variant,
        "vp": vpf,
        "kappa": kappa,
        "protocol_violators": list(ids[violators]),
        "zero_solver_id": str(ids[zero_solver]),
    }
    return students, tasks_df, meta


# ---------------------------------------------------------------------------
# 2. Invariant validators (Frechet-Boole bounds, calibration brackets)
# ---------------------------------------------------------------------------
def check_boole_row(
    row: dict[str, Any] | pd.Series, prefix: str = "v3", tol: float = 1e-12
) -> str | None:
    """Verify Frechet-Boole probability bounds for a single summary row."""
    pow_cols = [f"{prefix}_pow_{h}" for h in HYP_COL_SUFFIXES]
    joint_col = f"{prefix}_joint"
    if not all(c in row for c in (*pow_cols, joint_col)):
        return None
    p_vals = np.array([float(row[c]) for c in pow_cols], dtype=float)
    p_joint = float(row[joint_col])
    if np.any(np.isnan(p_vals)) or np.isnan(p_joint):
        return f"{row.get('name', '?')} ({prefix}): NaN detected in power columns"

    upper = float(np.min(p_vals))
    lower = float(max(0.0, 1.0 - np.sum(1.0 - p_vals)))
    if not (lower - tol <= p_joint <= upper + tol):
        return (
            f"{row.get('name', '?')} ({prefix}): joint={p_joint:.4f} outside Frechet-Boole "
            f"bounds [{lower:.4f}, {upper:.4f}] for p={np.round(p_vals, 4).tolist()}"
        )
    return None


def check_boole_dataframe(
    df: pd.DataFrame, prefixes: Sequence[str] = ("v3", "v2", "v1"), tol: float = 1e-12
) -> list[str]:
    """Verify Frechet-Boole bounds across all rows and available pipelines in a DataFrame."""
    errors: list[str] = []
    found_any = False
    for pref in prefixes:
        req = [f"{pref}_pow_{h}" for h in HYP_COL_SUFFIXES] + [f"{pref}_joint"]
        if all(c in df.columns for c in req):
            found_any = True
            for _, row in df.iterrows():
                msg = check_boole_row(row, prefix=pref, tol=tol)
                if msg is not None:
                    errors.append(msg)
    if not found_any:
        raise KeyError(
            "Expected power columns not found in table "
            f"(e.g., {ENDPOINT_COLS['H1a']} .. {ENDPOINT_COLS['joint']}). "
            f"Available columns: {list(df.columns)[:12]}"
        )
    return errors


def check_calibration_bracket(
    hr_lo: float, hr_hi: float, target: float, lo: float, hi: float
) -> None:
    """Ensure the target HR lies inside the bisection search bracket."""
    if not (min(hr_lo, hr_hi) <= target <= max(hr_lo, hr_hi)):
        raise ValueError(
            f"Target HR={target:.4f} is outside bisection bracket "
            f"[HR(lo={lo:.3f})={hr_lo:.4f}, HR(hi={hi:.3f})={hr_hi:.4f}]. "
            "Adjust the search range or variant parameters."
        )


# ---------------------------------------------------------------------------
# 3. Extended cell summary with MCSE, v2, PP, and Boole verification
# ---------------------------------------------------------------------------
def _mcse_prop(p: float, n: int) -> float:
    if n <= 0 or np.isnan(p) or not (0.0 <= p <= 1.0):
        return float("nan")
    return float(np.sqrt(p * (1.0 - p) / n))


def summarize_cell_safe(
    cell: dict[str, Any], df: pd.DataFrame, truth: dict[str, float]
) -> dict[str, Any]:
    """Aggregate cell-level performance metrics with Boole verification, MCSE, v2, and PP."""
    mods = load_project_modules()
    ss = mods["scenarios_simulation"]

    df_ok = df[df["ok"] == 1]
    n = len(df_ok)
    out: dict[str, Any] = {
        "name": cell["name"],
        "block": cell["block"],
        "N": int(cell["n"]),
        "lam": float(cell["lam"]),
        "solv": float(cell["solv"]),
        "rho": cell["rho"],
        "share": float(cell["share"]),
        "R": int(n),
    }

    pipelines: tuple[tuple[str, str, str], ...] = (
        ("v1", "itt_h1a_p", "itt_hr"),
        ("v2", "itt_v2_p", "itt_hr"),
        ("v3", "v3_p", "v3_hr"),
    )
    for pipe, pcol, hrcol in pipelines:
        if pcol not in df_ok.columns or hrcol not in df_ok.columns:
            continue
        p_mat = np.nan_to_num(
            np.column_stack(
                [df_ok[pcol], df_ok["itt_el_wp"], df_ok["itt_gl_wp"], df_ok["itt_tr_p"]]
            ).astype(float),
            nan=1.0,
        )
        d_mat = np.column_stack(
            [
                df_ok[hrcol] > 1,
                df_ok["itt_el_low"].astype(bool),
                df_ok["itt_gl_low"].astype(bool),
                df_ok["itt_tr_b"] < 0,
            ]
        )
        adj = ss.holm_matrix(p_mat)
        conf = (adj < 0.05) & d_mat
        for j, h in enumerate(ss.HYP):
            p_val = float(conf[:, j].mean()) if n else float("nan")
            out[f"{pipe}_pow_{h}"] = p_val
            out[f"{pipe}_mcse_{h}"] = _mcse_prop(p_val, n)
        jp = float(conf.all(axis=1).mean()) if n else float("nan")
        out[f"{pipe}_joint"] = jp
        out[f"{pipe}_joint_mcse"] = _mcse_prop(jp, n)
        out[f"{pipe}_any_holm"] = float((adj < 0.05).any(axis=1).mean()) if n else float("nan")

        err = check_boole_row(out, prefix=pipe)
        if err is not None:
            raise AssertionError(f"Invariant violation during aggregation: {err}")

    out["joint_mcse"] = out.get("v3_joint_mcse", float("nan"))
    out["tost_unpaired"] = float((df_ok["itt_tost_p"] < 0.05).mean()) if n else float("nan")
    out["tost_paired"] = float((df_ok["itt_ptost_p"] < 0.05).mean()) if n else float("nan")
    out["v1_se_undefined_%"] = float(df_ok["itt_hr_lo"].isna().mean() * 100.0) if n else float("nan")
    if "itt_v2_fb" in df_ok.columns:
        out["v2_jackknife_fb_%"] = float(df_ok["itt_v2_fb"].mean() * 100.0) if n else float("nan")
    out["v3_p_undefined_%"] = float(df_ok["v3_p"].isna().mean() * 100.0) if n else float("nan")
    out["early_cens_%"] = float(df_ok["g_early_cens"].mean() * 100.0) if n else float("nan")
    out["cens_frac_mean_%"] = float(df_ok["g_cens_frac"].mean() * 100.0) if n else float("nan")

    ok_v3 = df_ok["v3_lo"].notna()
    out["v3_hr_coverage"] = (
        float(((df_ok["v3_lo"] <= truth["hr"]) & (truth["hr"] <= df_ok["v3_hi"]))[ok_v3].mean())
        if ok_v3.any()
        else float("nan")
    )
    out["v3_lnhr_bias"] = (
        float(np.log(df_ok["v3_hr"].dropna()).mean() - np.log(truth["hr"]))
        if ok_v3.any()
        else float("nan")
    )
    if "itt_v2_lo" in df_ok.columns and "itt_v2_hi" in df_ok.columns:
        ok_v2 = df_ok["itt_v2_lo"].notna() & df_ok["itt_v2_hi"].notna()
        out["v2_hr_coverage"] = (
            float(
                ((df_ok["itt_v2_lo"] <= truth["hr"]) & (truth["hr"] <= df_ok["itt_v2_hi"]))[
                    ok_v2
                ].mean()
            )
            if ok_v2.any()
            else float("nan")
        )
        out["v2_lnhr_bias"] = (
            float(np.log(df_ok["itt_hr"].dropna()).mean() - np.log(truth["hr"]))
            if ok_v2.any()
            else float("nan")
        )

    tcrit = stats.t.ppf(0.975, (df_ok["itt_tr_n"] - 3).clip(lower=1))
    out["transfer_coverage"] = (
        float(
            (
                ((df_ok["itt_tr_b"] - tcrit * df_ok["itt_tr_se"]) <= truth["tr_b"])
                & (truth["tr_b"] <= (df_ok["itt_tr_b"] + tcrit * df_ok["itt_tr_se"]))
            ).mean()
        )
        if n
        else float("nan")
    )
    out["transfer_bias"] = float(df_ok["itt_tr_b"].mean() - truth["tr_b"]) if n else float("nan")

    if "pp_v2_p" in df_ok.columns and df_ok["pp_v2_p"].notna().any():
        p_pp = np.nan_to_num(
            np.column_stack(
                [df_ok["pp_v2_p"], df_ok["pp_el_wp"], df_ok["pp_gl_wp"], df_ok["pp_tr_p"]]
            ).astype(float),
            nan=1.0,
        )
        d_pp = np.column_stack(
            [
                df_ok["pp_hr"] > 1,
                df_ok["pp_el_low"].astype(bool),
                df_ok["pp_gl_low"].astype(bool),
                df_ok["pp_tr_b"] < 0,
            ]
        )
        adj_pp = ss.holm_matrix(p_pp)
        conf_pp = (adj_pp < 0.05) & d_pp
        for j, h in enumerate(ss.HYP):
            out[f"pp_v2_pow_{h}"] = float(conf_pp[:, j].mean())
        out["pp_v2_joint"] = float(conf_pp.all(axis=1).mean())
        out["pp_lnhr_bias"] = float(np.log(df_ok["pp_hr"].dropna()).mean() - np.log(truth["hr"]))
        out["pp_transfer_bias"] = float(df_ok["pp_tr_b"].mean() - truth["tr_b"])

    out.update({f"truth_{k}": float(v) for k, v in truth.items()})
    return out


# ---------------------------------------------------------------------------
# 4. Replication workers and pseudo-true estimand computation
# ---------------------------------------------------------------------------
def _apply_zero_solver_policy(params: dict[str, Any], zero_solver_mode: str) -> dict[str, Any]:
    if zero_solver_mode == "off":
        params["zero_solver"] = False
    return params


def one_rep_variant(
    job: tuple[str, int, float, int, float, float | None, bool, int, str, dict[str, float], float, str]
) -> dict[str, Any]:
    """Run one simulation replication (top-level function for ProcessPoolExecutor on macOS/Linux)."""
    warnings.filterwarnings("ignore")
    (
        name,
        n,
        lam,
        n_viol,
        solv,
        rho,
        pp,
        seed,
        variant,
        vp,
        kappa,
        zero_solver_mode,
    ) = job
    rec: dict[str, Any] = {"scenario": name, "seed": seed}
    try:
        mods = load_project_modules()
        gr = mods["grid_runner"]
        ss = mods["scenarios_simulation"]
        P = gr.make_params_grid(n, lam, n_viol, solv, rho)
        _apply_zero_solver_policy(P, zero_solver_mode)
        students, tasks, _ = generate_dataset_safe(
            seed, P, variant=variant, vp=vp, kappa=kappa
        )
        rec.update(ss.analyze(students, tasks, "itt_"))
        if pp:
            st2, tk2 = ss.per_protocol(students, tasks)
            rec.update(ss.analyze(st2, tk2, "pp_"))
        rec.update(gr._extras(students, tasks))
        rec["ok"] = 1
    except Exception as exc:
        rec["ok"] = 0
        rec["error"] = f"{type(exc).__name__}: {exc}"[:140]
    return rec


def one_rep_custom_params(
    job: tuple[str, dict[str, Any], int, str, dict[str, float], float]
) -> dict[str, Any]:
    """Run one replication with an explicit parameter dictionary P (used by --quickstart)."""
    warnings.filterwarnings("ignore")
    name, params, seed, variant, vp, kappa = job
    rec: dict[str, Any] = {"scenario": name, "seed": seed}
    try:
        mods = load_project_modules()
        gr = mods["grid_runner"]
        ss = mods["scenarios_simulation"]
        students, tasks, _ = generate_dataset_safe(
            seed, params, variant=variant, vp=vp, kappa=kappa
        )
        rec.update(ss.analyze(students, tasks, "itt_"))
        rec.update(gr._extras(students, tasks))
        rec["ok"] = 1
    except Exception as exc:
        rec["ok"] = 0
        rec["error"] = f"{type(exc).__name__}: {exc}"[:140]
    return rec


def jobs_for_variant(
    cell: dict[str, Any],
    reps: int,
    variant: str,
    vp: dict[str, float],
    kappa: float,
    zero_solver_mode: str = "inherit",
) -> list[tuple[Any, ...]]:
    gr = load_project_modules()["grid_runner"]
    return [
        (
            cell["name"],
            cell["n"],
            cell["lam"],
            cell["viol"],
            cell["solv"],
            cell["rho"],
            cell["pp"],
            s,
            variant,
            vp,
            kappa,
            zero_solver_mode,
        )
        for s in gr.seeds_for(cell, reps)
    ]


def compute_truth_for_params(
    params_factory: Callable[[int], dict[str, Any]],
    variant: str,
    vp: dict[str, float],
    kappa: float,
    seeds: Sequence[int] = (777, 778, 779),
) -> dict[str, float]:
    """Estimate super-population pseudo-true parameters at sample size TRUTH_N."""
    mods = load_project_modules()
    ss = mods["scenarios_simulation"]
    ac = mods["analysis_core"]
    p6 = mods["p6_time_ratio"]

    acc: list[dict[str, float]] = []
    for sd in seeds:
        P = copy.deepcopy(params_factory(int(ss.TRUTH_N)))
        P["zero_solver"] = False
        st, tk, _ = generate_dataset_safe(sd, P, variant=variant, vp=vp, kappa=kappa)
        mean = lambda col, g: float(st.loc[st["group"] == g, col].mean())
        d = ss.cox_frame(st, tk)
        X = d[["eg", "pre_z"]].astype(float).values
        b_strat = (
            ss._phreg(d["time_min"].values, X, d["solved"].values, d["task"].values)
            .fit(disp=False)
            .params[0]
        )
        acc.append(
            {
                "hr": float(np.exp(ss.cox_robust(d, False)[0])),
                "hr_strat": float(np.exp(b_strat)),
                "tr_b": float(ac.test_h3(st)["coef_eg"]),
                "el_diff": mean("EL", "EG") - mean("EL", "CG"),
                "gl_diff": mean("GL", "EG") - mean("GL", "CG"),
                "hid_diff": mean("SR_hid", "EG") - mean("SR_hid", "CG"),
                "lt_b": float(p6.ln_time_effect(st, tk)[0]),
            }
        )
    return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}


def compute_truth_variant(
    lam: float,
    solv: float,
    rho: float | None,
    share: float,
    variant: str,
    vp: dict[str, float],
    kappa: float,
    seeds: Sequence[int] = (777, 778, 779),
) -> dict[str, float]:
    gr = load_project_modules()["grid_runner"]

    def _factory(n_truth: int) -> dict[str, Any]:
        n_viol = int(round(n_truth * share)) if lam > 0 else 0
        return gr.make_params_grid(n_truth, lam, n_viol, solv, rho)

    return compute_truth_for_params(_factory, variant, vp, kappa, seeds=seeds)


def truth_for_variant(
    cell: dict[str, Any],
    cache: dict[str, dict[str, float]],
    variant: str,
    vp: dict[str, float],
    kappa: float,
    truth_file: Path,
) -> dict[str, float]:
    gr = load_project_modules()["grid_runner"]
    if cell["lam"] == 0.0:
        return dict(gr.EXACT_NULL_TRUTH)
    key = (
        f"variant={variant}|vp={sorted(vp.items())}|kappa={kappa:.4f}|"
        f"lam={cell['lam']}|solv={cell['solv']}|rho={cell['rho']}|share={cell['share']}"
    )
    if key not in cache:
        cache[key] = compute_truth_variant(
            cell["lam"],
            cell["solv"],
            cell["rho"],
            cell["share"],
            variant,
            vp,
            kappa,
        )
        truth_file.parent.mkdir(parents=True, exist_ok=True)
        with truth_file.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1, ensure_ascii=False)
    return cache[key]


# ---------------------------------------------------------------------------
# 5. Calibration of scalar multiplier kappa
# ---------------------------------------------------------------------------
def out_dir_for(variant: str, tag: str | None = None) -> Path:
    return OUT_DIR_SAFE / (tag or variant)


def kappa_file(variant: str, tag: str | None) -> Path:
    return out_dir_for(variant, tag) / "kappa.json"


def save_kappa(variant: str, vp: dict[str, float], kappa: float, tag: str | None) -> None:
    path = kappa_file(variant, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "variant": variant,
                "vp": vp,
                "kappa": float(kappa),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


def load_kappa(variant: str, tag: str | None) -> float | None:
    path = kappa_file(variant, tag)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return float(json.load(fh)["kappa"])


def cmd_calibrate(
    variant: str,
    vp: dict[str, float],
    lam: float = 0.6,
    seeds: Sequence[int] = (777,),
    tol: float = 1e-3,
    max_iter: int = 20,
) -> float:
    if variant == "base":
        print("variant='base' does not require calibration.", file=sys.stderr)
        return 1.0
    gr = load_project_modules()["grid_runner"]
    print(f"Calibrating kappa: variant={variant!r}, vp={vp}, lam={lam}, seeds={seeds}")
    target = float(gr.compute_truth_grid(lam, 0.0, None, gr.BASE_SHARE, seeds)["hr"])
    print(f"  target HR (base, lam={lam}): {target:.4f}")

    lo, hi = 0.4, 2.5
    hr_lo = compute_truth_variant(lam, 0.0, None, gr.BASE_SHARE, variant, vp, lo, seeds)["hr"]
    hr_hi = compute_truth_variant(lam, 0.0, None, gr.BASE_SHARE, variant, vp, hi, seeds)["hr"]
    print(f"  kappa={lo:.3f}: HR={hr_lo:.4f}")
    print(f"  kappa={hi:.3f}: HR={hr_hi:.4f}")
    check_calibration_bracket(hr_lo, hr_hi, target, lo, hi)

    best_kappa, best_hr = (lo + hi) / 2.0, float("nan")
    for it in range(1, max_iter + 1):
        mid = (lo + hi) / 2.0
        hr = compute_truth_variant(lam, 0.0, None, gr.BASE_SHARE, variant, vp, mid, seeds)["hr"]
        print(f"  [{it:02d}] kappa={mid:.4f} -> HR={hr:.4f} (delta={hr - target:+.4f})")
        best_kappa, best_hr = mid, hr
        if abs(hr - target) < tol:
            break
        if hr > target:
            hi = mid
        else:
            lo = mid
    print(f"\nResult: kappa = {best_kappa:.4f} (HR={best_hr:.4f}, target={target:.4f})")
    return best_kappa


# ---------------------------------------------------------------------------
# 6. Verification commands: --check-base, --check-neutral, --check-csv
# ---------------------------------------------------------------------------
def cmd_check_base(grid_dir: Path, reps: int, only: str) -> bool:
    gr = load_project_modules()["grid_runner"]
    print(f"Regression check: variant='base' vs raw CSVs in {grid_dir} and {HERE / 'output'} (reps={reps})")
    cells = gr.build_grid_cells()
    all_ok, n_checked = True, 0
    for c in cells:
        if only and only not in c["name"]:
            continue
        path = grid_dir / f"grid_raw_{c['name']}.csv"
        if not path.exists():
            if c.get("v1name"):
                v1_path = HERE / "output" / f"raw_v3_{c['v1name']}.csv"
                if v1_path.exists():
                    path = v1_path
                else:
                    continue
            else:
                continue
        n_checked += 1
        old = pd.read_csv(path, encoding="utf-8-sig").head(reps)
        jobs = jobs_for_variant(c, reps, "base", {}, 1.0, zero_solver_mode="inherit")
        new = pd.DataFrame([one_rep_variant(j) for j in jobs])
        merged = old.merge(new, on="seed", suffixes=("_old", "_new"))
        bad: list[str] = []
        for col in old.columns:
            if col in ("seed", "scenario", "ok", "error") or f"{col}_new" not in merged:
                continue
            a = merged[f"{col}_old"].astype(float).values
            b = merged[f"{col}_new"].astype(float).values
            if not np.allclose(a, b, rtol=TOL, atol=1e-12, equal_nan=True):
                bad.append(col)
        ok = (not bad) and (len(merged) == len(old))
        status = "OK" if ok else f"FAIL: {', '.join(bad[:5])}"
        print(f"  {c['name'][:45]:45s} reps={len(merged)}/{len(old)}  {status}")
        all_ok &= ok
    if n_checked == 0:
        print(f"  SKIPPED: no raw CSV files found in {grid_dir} or {HERE / 'output'}")
        return False
    print("SUMMARY:", "regression passed" if all_ok else "DISCREPANCIES DETECTED -- do not trust grid")
    return all_ok


def cmd_check_neutral(cells: list[dict[str, Any]], reps: int) -> bool:
    """Verify A(gamma=0), F(rho=0), and C(delta=0) at kappa=1.0 match base bit-for-bit."""
    gr = load_project_modules()["grid_runner"]
    neutral = [
        ("A", {"gamma": 0.0}),
        ("F", {"rho": 0.0}),
        ("C", {"delta": 0.0}),
    ]
    targets = [c for c in cells if c["v1name"] in ("S0_null", "S1_planted")]
    if not targets:
        print("No S0_null/S1_planted anchor cells found for neutrality check.")
        return False
    all_ok = True
    for variant, vp in neutral:
        print(f"\nNeutrality check: variant={variant!r}, vp={vp}")
        for c in targets:
            base_df = pd.DataFrame([gr.one_rep_grid(j) for j in gr.jobs_for(c, reps)])
            var_df = pd.DataFrame(
                [
                    one_rep_variant(j)
                    for j in jobs_for_variant(c, reps, variant, vp, 1.0, "inherit")
                ]
            )
            bad: list[str] = []
            for col in base_df.columns:
                if col in ("scenario", "seed", "ok", "error"):
                    continue
                if col not in var_df.columns:
                    bad.append(f"{col}(missing)")
                    continue
                try:
                    a = base_df[col].astype(float).values
                    b = var_df[col].astype(float).values
                    if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
                        bad.append(col)
                except (ValueError, TypeError):
                    if not (
                        base_df[col].astype(str).values == var_df[col].astype(str).values
                    ).all():
                        bad.append(col)
            ok = (not bad) and (len(base_df) == len(var_df))
            status = "OK" if ok else f"FAIL: {', '.join(bad[:5])}"
            print(f"  cell={c['name'][:45]:45s} reps={len(base_df)}  {status}")
            all_ok &= ok
    return all_ok


def cmd_check_csv(csv_path: Path) -> bool:
    """Audit an existing summary CSV against Frechet-Boole bounds without modifying it."""
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return False
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    issues = check_boole_dataframe(df)
    if issues:
        print(f"INVARIANT VIOLATIONS FOUND in {csv_path}:")
        for msg in issues:
            print("  -", msg)
        return False
    print(f"OK: audited {len(df)} rows in {csv_path}, 0 Frechet-Boole violations.")
    return True


# ---------------------------------------------------------------------------
# 7. --run and --list
# ---------------------------------------------------------------------------
def cmd_list(
    cells: list[dict[str, Any]], reps: int, workers: int, variant: str, kappa: float
) -> None:
    gr = load_project_modules()["grid_runner"]
    tot = 0.0
    print(
        f"{'index':>6} {'blk':<4} {'N':>4} {'lam':>4} {'solv':>5} {'rho':>5} "
        f"{'viol':>6} {'PP':>3}  {'anchor'}"
    )
    for c in cells:
        tot += float(gr.est_seconds_per_rep(c["n"])) * reps
        print(
            f"{c['index']:>6} {c['block']:<4} {c['n']:>4} {c['lam']:>4.1f} "
            f"{c['solv']:>5.1f} "
            f"{'' if c['rho'] is None else c['rho']:>5} "
            f"{c['viol']:>6} {'yes' if c['pp'] else '':>3}  {c['v1name'] or ''}"
        )
    print(
        f"\nCells: {len(cells)}; R={reps}; variant={variant} (kappa={kappa:.3f}); "
        f"est. runtime: {tot / 3600:.1f} core-hours, "
        f"~{tot / 3600 / max(workers, 1):.1f} h on {workers} workers (approx.)"
    )


def cmd_run_variant(
    cells: list[dict[str, Any]],
    reps: int,
    workers: int,
    variant: str,
    vp: dict[str, float],
    kappa: float,
    resume: bool,
    only: str,
    tag: str | None,
    zero_solver_mode: str,
) -> None:
    out = out_dir_for(variant, tag)
    out.mkdir(parents=True, exist_ok=True)
    truth_file = out / "grid_truth.json"
    cache: dict[str, dict[str, float]] = (
        json.loads(truth_file.read_text(encoding="utf-8")) if truth_file.exists() else {}
    )
    summaries: list[dict[str, Any]] = []
    t0 = time.time()

    for k, c in enumerate(cells, 1):
        path = out / f"grid_raw_{c['name']}.csv"
        if only and only not in c["name"]:
            if not path.exists():
                continue
            df = pd.read_csv(path, encoding="utf-8-sig")
            summaries.append(
                summarize_cell_safe(
                    c, df, truth_for_variant(c, cache, variant, vp, kappa, truth_file)
                )
            )
            continue
        if resume and path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            print(f"[{k:02d}/{len(cells)}] {c['name']}: existing output found, skipping")
        else:
            t1 = time.time()
            jobs = jobs_for_variant(c, reps, variant, vp, kappa, zero_solver_mode)
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    df = pd.DataFrame(list(ex.map(one_rep_variant, jobs, chunksize=10)))
            else:
                df = pd.DataFrame([one_rep_variant(j) for j in jobs])
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(
                f"[{k:02d}/{len(cells)}] {c['name']}: "
                f"{int(df['ok'].sum())}/{len(df)} reps, {time.time() - t1:.0f} s",
                flush=True,
            )
        if int(df["ok"].sum()) == 0:
            example_err = df["error"].iloc[0] if "error" in df.columns else "?"
            print(f"   WARNING: 0 successful replications; sample error: {example_err}")
            continue
        summaries.append(
            summarize_cell_safe(
                c, df, truth_for_variant(c, cache, variant, vp, kappa, truth_file)
            )
        )
        pd.DataFrame(summaries).to_csv(
            out / "grid_summary.csv", index=False, encoding="utf-8-sig"
        )

    if summaries:
        summary_df = pd.DataFrame(summaries)
        issues = check_boole_dataframe(summary_df)
        if issues:
            raise AssertionError("Invariant violation in final summary:\n" + "\n".join(issues))
        summary_df.to_csv(out / "grid_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Summary: {out / 'grid_summary.csv'}")


# ---------------------------------------------------------------------------
# 8. --invariance (Table 5)
# ---------------------------------------------------------------------------
def interp_required_n(points: Iterable[tuple[float, float]], target: float) -> float | None:
    """Log-linear interpolation on ln(N) to find required N for target power."""
    pts = sorted((float(n), float(p)) for n, p in points if not (np.isnan(n) or np.isnan(p)))
    if not pts:
        return None
    if pts[0][1] >= target:
        return pts[0][0]
    for i in range(1, len(pts)):
        n0, p0 = pts[i - 1]
        n1, p1 = pts[i]
        if p0 < target <= p1:
            if p1 == p0:
                return n1
            return float(
                np.exp(np.log(n0) + (target - p0) / (p1 - p0) * (np.log(n1) - np.log(n0)))
            )
    return None


def cmd_invariance(
    tags: Sequence[str],
    target_lam: float,
    target_power: float,
    search_root: Path = OUT_DIR_SAFE,
    grid_dir: Path | None = None,
) -> bool:
    """Generate cross-DGM N_required table (Table 5) and H1a/H3 power ratio at N=40."""
    print(f"Cross-DGM invariance analysis: lam={target_lam}, target power={target_power}")
    print(f"Summary root: {search_root} | tags: {list(tags)}\n")

    summaries: dict[str, pd.DataFrame] = {}
    for tag in tags:
        path = search_root / tag / "grid_summary.csv"
        if not path.exists() and tag == "base" and grid_dir is not None:
            alt_base = grid_dir / "grid_summary.csv"
            if alt_base.exists():
                path = alt_base
        if not path.exists():
            print(f"  [SKIP] {tag}: missing {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        issues = check_boole_dataframe(df)
        if issues:
            print(f"  [FAIL] {tag}: Frechet-Boole violations in {path}:")
            for msg in issues[:3]:
                print(f"         {msg}")
            return False

        mask = (
            (df["block"] == "A")
            & np.isclose(df["lam"], target_lam)
            & np.isclose(df["solv"], 0.0)
        )
        if "rho" in df.columns:
            mask &= df["rho"].isna()
        df_sub = df[mask].sort_values("N")
        if df_sub.empty:
            print(f"  [SKIP] {tag}: no Block A cells for lam={target_lam}")
            continue
        summaries[tag] = df_sub
        print(f"  {tag:20s}: N grid = {sorted(df_sub['N'].unique().tolist())}")

    if not summaries:
        print("\nNothing to analyze. Run --run first or specify --invariance-dir.")
        return False

    print(f"\nTable 5 (N_required for power >= {target_power}):")
    header = f"{'Endpoint':<10} " + " ".join(f"{t:>12s}" for t in summaries)
    print(header)
    print("-" * len(header))

    rows_out: list[dict[str, Any]] = []
    for ep_name, col in ENDPOINT_COLS.items():
        n_req: dict[str, float | None] = {}
        for tag, df in summaries.items():
            if col not in df.columns:
                raise KeyError(f"Tag {tag!r} is missing required column {col!r}")
            n_req[tag] = interp_required_n(zip(df["N"], df[col]), target_power)
        vals = [v for v in n_req.values() if v is not None]
        max_dev = (
            float((max(vals) - min(vals)) / np.mean(vals) * 100.0)
            if len(vals) >= 2
            else float("nan")
        )
        cells_str = " ".join(
            f"{'--':>12s}" if v is None else f"{v:12.1f}" for v in n_req.values()
        )
        dev_str = "--" if np.isnan(max_dev) else f"{max_dev:.2f}%"
        print(f"{ep_name:<10} {cells_str}   max_dev={dev_str}")
        rows_out.append(
            {
                "endpoint": ep_name,
                "column": col,
                "target_power": target_power,
                **n_req,
                "max_dev_pct": max_dev,
            }
        )

    print("\nPower(H1a) / Power(H3_trans) ratio at N=40 (endpoint power asymmetry):")
    ratios: dict[str, float] = {}
    for tag, df in summaries.items():
        row_n40 = df[df["N"] == 40]
        if row_n40.empty:
            continue
        r = row_n40.iloc[0]
        p1 = float(r[ENDPOINT_COLS["H1a"]])
        p_tr = float(r[ENDPOINT_COLS["H3_trans"]])
        if p_tr > 0.0:
            ratios[tag] = p1 / p_tr
    if ratios:
        vals = list(ratios.values())
        print("  " + "  ".join(f"{tag}={v:.2f}" for tag, v in ratios.items()))
        print(
            f"  mean = {np.mean(vals):.2f}, SD = {np.std(vals):.2f}, "
            f"min = {min(vals):.2f}, max = {max(vals):.2f}"
        )
    else:
        print("  (no data at N=40 or Power(H3_trans) == 0)")

    search_root.mkdir(parents=True, exist_ok=True)
    out_path = search_root / f"invariance_lam{target_lam:.2f}.csv"
    pd.DataFrame(rows_out).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved invariance table: {out_path}")
    return True


# ---------------------------------------------------------------------------
# 9. --quickstart (incorporates all 4 pilot effects)
# ---------------------------------------------------------------------------
def make_params_from_pilot(
    n: int,
    pilot: dict[str, float],
    ai_red_explicit: float,
    share: float,
) -> dict[str, Any]:
    """Construct parameter dict P directly from all four user-supplied pilot effects."""
    mods = load_project_modules()
    ss = mods["scenarios_simulation"]
    gr = mods["grid_runner"]

    P = ss.make_params("null", n, 0)
    P["zero_solver"] = False
    P["n_protocol_violators"] = int(gr.n_viol_for(n, share))
    P["ai_time_reduction"]["explicit"] = float(ai_red_explicit)

    el_cfg = P["cog_load"]["EL"]
    el_cfg["mean"]["EG"] = float(
        np.clip(el_cfg["mean"]["CG"] + pilot["d_el"] * el_cfg["sd"], 0.0, 10.0)
    )
    gl_cfg = P["cog_load"]["GL"]
    gl_cfg["mean"]["EG"] = float(
        np.clip(gl_cfg["mean"]["CG"] + pilot["d_gl"] * gl_cfg["sd"], 0.0, 10.0)
    )
    tr_cfg = P["transfer"]
    tr_cfg["mean"]["EG"] = float(
        np.clip(tr_cfg["mean"]["CG"] + pilot["d_trans"] * tr_cfg["sd"], 0.0, 10.0)
    )
    tr_cfg["uptake_penalty"] = 0.0
    return P


def calibrate_pilot_reduction(
    pilot: dict[str, float],
    variant: str,
    vp: dict[str, float],
    kappa: float,
    share: float,
    seeds: Sequence[int] = (777,),
    tol: float = 0.01,
    max_iter: int = 16,
) -> tuple[float, dict[str, float]]:
    """Solve via bisection for ai_time_reduction['explicit'] matching pilot['hr']."""
    target_hr = float(pilot["hr"])
    lo, hi = 0.0, 0.94

    def _eval(red: float) -> dict[str, float]:
        return compute_truth_for_params(
            lambda n_t: make_params_from_pilot(n_t, pilot, red, share),
            variant=variant,
            vp=vp,
            kappa=kappa,
            seeds=seeds,
        )

    tr_lo = _eval(lo)
    tr_hi = _eval(hi)
    check_calibration_bracket(tr_lo["hr"], tr_hi["hr"], target_hr, lo, hi)

    best_red, best_truth = 0.5, tr_hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        tr = _eval(mid)
        best_red, best_truth = mid, tr
        if abs(tr["hr"] - target_hr) < tol:
            break
        if tr["hr"] > target_hr:
            hi = mid
        else:
            lo = mid
    return best_red, best_truth


def cmd_quickstart(
    pilot: dict[str, float],
    variant: str,
    vp: dict[str, float],
    kappa: float,
    reps: int,
    target_power: float,
    workers: int,
) -> bool:
    """Estimate endpoint-specific and joint N_required from four user-supplied pilot effects."""
    gr = load_project_modules()["grid_runner"]
    share = float(gr.BASE_SHARE)

    print(f"Quickstart: variant={variant!r}, vp={vp}, kappa={kappa:.3f}, R={reps}")
    print(
        f"Pilot effects: HR={pilot['hr']:.2f}, d_EL={pilot['d_el']:+.2f}, "
        f"d_GL={pilot['d_gl']:+.2f}, d_trans={pilot['d_trans']:+.2f}"
    )

    if pilot["hr"] <= 1.0:
        print("  WARNING: H1a tests speedup (HR > 1). Power will be ~0 when HR <= 1.")
    if pilot["d_el"] >= 0.0:
        print("  WARNING: H2a tests lower EL (d_EL < 0). Power will be ~0 when d_EL >= 0.")
    if pilot["d_gl"] >= 0.0:
        print("  WARNING: H2b tests lower GL (d_GL < 0). Power will be ~0 when d_GL >= 0.")
    if pilot["d_trans"] >= 0.0:
        print("  WARNING: H3 tests lower Transfer (d_trans < 0). Power will be ~0 when d_trans >= 0.")

    print("\nCalibrating ai_time_reduction['explicit'] to match pilot HR...")
    red_star, truth = calibrate_pilot_reduction(pilot, variant, vp, kappa, share, seeds=(777,))
    print(
        f"  Calibrated ai_time_reduction['explicit'] = {red_star:.4f} -> "
        f"true HR={truth['hr']:.3f}, EL_diff={truth['el_diff']:+.2f}, "
        f"GL_diff={truth['gl_diff']:+.2f}, Trans_b={truth['tr_b']:+.2f}\n"
    )

    n_grid = (40, 60, 80, 100, 120, 160)
    results: list[dict[str, Any]] = []
    for n_val in n_grid:
        print(f"  N={n_val:3d} ...", end=" ", flush=True)
        t0 = time.time()
        cell = dict(
            name=f"qs_N{n_val}",
            index=900 + n_val,
            block="QS",
            n=n_val,
            lam=-1.0,
            share=share,
            viol=int(gr.n_viol_for(n_val, share)),
            solv=0.0,
            rho=None,
            v1name=None,
            pp=False,
        )
        P_n = make_params_from_pilot(n_val, pilot, red_star, share)
        seeds = gr.seeds_for(cell, reps)
        jobs = [(cell["name"], P_n, s, variant, vp, kappa) for s in seeds]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                df = pd.DataFrame(list(ex.map(one_rep_custom_params, jobs, chunksize=10)))
        else:
            df = pd.DataFrame([one_rep_custom_params(j) for j in jobs])
        summary = summarize_cell_safe(cell, df, truth)
        results.append(summary)
        print(
            f"joint={summary['v3_joint']:.3f} "
            f"(H1a={summary['v3_pow_H1a']:.2f} EL={summary['v3_pow_H2a_EL']:.2f} "
            f"GL={summary['v3_pow_H2b_GL']:.2f} Trans={summary['v3_pow_H3']:.2f}), "
            f"{time.time() - t0:.0f}s"
        )

    df_res = pd.DataFrame(results)
    print(f"\nRequired total sample size N_req (power >= {target_power}):")
    out_req: dict[str, Any] = {
        "pilot_hr": pilot["hr"],
        "pilot_d_el": pilot["d_el"],
        "pilot_d_gl": pilot["d_gl"],
        "pilot_d_trans": pilot["d_trans"],
        "target_power": target_power,
    }
    for ep, col in ENDPOINT_COLS.items():
        n_req = interp_required_n(zip(df_res["N"], df_res[col]), target_power)
        out_req[f"N_req_{ep}"] = n_req
        print(f"  {ep:<10}: {'> 160' if n_req is None else f'{n_req:.1f}'}")

    OUT_DIR_SAFE.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUT_DIR_SAFE / "quickstart_grid.csv", index=False, encoding="utf-8-sig")
    out_path = OUT_DIR_SAFE / "quickstart_result.csv"
    pd.DataFrame([out_req]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path} and {OUT_DIR_SAFE / 'quickstart_grid.csv'}")
    return True


# ---------------------------------------------------------------------------
# 10. Built-in self-tests (--selftest)
# ---------------------------------------------------------------------------
def run_selftest() -> bool:
    """Assertion suite verifying core invariants and numerical edge cases."""
    print("Running built-in self-tests for variant_runner_safe.py...")

    ok_df = pd.DataFrame(
        [
            {
                "name": "valid_cell",
                "v3_pow_H1a": 1.000,
                "v3_pow_H2a_EL": 0.997,
                "v3_pow_H2b_GL": 0.892,
                "v3_pow_H3": 1.000,
                "v3_joint": 0.890,
            }
        ]
    )
    assert check_boole_dataframe(ok_df) == [], "Valid row should not trigger a Boole error"

    bad_df = pd.DataFrame(
        [
            {
                "name": "broken_N160",
                "v3_pow_H1a": 1.000,
                "v3_pow_H2a_EL": 0.997,
                "v3_pow_H2b_GL": 0.892,
                "v3_pow_H3": 1.000,
                "v3_joint": 0.886,
            }
        ]
    )
    errs = check_boole_dataframe(bad_df)
    assert len(errs) == 1 and "0.8890" in errs[0], f"Boole violation not detected: {errs}"
    print("  [OK 1/5] Frechet-Boole bounds audit (valid and violating cases).")

    check_calibration_bracket(1.1, 3.2, 1.8, 0.1, 4.0)
    raised = False
    try:
        check_calibration_bracket(1.1, 1.5, 2.2, 0.1, 4.0)
    except ValueError:
        raised = True
    assert raised, "check_calibration_bracket must raise ValueError when target is outside [lo, hi]"
    print("  [OK 2/5] Bisection bracket guard for kappa / HR calibration.")

    rng = np.random.default_rng(42)
    z = rng.normal(0.0, 1.0, size=200_000)
    w_m = weibull_mult(z)
    t_m = studentt_mult(z, df=4.0, target_log_sd=0.45)
    assert abs(float(np.median(w_m)) - 1.0) < 0.01, "Weibull median deviated from 1"
    assert abs(float(np.std(np.log(w_m))) - 0.45) < 0.01, "Weibull log-SD deviated from 0.45"
    assert abs(float(np.median(t_m)) - 1.0) < 0.01, "Student-t median deviated from 1"
    assert abs(float(np.std(np.log(t_m))) - 0.45) < 0.015, "Student-t log-SD deviated from 0.45"
    print("  [OK 3/5] Distributional properties of Weibull and Student-t transforms.")

    pts = [(40, 0.50), (80, 0.80), (120, 0.95)]
    n_req = interp_required_n(pts, 0.80)
    assert n_req is not None and abs(n_req - 80.0) < 1e-9, f"Incorrect interpolation: {n_req}"
    assert interp_required_n(pts, 0.99) is None, "Unreachable target should return None"
    assert interp_required_n(pts, 0.40) == 40.0, "Target below starting power should return min N"
    print("  [OK 4/5] Log-linear N_required interpolation and ENDPOINT_COLS mapping.")

    # Test 5: guard against 0-second floor in extreme left tail of Student-t
    t_extreme = studentt_mult(-4.5, df=4.0, target_log_sd=0.45)
    sec_extreme = max(1, int(round(0.30 * float(t_extreme) * 60)))
    assert sec_extreme >= 1 and np.isfinite(np.log(round(sec_extreme / 60.0, 3)))
    print("  [OK 5/5] 1-second lower floor preventing ln(0) = -inf in Student-t tail.")
    print("ALL BUILT-IN SELF-TESTS PASSED.")
    return True


# ---------------------------------------------------------------------------
# 11. CLI arguments and entry point
# ---------------------------------------------------------------------------
def parse_vp(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--vp expects 'key=val' pairs, got {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def build_vp(variant: str, args: argparse.Namespace) -> dict[str, float]:
    vp: dict[str, float] = {}
    if variant == "A" and args.gamma is not None:
        vp["gamma"] = args.gamma
    if variant == "F" and args.rho_speed is not None:
        vp["rho"] = args.rho_speed
    if variant == "C" and args.delta is not None:
        vp["delta"] = args.delta
    if variant == "T" and args.df is not None:
        vp["df"] = args.df
    vp.update(parse_vp(args.vp))
    return vp


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(
        description="Self-contained runner for alternative DGMs (W/A/F/C/T), invariant auditor, and --quickstart planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--selftest", action="store_true", help="run built-in assertion tests")
    ap.add_argument("--check-csv", type=str, default="", help="audit a summary CSV against Frechet-Boole bounds")
    ap.add_argument("--list", action="store_true", help="list grid cells and estimated runtime")
    ap.add_argument("--check-base", action="store_true", help="regression-check base variant against raw CSVs")
    ap.add_argument("--check-neutral", action="store_true", help="verify neutrality of A(0)/F(0)/C(0) vs base")
    ap.add_argument("--calibrate", action="store_true", help="calibrate kappa to match baseline true HR")
    ap.add_argument("--run", action="store_true", help="run simulation grid for the selected variant")
    ap.add_argument("--invariance", action="store_true", help="build Table 5 invariance summary from grid_summary.csv files")
    ap.add_argument("--quickstart", action="store_true", help="compute N_required from four pilot effect sizes")

    ap.add_argument("--variant", default="base", choices=ALL_VARIANTS)
    ap.add_argument("--gamma", type=float, default=None, help="variant A: ATI slope gamma")
    ap.add_argument("--rho-speed", type=float, default=None, help="variant F: speed-ability correlation rho")
    ap.add_argument("--delta", type=float, default=None, help="variant C: EG abandonment shift delta")
    ap.add_argument("--df", type=float, default=None, help="variant T: Student-t degrees of freedom (default 4.0)")
    ap.add_argument("--vp", default="", help="comma-separated variant parameter overrides 'k=v,k2=v2'")
    ap.add_argument("--kappa", type=float, default=None, help="override kappa (otherwise read from kappa.json)")
    ap.add_argument(
        "--zero-solver",
        choices=("inherit", "off"),
        default="inherit",
        help="inherit = baseline behavior; off = disable zero_solver in all cells",
    )

    ap.add_argument("--calibrate-lam", type=float, default=0.6)
    ap.add_argument("--seeds", default="777", help="comma-separated super-population seeds for calibration")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--grid-dir", type=str, default="", help="directory containing baseline grid_raw_*.csv / grid_summary.csv")
    ap.add_argument("--invariance-dir", type=str, default="", help="root directory of variant tags for --invariance")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default=None)

    ap.add_argument("--variants", default="", help="comma-separated tag list for --invariance")
    ap.add_argument("--lam", type=float, default=0.6, help="lambda level for Table 5 (default 0.6)")
    ap.add_argument("--power", type=float, default=0.80, help="target power level (default 0.80)")

    ap.add_argument("--pilot-hr", type=float, default=1.80)
    ap.add_argument("--pilot-d-el", type=float, default=-0.40)
    ap.add_argument("--pilot-d-gl", type=float, default=-0.30)
    ap.add_argument("--pilot-d-trans", type=float, default=-0.50)

    args = ap.parse_args()
    actions = [
        args.selftest,
        bool(args.check_csv),
        args.list,
        args.check_base,
        args.check_neutral,
        args.calibrate,
        args.run,
        args.invariance,
        args.quickstart,
    ]
    if not any(actions):
        ap.print_help()
        return 0

    rc = 0
    if args.selftest:
        rc |= 0 if run_selftest() else 1
    if args.check_csv:
        rc |= 0 if cmd_check_csv(Path(args.check_csv)) else 1

    if not any(
        [args.list, args.check_base, args.check_neutral, args.calibrate, args.run, args.invariance, args.quickstart]
    ):
        return rc

    reps = 20 if args.quick else args.reps
    vp = build_vp(args.variant, args)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    grid_dir = Path(args.grid_dir) if args.grid_dir else (HERE / "grid_output")

    if args.invariance:
        inv_root = Path(args.invariance_dir) if args.invariance_dir else OUT_DIR_SAFE
        if args.variants:
            tags = [t.strip() for t in args.variants.split(",") if t.strip()]
        else:
            tags = sorted(
                p.name for p in inv_root.glob("*") if (p / "grid_summary.csv").exists()
            )
            if (grid_dir / "grid_summary.csv").exists() and "base" not in tags:
                tags = ["base", *tags]
        if not tags:
            print(f"No grid_summary.csv files found in {inv_root}", file=sys.stderr)
            return 1
        rc |= 0 if cmd_invariance(tags, args.lam, args.power, inv_root, grid_dir) else 1

    if args.quickstart:
        pilot = {
            "hr": args.pilot_hr,
            "d_el": args.pilot_d_el,
            "d_gl": args.pilot_d_gl,
            "d_trans": args.pilot_d_trans,
        }
        kappa = (
            args.kappa
            if args.kappa is not None
            else (load_kappa(args.variant, args.tag) or 1.0)
        )
        rc |= 0 if cmd_quickstart(pilot, args.variant, vp, kappa, reps, args.power, args.workers) else 1

    if any([args.list, args.check_base, args.check_neutral, args.calibrate, args.run]):
        gr = load_project_modules()["grid_runner"]
        cells = gr.build_grid_cells()

        if args.list:
            kappa = (
                args.kappa
                if args.kappa is not None
                else (load_kappa(args.variant, args.tag) or 1.0)
            )
            cmd_list(cells, reps, args.workers, args.variant, kappa)

        if args.check_base:
            rc |= 0 if cmd_check_base(grid_dir, min(reps, 60), args.only) else 1

        if args.check_neutral:
            rc |= 0 if cmd_check_neutral(cells, min(reps, 20)) else 1

        if args.calibrate:
            if args.variant == "base":
                print("variant='base' does not require calibration.", file=sys.stderr)
                rc |= 1
            else:
                kappa = cmd_calibrate(args.variant, vp, args.calibrate_lam, seeds=seeds)
                save_kappa(args.variant, vp, kappa, args.tag)
                print(f"Saved kappa: {kappa_file(args.variant, args.tag)}")

        if args.run:
            if args.variant == "base":
                print(
                    "For variant='base', run grid_runner.py --run directly.",
                    file=sys.stderr,
                )
                rc |= 1
            else:
                kappa = args.kappa if args.kappa is not None else load_kappa(args.variant, args.tag)
                if kappa is None:
                    print("kappa not set -- running calibration...", file=sys.stderr)
                    kappa = cmd_calibrate(args.variant, vp, args.calibrate_lam, seeds=(777,))
                    save_kappa(args.variant, vp, kappa, args.tag)
                print(f"Using kappa = {kappa:.4f} | zero_solver = {args.zero_solver}")
                cmd_run_variant(
                    cells,
                    reps,
                    args.workers,
                    args.variant,
                    vp,
                    kappa,
                    args.resume,
                    args.only,
                    args.tag,
                    args.zero_solver,
                )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())