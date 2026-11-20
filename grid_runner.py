# -*- coding: utf-8 -*-
"""
grid_runner.py — сетка сценариев ПОВЕРХ проверенного кода v1 (вариант 2: v1 не изменяется).

Идея: v1 (sim_core.py, analysis_core.py, scenarios_simulation.py, p5_drop_early.py, p6_time_ratio.py)
остаётся как есть. Этот файл только
  1) строит параметры генератора для новых ячеек (make_params_grid),
  2) запускает реплики теми же функциями анализа, что в v1,
  3) считает «истинные» значения для каждой ячейки и сводку (мощность, ошибка I рода, сбои, покрытие).

ФАКТОРЫ СЕТКИ
  N      — число студентов (чётное).
  lam    — множитель эффекта λ ∈ [0, 1]: параметры ЭГ линейно между «нулём» (λ=0, ЭГ = КГ) и
           заложенными эффектами v1 (λ=1, это S1/S2). Всё, что отличает ЭГ от КГ, масштабируется вместе.
  solv   — сдвиг (в логитах) решаемости ЯВНЫХ задач T1–T2 у обеих групп: чем он отрицательнее, тем больше
           цензурирования в модели Кокса. Реальная доля цензурирования ИЗМЕРЯЕТСЯ в каждой реплике
           (столбец g_cens_frac), а не предполагается.
  rho    — корреляция шкал EL и GL (в v1 это −0.3).
  share  — число нарушителей регламента как доля от N (v1: 0.05 = N/20; S3: 0.15).

РЕГРЕССИОННЫЙ ТЕСТ  (python grid_runner.py --regress --v1-dir <папка с raw_v3_*.csv>)
  Ячейки λ=0, λ=1 (N=40) и λ=1 (N=60) с seed v1 обязаны воспроизвести CSV v1 побитово. Если нет —
  расширение что-то сломало, и сетке доверять нельзя.

Запуск:
  python grid_runner.py --list                         список ячеек и оценка времени
  python grid_runner.py --regress --v1-dir ПАПКА        регрессионный тест
  python grid_runner.py --run --reps 1000 --workers 8   полный расчёт (можно прерывать; см. --resume)
  python grid_runner.py --run --quick                   быстрая проверка (R=20, несколько ячеек)

Подпись автора: a1d0s_3l1u45
"""
import argparse
import copy
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import sim_core as sc
import scenarios_simulation as ss
from p5_drop_early import h1a_v3            # вариант «v3»: убрать раннюю цензуру, затем робастная SE
from p6_time_ratio import ln_time_effect    # ANCOVA по ln-времени (без допущения о пропорц. рисках)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "grid_output")
TRUTH_FILE = os.path.join(OUT_DIR, "grid_truth.json")
SEED_BASE = 500000          # как в v1: seed = 500000 + индекс_ячейки * 1 000 000 + номер_реплики
INDEX_STEP = 1_000_000
BASE_SHARE = 0.05           # доля нарушителей в v1 (n_viol = N // 20)
EXACT_NULL_TRUTH = {
    "hr": 1.0,
    "hr_strat": 1.0,
    "tr_b": 0.0,
    "el_diff": 0.0,
    "gl_diff": 0.0,
    "hid_diff": 0.0,
    "lt_b": 0.0,
}

# Ячейки, которые ОБЯЗАНЫ совпасть с v1 (их индекс = индекс ячейки v1, поэтому совпадают и seed):
# ключ: (N, lam, share) при solv = 0 и rho = None
V1_ANCHORS = {
    (40, 0.0, BASE_SHARE): (1, "S0_null"),
    (40, 1.0, BASE_SHARE): (2, "S1_planted"),
    (60, 1.0, BASE_SHARE): (3, "S2_N60"),
    (80, 1.0, BASE_SHARE): (4, "S2_N80"),
    (100, 1.0, BASE_SHARE): (5, "S2_N100"),
    (120, 1.0, BASE_SHARE): (6, "S2_N120"),
    (160, 1.0, BASE_SHARE): (7, "S2_N160"),
    (40, 1.0, 0.15): (8, "S3_viol6"),
}


# ----------------------------------------------------------------------------
# 1. Параметры генератора для ячейки сетки
# ----------------------------------------------------------------------------
def _lin(a, b, lam):
    """Линейная интерполяция между «нулём» a (λ=0) и заложенным значением b (λ=1)."""
    return float(a) + lam * (float(b) - float(a))


def _lin_dict(a, b, lam):
    return {k: _lin(a[k], b[k], lam) for k in b}


def make_params_grid(n, lam, n_viol, solv_shift=0.0, rho_el_gl=None):
    """
    Возвращает словарь параметров для sim_core.generate_dataset.
    При lam=1, solv_shift=0, rho_el_gl=None результат ТОЧНО равен ss.make_params("planted", ...),
    при lam=0 — ss.make_params("null", ...). Это нужно для регрессионного теста.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam должен быть в [0, 1], получено {lam}")
    if n % 2 != 0:
        raise ValueError("N должно быть чётным (парная рандомизация)")

    if lam == 0.0:
        P = ss.make_params("null", n, 0)
    else:
        P = ss.make_params("planted", n, n_viol)
        if lam < 1.0:
            N0 = ss.make_params("null", n, 0)   # «нулевые» значения для интерполяции
            P["p_solvable"]["EG"] = _lin_dict(N0["p_solvable"]["EG"], P["p_solvable"]["EG"], lam)
            P["ai_time_reduction"] = _lin_dict(N0["ai_time_reduction"], P["ai_time_reduction"], lam)
            P["attempts_mean"]["EG"] = _lin_dict(N0["attempts_mean"]["EG"], P["attempts_mean"]["EG"], lam)
            for name in P["cog_load"]:
                P["cog_load"][name]["mean"]["EG"] = _lin(N0["cog_load"][name]["mean"]["EG"],
                                                         P["cog_load"][name]["mean"]["EG"], lam)
            P["transfer"]["mean"]["EG"] = _lin(N0["transfer"]["mean"]["EG"], P["transfer"]["mean"]["EG"], lam)
            P["transfer"]["uptake_penalty"] = _lin(N0["transfer"]["uptake_penalty"],
                                                   P["transfer"]["uptake_penalty"], lam)

    if solv_shift != 0.0:   # сдвиг решаемости явных задач у обеих групп -> больше/меньше цензурирования
        for g in ("CG", "EG"):
            for t in ("T1", "T2"):
                P["p_solvable"][g][t] = float(expit(logit(P["p_solvable"][g][t]) + solv_shift))

    if rho_el_gl is not None:   # корреляция шкал EL и GL
        C = copy.deepcopy(P["scale_corr"])
        C[1][2] = C[2][1] = float(rho_el_gl)
        np.linalg.cholesky(np.array(C))   # ошибка, если матрица не положительно определена
        P["scale_corr"] = C
    return P


# ----------------------------------------------------------------------------
# 2. Список ячеек
# ----------------------------------------------------------------------------
def n_viol_for(n, share):
    """v1: N // 20 при share = 0.05; иначе округлённая доля от N."""
    return n // 20 if abs(share - BASE_SHARE) < 1e-12 else int(round(n * share))


def build_grid_cells():
    """56 ячеек: A — ядро (N × λ), B — цензурирование, C — корреляция EL–GL, D — нарушители."""
    cells, counter = [], [100]    # индексы новых ячеек начинаются со 100 (индексы 1–9 заняты v1)

    def add(block, n, lam, share=BASE_SHARE, solv=0.0, rho=None):
        key = (n, lam, share)
        anchor = V1_ANCHORS.get(key) if (solv == 0.0 and rho is None) else None
        if anchor:
            index, v1name = anchor
        else:
            counter[0] += 1
            index, v1name = counter[0], None
        nv = 0 if lam == 0.0 else n_viol_for(n, share)
        name = f"G{index:03d}_{block}_N{n}_L{int(round(lam * 100)):03d}_s{solv:+.0f}_v{int(round(share * 100)):02d}"
        if rho is not None:
            name += f"_r{rho:+.1f}"
        cells.append(dict(name=name, index=index, block=block, n=n, lam=lam, share=share, viol=nv,
                          solv=solv, rho=rho, v1name=v1name,
                          pp=bool(v1name in ("S1_planted", "S3_viol6")) or (block == "D")))

    for lam in (0.0, 0.4, 0.6, 0.8, 1.0):                       # A: ядро
        for n in (40, 60, 80, 100, 120, 160):
            add("A", n, lam)
    for solv in (-1.0, -2.0):                                    # B: цензурирование
        for lam in (0.0, 0.6, 1.0):
            for n in (40, 80, 160):
                add("B", n, lam, solv=solv)
    for rho in (-0.5, 0.0):                                      # C: корреляция EL–GL
        for n in (60, 120):
            add("C", n, 1.0, rho=rho)
    for share in (0.15, 0.25):                                   # D: нарушители регламента
        for n in (40, 80):
            add("D", n, 1.0, share=share)
    return cells


# ----------------------------------------------------------------------------
# 3. Одна реплика
# ----------------------------------------------------------------------------
def _extras(st, tk):
    """Дополнительные столбцы сетки: измеренное цензурирование, v3, ANCOVA по ln-времени."""
    out = {}
    d = ss.cox_frame(st, tk)
    out["g_cens_frac"] = float(1.0 - d["solved"].mean())              # доля цензурированных строк в модели Кокса
    first = d.loc[d.solved == 1, "time_min"].min()
    out["g_early_cens"] = int(((d.solved == 0) & (d.time_min < first)).any())   # есть ранняя цензура
    try:
        out.update(h1a_v3(st, tk))
    except Exception:
        out.update(v3_hr=np.nan, v3_lo=np.nan, v3_hi=np.nan, v3_p=np.nan)
    try:
        b, se, p, m = ln_time_effect(st, tk)
        out.update(g_lt_b=b, g_lt_se=se, g_lt_p=p, g_lt_n=m)
    except Exception:
        out.update(g_lt_b=np.nan, g_lt_se=np.nan, g_lt_p=np.nan, g_lt_n=np.nan)
    return out


def one_rep_grid(job):
    """Одна реплика. job = (name, n, lam, n_viol, solv, rho, pp, seed)."""
    warnings.filterwarnings("ignore")
    name, n, lam, n_viol, solv, rho, pp, seed = job
    rec = {"scenario": name, "seed": seed}
    try:
        P = make_params_grid(n, lam, n_viol, solv, rho)
        students, tasks, _ = sc.generate_dataset(seed, params=P)
        rec.update(ss.analyze(students, tasks, "itt_"))             # ровно тот же анализ, что в v1
        if pp:
            st2, tk2 = ss.per_protocol(students, tasks)
            rec.update(ss.analyze(st2, tk2, "pp_"))
        rec.update(_extras(students, tasks))
        rec["ok"] = 1
    except Exception as e:                  # одна сбойная реплика не должна ронять весь расчёт
        rec["ok"] = 0
        rec["error"] = f"{type(e).__name__}: {e}"[:120]
    return rec


def seeds_for(cell, reps, start=0):
    return [SEED_BASE + cell["index"] * INDEX_STEP + i for i in range(start, start + reps)]


def jobs_for(cell, reps):
    return [(cell["name"], cell["n"], cell["lam"], cell["viol"], cell["solv"], cell["rho"], cell["pp"], s)
            for s in seeds_for(cell, reps)]


# ----------------------------------------------------------------------------
# 4. «Истинные» значения для ячейки (большая когорта) и сводка
# ----------------------------------------------------------------------------
def compute_truth_grid(lam, solv, rho, share, seeds=(777, 778, 779)):
    """Псевдоистинные значения (как ss.compute_truth, но для параметров сетки; среднее по нескольким когортам)."""
    acc = []
    for sd in seeds:
        n_viol = int(round(ss.TRUTH_N * share)) if lam > 0 else 0
        P = make_params_grid(ss.TRUTH_N, lam, n_viol, solv, rho)
        P["zero_solver"] = False
        st, tk, _ = sc.generate_dataset(sd, params=P)
        mean = lambda col, g: float(st.loc[st.group == g, col].mean())
        d = ss.cox_frame(st, tk)
        X = d[["eg", "pre_z"]].astype(float).values
        b_strat = ss._phreg(d["time_min"].values, X, d["solved"].values, d["task"].values).fit(disp=False).params[0]
        import analysis_core as ac
        acc.append({
            "hr": float(ac.test_h1a(st, tk)["hr"]), "hr_strat": float(np.exp(b_strat)),
            "tr_b": float(ac.test_h3(st)["coef_eg"]),
            "el_diff": mean("EL", "EG") - mean("EL", "CG"), "gl_diff": mean("GL", "EG") - mean("GL", "CG"),
            "hid_diff": mean("SR_hid", "EG") - mean("SR_hid", "CG"),
            "lt_b": float(ln_time_effect(st, tk)[0]),
        })
    return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}


def truth_for(cell, cache):
    key = f"lam={cell['lam']}|solv={cell['solv']}|rho={cell['rho']}|share={cell['share']}"
    if key not in cache:
        cache[key] = compute_truth_grid(cell["lam"], cell["solv"], cell["rho"], cell["share"])
        os.makedirs(OUT_DIR, exist_ok=True)
        json.dump(cache, open(TRUTH_FILE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return cache[key]


def summarize_cell(cell, df, truth):
    """Сводка по ячейке: мощность (Holm + направление), FWER, TOST, сбои, измеренное цензурирование, покрытие."""
    from scipy import stats
    df = df[df.ok == 1]
    n = len(df)
    out = {"name": cell["name"], "block": cell["block"], "N": cell["n"], "lam": cell["lam"], "solv": cell["solv"],
           "rho": cell["rho"], "share": cell["share"], "R": n}
    for pipe, pcol, hrcol in (("v1", "itt_h1a_p", "itt_hr"), ("v3", "v3_p", "v3_hr")):
        P = np.nan_to_num(np.column_stack([df[pcol], df.itt_el_wp, df.itt_gl_wp, df.itt_tr_p]).astype(float), nan=1.0)
        D = np.column_stack([df[hrcol] > 1, df.itt_el_low.astype(bool), df.itt_gl_low.astype(bool), df.itt_tr_b < 0])
        adj = ss.holm_matrix(P)
        conf = (adj < 0.05) & D
        for j, h in enumerate(ss.HYP):
            out[f"{pipe}_pow_{h}"] = conf[:, j].mean()
        out[f"{pipe}_joint"] = conf.all(axis=1).mean()
        out[f"{pipe}_any_holm"] = (adj < 0.05).any(axis=1).mean()      # при λ=0 это FWER
    jp = out["v3_joint"]
    out["joint_mcse"] = float(np.sqrt(jp * (1 - jp) / n)) if n else np.nan
    out["tost_unpaired"] = (df.itt_tost_p < 0.05).mean()
    out["tost_paired"] = (df.itt_ptost_p < 0.05).mean()
    out["v1_se_undefined_%"] = df.itt_hr_lo.isna().mean() * 100
    out["v3_p_undefined_%"] = df.v3_p.isna().mean() * 100
    out["early_cens_%"] = df.g_early_cens.mean() * 100
    out["cens_frac_mean_%"] = df.g_cens_frac.mean() * 100            # ИЗМЕРЕННАЯ доля цензурирования
    ok = df.v3_lo.notna()
    out["v3_hr_coverage"] = ((df.v3_lo <= truth["hr"]) & (truth["hr"] <= df.v3_hi))[ok].mean() if ok.any() else np.nan
    out["v3_lnhr_bias"] = float(np.log(df.v3_hr.dropna()).mean() - np.log(truth["hr"])) if ok.any() else np.nan
    tcrit = stats.t.ppf(0.975, (df.itt_tr_n - 3).clip(lower=1))
    out["transfer_coverage"] = (((df.itt_tr_b - tcrit * df.itt_tr_se) <= truth["tr_b"]) &
                                (truth["tr_b"] <= (df.itt_tr_b + tcrit * df.itt_tr_se))).mean()
    out["transfer_bias"] = float(df.itt_tr_b.mean() - truth["tr_b"])
    out.update({f"truth_{k}": v for k, v in truth.items()})
    return out


# ----------------------------------------------------------------------------
# 5. Команды: --list, --regress, --run
# ----------------------------------------------------------------------------
def est_seconds_per_rep(n):
    return 0.19 * (n / 40.0) ** 1.65      # оценка по замерам на одном ядре (N=40: 0.19 с; N=120: 1.16 с)


def cmd_list(cells, reps, workers):
    tot = 0.0
    print(f"{'индекс':>6} {'блок':<4} {'N':>4} {'λ':>4} {'solv':>5} {'rho':>5} {'нарушит.':>8} {'PP':>3}  {'ячейка v1'}")
    for c in cells:
        tot += est_seconds_per_rep(c["n"]) * reps
        print(f"{c['index']:>6} {c['block']:<4} {c['n']:>4} {c['lam']:>4.1f} {c['solv']:>5.1f} "
              f"{'' if c['rho'] is None else c['rho']:>5} {c['viol']:>8} {'да' if c['pp'] else '':>3}  {c['v1name'] or ''}")
    print(f"\nЯчеек: {len(cells)}; R = {reps}; оценка: {tot / 3600:.1f} ядро-часов, "
          f"≈ {tot / 3600 / max(workers, 1):.1f} ч на {workers} ядрах (грубо, ±50%)")


def cmd_regress(v1_dir, reps):
    """Бит-в-бит сверка с CSV v1 для ячеек S0, S1, S2_N60 (+ сверка «истины»)."""
    print("Регрессионный тест: сетка против v1")
    cells = {c["v1name"]: c for c in build_grid_cells() if c["v1name"]}
    all_ok = True
    for v1name in ("S0_null", "S1_planted", "S2_N60"):
        path = os.path.join(v1_dir, f"raw_v3_{v1name}.csv")
        if not os.path.exists(path):
            print(f"  ПРОПУЩЕНО: нет файла {path}")
            all_ok = False
            continue
        old = pd.read_csv(path, encoding="utf-8-sig").head(reps)
        c = cells[v1name]
        new = pd.DataFrame([one_rep_grid(j) for j in jobs_for(c, reps)])
        m = old.merge(new, on="seed", suffixes=("_v1", "_grid"))
        bad = []
        for col in old.columns:
            if col in ("seed", "scenario", "ok", "error") or f"{col}_grid" not in m:
                continue
            a = m[f"{col}_v1"].astype(float).values
            b = m[f"{col}_grid"].astype(float).values
            if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
                bad.append(col)
        status = "ОК" if (not bad and len(m) == len(old)) else f"РАСХОЖДЕНИЕ в столбцах: {bad[:6]}"
        all_ok &= (not bad and len(m) == len(old))
        print(f"  {v1name:12s} реплик сверено: {len(m):3d}  -> {status}")
    t_grid = compute_truth_grid(1.0, 0.0, None, BASE_SHARE, seeds=(777,))
    t_v1 = ss.compute_truth("planted")
    same = all(abs(t_grid[k] - t_v1[k]) < 1e-9 for k in t_v1)
    all_ok &= same
    print(f"  «истина» planted: HR {t_grid['hr']:.4f} (v1 {t_v1['hr']:.4f}), b {t_grid['tr_b']:.4f} -> {'ОК' if same else 'РАСХОЖДЕНИЕ'}")
    print("ИТОГ:", "регрессия пройдена" if all_ok else "РЕГРЕССИЯ НЕ ПРОЙДЕНА — сетке доверять нельзя")
    return all_ok


def cmd_run(cells, reps, workers, resume, only):
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = json.load(open(TRUTH_FILE, encoding="utf-8")) if os.path.exists(TRUTH_FILE) else {}
    summaries, t0 = [], time.time()
    for k, c in enumerate(cells, 1):
        path = os.path.join(OUT_DIR, f"grid_raw_{c['name']}.csv")
        if only and only not in c["name"]:
            if not os.path.exists(path):
                continue                     # ячейка не выбрана и не считалась раньше
            df = pd.read_csv(path)           # не пересчитываем, но включаем в общую сводку
            summaries.append(summarize_cell(c, df, truth_for(c, cache)))
            continue
        if resume and os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[{k:02d}/{len(cells)}] {c['name']}: найден готовый файл, пропуск")
        else:
            t1 = time.time()
            with ProcessPoolExecutor(max_workers=workers) as ex:
                df = pd.DataFrame(list(ex.map(one_rep_grid, jobs_for(c, reps), chunksize=10)))
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[{k:02d}/{len(cells)}] {c['name']}: {int(df.ok.sum())}/{len(df)} реплик, {time.time() - t1:.0f} с", flush=True)
        if df.ok.sum() == 0:
            print("   ВНИМАНИЕ: ни одной успешной реплики; пример ошибки:", df.get("error", pd.Series(["?"])).iloc[0])
            continue
        summaries.append(summarize_cell(c, df, truth_for(c, cache)))
        pd.DataFrame(summaries).to_csv(os.path.join(OUT_DIR, "grid_summary.csv"), index=False, encoding="utf-8-sig")
    if summaries:    # финальная запись: в сводке все ячейки, у которых есть готовый файл
        pd.DataFrame(summaries).to_csv(os.path.join(OUT_DIR, "grid_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин. Сводка: {os.path.join(OUT_DIR, 'grid_summary.csv')}")


def main():
    warnings.filterwarnings("ignore")   # убираем шум предупреждений scipy/statsmodels в главном процессе
    ap = argparse.ArgumentParser(description="Сетка сценариев поверх кода v1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--v1-dir", default=HERE, help="папка с raw_v3_*.csv из v1")
    ap.add_argument("--resume", action="store_true", help="пропускать ячейки с готовым CSV")
    ap.add_argument("--only", default="", help="считать только ячейки, в имени которых есть эта строка")
    ap.add_argument("--quick", action="store_true", help="R = 20 для проверки")
    a = ap.parse_args()
    cells = build_grid_cells()
    reps = 20 if a.quick else a.reps
    if a.list:
        cmd_list(cells, reps, a.workers)
    elif a.regress:
        sys.exit(0 if cmd_regress(a.v1_dir, min(reps, 60) if a.quick else 60) else 1)
    elif a.run:
        cmd_run(cells, reps, a.workers, a.resume, a.only)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
