# -*- coding: utf-8 -*-
"""
p6_time_ratio.py — ANCOVA по ln-времени (без допущения о пропорц. рисках).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import analysis_core as ac


def ln_time_effect(students, tasks):
    """
    Вторичный анализ: ANCOVA по среднему ln(времени) решённых явных задач с контролем Pre-test.
    Возвращает (b, se, p, m), где:
      b: коэффициент группы (EG - CG)
      se: стандартная ошибка коэффициента
      p: p-значение
      m: число включённых студентов
    """
    s = ac.add_pre_z(students)
    d = tasks[(tasks.task_type == "explicit") & (tasks.attempted == 1)].merge(
        s[["student_id", "pre_z", "eg"]], on="student_id")
    ds = d[d.solved == 1].copy()
    if ds.empty:
        return np.nan, np.nan, np.nan, 0
    ds["ln_time"] = np.log(ds["time_min"])
    agg = ds.groupby("student_id")["ln_time"].mean().rename("ln_T_expl").reset_index()
    sec = s.merge(agg, on="student_id", how="inner")
    if len(sec) < 3 or sec["eg"].nunique() < 2:
        return np.nan, np.nan, np.nan, len(sec)
    res = ac.ancova(sec, "ln_T_expl")
    return float(res["coef_eg"]), float(res["se"]), float(res["p"]), int(res["n"])

