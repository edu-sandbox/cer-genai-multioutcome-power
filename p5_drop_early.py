# -*- coding: utf-8 -*-
"""
p5_drop_early.py — вариант «v3»: убрать раннюю цензуру, затем робастная SE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import scenarios_simulation as ss

Z975 = 1.959963984540054


def h1a_v3(students, tasks):
    """
    Вариант «v3»: убрать наблюдения с ранней цензурой (solved == 0 и time_min < min(time_min[solved == 1])),
    затем фит модели Кокса с робастной SE (кластеризованной по student_id).
    """
    d = ss.cox_frame(students, tasks)
    solved_mask = d["solved"] == 1
    if not solved_mask.any():
        return {"v3_hr": np.nan, "v3_lo": np.nan, "v3_hi": np.nan, "v3_p": np.nan}

    first_time = d.loc[solved_mask, "time_min"].min()
    early_cens = (d["solved"] == 0) & (d["time_min"] < first_time)
    d_clean = d[~early_cens].reset_index(drop=True)

    if not (d_clean["solved"] == 1).any():
        return {"v3_hr": np.nan, "v3_lo": np.nan, "v3_hi": np.nan, "v3_p": np.nan}

    b, se, _ = ss.cox_robust(d_clean, strata=False)
    hr = float(np.exp(b))
    if np.isfinite(se) and se > 0:
        return {
            "v3_hr": hr,
            "v3_lo": float(np.exp(b - Z975 * se)),
            "v3_hi": float(np.exp(b + Z975 * se)),
            "v3_p": float(2 * stats.norm.sf(abs(b / se))),
        }
    return {
        "v3_hr": hr,
        "v3_lo": np.nan,
        "v3_hi": np.nan,
        "v3_p": np.nan,
    }

