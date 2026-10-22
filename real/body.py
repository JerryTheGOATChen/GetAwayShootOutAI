"""Steve's head and jeans from a trial's recorded patches (real.trials), and the angle of his body from them."""
from __future__ import annotations

import math

import numpy as np

UPRIGHT = (8.0, -150.0)  # head minus jeans, screen px, standing upright (measured at the start of a run)


def head_and_jeans(trial, t_from=-0.5):
    """(times, jeans xy, head xy, body angle in degrees: + = leaning right) for grabs where both were seen."""
    out_t, js, hs = [], [], []
    last = None
    for ts, jeans, *rest in trial["samples"]:
        if ts < t_from or not rest:
            continue
        big = [b for b in jeans if b[2] >= 300]
        heads = [b for b in rest[0] if b[2] >= 1500 and b[1] > 150]  # not the progress bar's little head
        if not big or not heads:
            continue
        j = big[0] if last is None else min(big, key=lambda b: math.dist(b[:2], last))
        h = min(heads, key=lambda b: math.dist(b[:2], j[:2]))
        if math.dist(h[:2], j[:2]) > 220:
            continue
        last = j[:2]
        out_t.append(ts)
        js.append(j[:2])
        hs.append(h[:2])
    t, js, hs = np.asarray(out_t), np.asarray(js, float), np.asarray(hs, float)
    v = hs - js
    ang = np.degrees(np.arctan2(v[:, 0], -v[:, 1]) - math.atan2(UPRIGHT[0], -UPRIGHT[1]))
    return t, js, hs, ang
