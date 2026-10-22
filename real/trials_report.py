"""How long each real-game trial (real.trials) stayed on its sim run: the player's jeans on the real screen, against
where the sim says they should be, filmed with the game's camera.

    py -m real.trials_report "real/trials/<date time>"            # prints a table, saves a chart there

The sim run is filmed with the game's camera (each 60 Hz frame it moves 5 x 1/60 of the way to the player). Three
things the sim doesn't know are fitted from each trial's first second, when the real run still follows the sim:
where the jeans sit on the screen when the camera is centred on the player, how far the camera still lagged the
drop-in when the clock started, and (once for each run, from all its trials) where the jeans are on the body. A
trial has left its sim run when the jeans are over OFF_BY px from where the sim says, three grabs in a row.

It also compares the trials of each run with each other the same way ("together"): how repeatable the real game is.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from real.robustness import run_ticks
from sim.world import DT, Game

PX = 112.4
FRAME_HZ, CAM_SPEED = 60, 5.0
FIT_S = 1.0  # fit on the first second
OFF_BY = 100  # px


def trial_ticks(trial) -> list[int]:
    """The keys a trial played, one per tick: saved with it, or (older trials of a trained agent) rebuilt from the
    keys it sent, which were on the agent's 0.04 s grid and up to 20 ms late. Not the agent's run in today's sim:
    after a change to the sim, the agent plays differently there."""
    if "ticks" in trial:
        return trial["ticks"]
    if trial["run"].endswith(".json") and Path(trial["run"]).exists():
        return run_ticks(trial["run"])
    step = 0.04  # a trained agent's keys (or its recorded run, moved since)
    changes = [(math.floor(t / step + 1e-6), a) for t, a in trial["sent"]]
    changes[0] = (0, changes[0][1])
    acts, held, ci = [], 0, 0
    for k in range(int(trial["length_s"] / step) + 1):
        while ci < len(changes) and changes[ci][0] <= k:
            held = changes[ci][1]
            ci += 1
        acts.append(held)
    return [a for a in acts for _ in range(2)]


def sim_path(ticks):
    """The sim path of these keys: the player's feet (px, y up) and body angle, after each tick."""
    g = Game("Tower")
    pos, ang = [], []
    for a in ticks:
        g.step(a)
        pos.append((g.player.position.x * PX, g.player.position.y * PX))
        ang.append(g.player.angle)
        if g.finished:
            break
    return np.asarray(pos), np.asarray(ang)


def filmed(pos, ang, times):
    """For each time: the terms of the predicted screen position that don't depend on the fitted numbers.
    Returns (player - camera) in screen axes, the body's rotation (cos, sin), and the camera's leftover lag factor."""
    n_frames = int(times.max() * FRAME_HZ) + 2
    alpha = CAM_SPEED / FRAME_HZ
    cam = np.empty((n_frames, 2))
    c = pos[0].copy()
    for f in range(n_frames):
        k = min(int(f / FRAME_HZ / DT), len(pos) - 1)
        c = c + (pos[k] - c) * alpha
        cam[f] = c
    f = np.clip((times * FRAME_HZ).astype(int), 0, n_frames - 1)
    k = np.clip((times / DT).astype(int), 0, len(pos) - 1)
    rel = pos[k] - cam[f]
    return np.stack([rel[:, 0], -rel[:, 1]], 1), np.cos(ang[k]), np.sin(ang[k]), (1 - alpha) ** f


def observed(trial) -> tuple[np.ndarray, np.ndarray]:
    """(times, jeans xy) for every grab after the start where the jeans were seen: the patch nearest where they were
    last seen (the biggest one at the start), so something blue flashing up elsewhere on the screen isn't taken
    for them."""
    t, xy, last = [], [], None
    for ts, blobs, *_ in trial["samples"]:
        big = [b for b in blobs if b[2] >= 300]
        if ts >= 0 and big:
            pick = big[0] if last is None else min(big, key=lambda b: math.dist(b[:2], last))
            last = pick[:2]
            t.append(ts)
            xy.append(pick[:2])
    return np.asarray(t), np.asarray(xy, float)


BAR_START_X, BAR_END_X = 701.0, 1219.0  # the progress bar's head icon at 0% and 100% (the bar runs from x 655 to 1265)


def bar_progress(trial) -> tuple[float, float]:
    """How far the real game's own progress bar got in a trial: (furthest, at the end), 0-1. Read from its little
    head icon at the top of the screen (the same skin colour as Steve's head)."""
    xs = [b[0] for ts, jeans, *rest in trial["samples"] if rest and ts > 0.5
          for b in rest[0] if abs(b[1] - 88) < 6 and 2200 < b[2] < 2900 and 655 < b[0] < 1265]
    if not xs:
        return float("nan"), float("nan")
    p = sorted(min(max((x - BAR_START_X) / (BAR_END_X - BAR_START_X), 0.0), 1.0) for x in xs)
    furthest = p[int(0.98 * (len(p) - 1))]  # held for a while, not one stray frame (Steve's own head near the top)
    last = min(max((xs[-1] - BAR_START_X) / (BAR_END_X - BAR_START_X), 0.0), 1.0)
    return furthest, last


def fit(trials, pos, ang):
    """Least squares over all trials of a run: the jeans' place on the body (shared), and for each trial the screen
    point the camera centres on and its leftover lag (vertical) at the start."""
    rows, rhs, n = [], [], len(trials)
    for i, (t, xy) in enumerate(trials):
        m = t <= FIT_S
        rel, cos, sin, lag = filmed(pos, ang, t[m])
        for j in range(m.sum()):
            # screen x = Ax + rel_x + (dx cos - dy sin);  screen y = Ay + rel_y - (dx sin + dy cos) + lag * L
            rx = np.zeros(2 + 3 * n); ry = np.zeros(2 + 3 * n)
            rx[0], rx[1] = cos[j], -sin[j]
            ry[0], ry[1] = -sin[j], -cos[j]
            rx[2 + 3 * i] = 1
            ry[3 + 3 * i] = 1
            ry[4 + 3 * i] = lag[j]
            rows += [rx, ry]
            rhs += [xy[m][j, 0] - rel[j, 0], xy[m][j, 1] - rel[j, 1]]
    sol = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0]
    return sol[:2], [sol[2 + 3 * i: 5 + 3 * i] for i in range(n)]


def predict(t, pos, ang, d, own):
    rel, cos, sin, lag = filmed(pos, ang, t)
    x = own[0] + rel[:, 0] + d[0] * cos - d[1] * sin
    y = own[1] + rel[:, 1] - (d[0] * sin + d[1] * cos) + own[2] * lag
    return np.stack([x, y], 1)


def stays(t, err, length) -> float:
    off = err > OFF_BY
    for i in range(len(off) - 2):
        if off[i] and off[i + 1] and off[i + 2]:
            return float(t[i])
    return float(length)


def together(obs, length) -> list[float]:
    """For each pair of trials of a run: how long their jeans stayed within OFF_BY px of each other."""
    grid = np.arange(0, length, 1 / 30)
    paths = [np.stack([np.interp(grid, t, xy[:, i], right=np.nan) for i in (0, 1)], 1) for t, xy in obs]
    out = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            err = np.hypot(*(paths[i] - paths[j]).T)
            out.append(stays(grid, np.nan_to_num(err, nan=0.0), length))
    return out


def report(folder: Path):
    by_run = defaultdict(list)
    for f in sorted(folder.glob("*.json")):
        if f.name == "report.json":
            continue
        trial = json.loads(f.read_text())
        by_run[trial["run"]].append((f.stem, trial))
    out = {}
    for run, trials in by_run.items():
        pos, ang = sim_path(trial_ticks(trials[0][1]))
        obs = [observed(tr) for _, tr in trials]
        d, owns = fit(obs, pos, ang)
        results = []
        for (name, tr), (t, xy), own in zip(trials, obs, owns):
            err = np.hypot(*(xy - predict(t, pos, ang, d, own)).T)
            early = err[t <= FIT_S]
            furthest, last = bar_progress(tr)
            results.append({"trial": name, "stays_s": stays(t, err, tr["length_s"]), "fit_error_px": float(np.median(early)),
                            "bar_furthest": furthest, "bar_end": last, "t": t.tolist(), "error_px": err.tolist()})
        s = [r["stays_s"] for r in results]
        pairs = together(obs, trials[0][1]["length_s"])
        out[run] = {"jeans_on_body_px": d.tolist(), "stays_s": s, "median_s": float(np.median(s)),
                    "together_s": pairs, "together_median_s": float(np.median(pairs)) if pairs else None,
                    "trials": results}
        print(f"{run}: {len(s)} trials, stayed on the sim run for a median {np.median(s):.2f} s "
              f"(each: {', '.join(f'{x:.2f}' for x in s)}); fit error in the first second "
              f"{np.median([r['fit_error_px'] for r in results]):.0f} px"
              + (f"; trials stayed together for a median {np.median(pairs):.2f} s" if pairs else "")
              + f"; the game's progress bar got to {np.nanmedian([r['bar_furthest'] for r in results]):.0%}")
    (folder / "report.json").write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    report(Path(ap.parse_args().folder))


if __name__ == "__main__":
    main()
