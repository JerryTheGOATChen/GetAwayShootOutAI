"""For probe runs (real/probes/*.json, played into the game with real.trials): the velocity just after the probe's
last jump that best explains the real game's screen path, next to the sim's.

    py -m real.probe_fit "real/trials/<date time>"

The probe's last key change is a release (a jump). The sim's path up to it is used as is (the trials follow it closely
up to there; check with real.trials_report), then the player's velocity right after the jump is set to each candidate
and the sim carries on; the candidate whose filmed path is closest to the real one over the next WINDOW_S wins.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from real.trials_report import fit, observed, predict
from sim.world import DT, Game

PX = 112.4
WINDOW_S = 0.5


def path(acts, at=None, v=None):
    g = Game("Tower")
    pos, ang, before = [], [], None
    for i, a in enumerate(acts):
        if i == at:
            before = (g.player.linearVelocity.x * PX, g.player.linearVelocity.y * PX)
        g.step(a)
        if i == at and v is not None:
            g.player.linearVelocity = (v[0] / PX, v[1] / PX)
        pos.append((g.player.position.x * PX, g.player.position.y * PX))
        ang.append(g.player.angle)
    return np.asarray(pos), np.asarray(ang), before, g


def error(acts, at, v, obs, d, owns, t0):
    pos, ang, _, _ = path(acts, at, v)
    errs = []
    for (t, xy), own in zip(obs, owns):
        m = (t > t0) & (t < t0 + WINDOW_S)
        errs.append(np.median(np.hypot(*(xy[m] - predict(t[m], pos, ang, d, own)).T)))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    folder = Path(ap.parse_args().folder)
    by_probe = defaultdict(list)
    for f in sorted(folder.glob("*.json")):
        if f.name != "report.json":
            tr = json.loads(f.read_text())
            by_probe[tr["run"]].append(tr)
    for probe, trials in sorted(by_probe.items()):
        acts = json.loads(Path(probe).read_text())["actions"]
        at = max(i for i in range(1, len(acts)) if acts[i] != acts[i - 1])  # the last key change: the jump
        obs = [observed(tr) for tr in trials]
        pos, ang, before, g_sim = path(acts, at)
        sim_after = None
        g = Game("Tower")
        for i, a in enumerate(acts[:at + 1]):
            g.step(a)
        sim_after = (g.player.linearVelocity.x * PX, g.player.linearVelocity.y * PX)
        jumped = g.player.linearVelocity.y * PX - before[1] > 300
        d, owns = fit(obs, pos, ang)  # the screen anchors, from the first second
        t0 = (at + 1) * DT
        best = min(((vx, vy) for vx in range(-400, 2201, 100) for vy in range(-200, 1501, 100)),
                   key=lambda v: error(acts, at, v, obs, d, owns, t0))
        fine = min(((best[0] + dx, best[1] + dy) for dx in range(-80, 81, 20) for dy in range(-80, 81, 20)),
                   key=lambda v: error(acts, at, v, obs, d, owns, t0))
        e = error(acts, at, fine, obs, d, owns, t0)
        e_sim = error(acts, at, sim_after, obs, d, owns, t0)
        jump_real = (fine[0] - before[0], fine[1] - before[1])
        angle = math.degrees(math.atan2(jump_real[0], jump_real[1]))
        print(f"{Path(probe).stem}: before the jump ({before[0]:5.0f},{before[1]:5.0f}); sim after ({sim_after[0]:5.0f},"
              f"{sim_after[1]:5.0f}) {'jump' if jumped else 'NO jump'} [fit {e_sim:3.0f} px]; real after ({fine[0]:5.0f},"
              f"{fine[1]:5.0f}) [fit {e:3.0f} px] = a {math.hypot(*jump_real):4.0f} px/s jump {angle:+4.0f} deg from upright")


if __name__ == "__main__":
    main()
