"""Closed-loop rehearsal of agents in the sim (real.shadow.sim_test), spread over the cores: each agent plays a
"truth" sim standing in for the real game, through a shadow corrected from noisy, late grabs of it ("shadow"), and
seeing the truth itself ("sees", the best it could do). Appends the results to runs/closed_loop_rehearsal.txt.

    py -m real.rehearse runs/07-helicopter/ppo_tower.pt runs/05-shadow/ppo_tower.pt
    py -m real.rehearse runs/08-late/ppo_tower.pt --late 16      # each run stands still up to 16 s first
    py -m real.rehearse runs/05-shadow/ppo_tower.pt --physics 2026-10-20   # the sim batches 01-06 trained in
    py -m real.rehearse runs/07-helicopter/ppo_tower.pt --lift-shift 0.5    # the game's lifts off the sim's schedule

Run n uses seed 2000 + n (the same noise for every agent), with keys expected 2 ticks late, as real.live sends them.
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "closed_loop_rehearsal.txt"


def job(args):
    agent, mode, n, late, shift, physics = args
    from real.shadow import sim_test
    from sim.world import MOVEMENT, MOVEMENT_2026_10_20
    movement = MOVEMENT_2026_10_20 if physics == "2026-10-20" else MOVEMENT  # (the helicopter where the scene puts it)
    return sim_test(agent, 1, mode, seed=2000 + n, shadow_delay=2, late_s=late, lift_shift_s=shift,
                    movement=movement)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="+")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--modes", default="sees,shadow")
    ap.add_argument("--late", type=float, default=0.0, help="stand still up to this long (s, at random) first")
    ap.add_argument("--lift-shift", type=float, default=0.0,
                    help="the stand-in game's lifts run up to this much (s, at random) off the sim's schedule")
    ap.add_argument("--physics", default=None, help="2026-10-20: the sim before the helicopter hovered")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    modes = args.modes.split(",")
    jobs = [(a, m, n, args.late, args.lift_shift, args.physics) for a in args.agents for m in modes
            for n in range(args.runs)]
    with Pool(args.procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    notes = ", ".join(["seeds 2000+, shadow_delay 2"] + ([f"late up to {args.late:g} s"] if args.late else [])
                      + ([f"lifts off by up to {args.lift_shift:g} s"] if args.lift_shift else [])
                      + (["old helicopter"] if args.physics == "2026-10-20" else []))
    lines = []
    for a in args.agents:
        for m in modes:
            rs = [r for (aa, mm, *_), r in zip(jobs, res) if aa == a and mm == m]
            done = [r["time_s"] - r["late_s"] for r in rs if r["finished"]]  # not counting the standing still
            lines.append(f"{a:40s} {m:6s} finished {len(done):3d}/{len(rs)}"
                         + (f"  median {np.median(done):.2f} s" + (" after standing" if args.late else "") if done else "")
                         + f"  deaths {sum(r['deaths'] for r in rs)}  ({notes})")
    print("\n".join(lines))
    with open(OUT, "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
