"""A run you played in the real game (recorded with real.record), as keys one per sim tick, and how it goes in the sim.

    py -m real.recorded_run "real/recordings/2026-10-09 20-22-54" 8.261     # start: the drop-in landing + 0.04 s,
                                                                           # seconds after the first frame
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.world import DT, LEFT, NONE, RIGHT, Game

# A key pressed in the game acts 0.04 s later, and the game takes keys from 0.16 s after the landing
# (real/replay.py: KEY_LEAD, CLOCK_AFTER_LANDING): so tick 0 is 0.12 s after "landing + 0.04 s", minus the 0.04.
FIRST_TICK_AFTER_START = 0.12 - 0.04


def recorded_ticks(folder, start_s: float, shift_s: float = 0.0) -> list[int]:
    """The recording's W / E presses as sim actions, one per tick, from the run's first tick."""
    log = json.loads((Path(folder) / "log.json").read_text())
    t0 = log["frame_times"][0] + start_s + FIRST_TICK_AFTER_START + shift_s
    down = {"W": None, "E": None}
    events = []  # (tick, key, down)
    for t, k, d in log["keys"]:
        events.append((round((t - t0) / DT), k, d))
    n = max(e[0] for e in events) + 50
    ticks, ei, last = [], 0, None
    for i in range(n):
        while ei < len(events) and events[ei][0] <= i:
            _, k, d = events[ei]
            down[k] = d
            if d:
                last = k
            ei += 1
        held = [k for k in ("W", "E") if down[k]]
        key = last if last in held else (held[0] if held else None)
        ticks.append({"W": LEFT, "E": RIGHT, None: NONE}[key])
    return ticks[max(0, 0):]


def play(ticks):
    g = Game("Tower")
    for a in ticks:
        g.step(a)
        if g.finished:
            break
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("start", type=float)
    args = ap.parse_args()
    for shift in (-0.04, -0.02, 0.0, 0.02, 0.04):
        g = play(recorded_ticks(args.folder, args.start, shift))
        prog = 1 - g.dist_to_finish() / g.route_length
        print(f"shift {shift:+.2f} s: {'finished in %.2f s' % g.time_s if g.finished else 'not finished, %.0f%% of the route' % (100 * prog)}"
              f", {g.deaths} deaths")


if __name__ == "__main__":
    main()
