"""Side by side, real trials against the sim around a moment: body angle, keys, contacts, speed and the jeans on screen.

    py -m real.inspect "real/trials/<date time>" <run name part> <from s> <to s>
"""
import glob
import json
import sys

from real.body import head_and_jeans
from real.trials_report import fit, observed, predict, sim_path, trial_ticks
from sim.world import DT, Game

PX = 112.4


def main():
    folder, part, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    trs = [json.load(open(f)) for f in sorted(glob.glob(f"{folder}/*{part}*.json"))]
    ticks = trial_ticks(trs[0])
    g = Game("Tower")
    rows = {}
    for a in ticks:
        g.step(a)
        v = g.player.linearVelocity
        rows[g.tick] = (a, int(g.grounded), int(g.in_contact), int(g.is_turning), -g.rotation, v.x * PX, v.y * PX,
                        g.player.position.x * PX, g.player.position.y * PX)
    pos, ang = sim_path(ticks)
    d, owns = fit([observed(t) for t in trs], pos, ang)
    t, j, h, a = head_and_jeans(trs[0])
    pr = predict(t, pos, ang, d, owns[0])
    for ti, ai, jj, pp in zip(t, a, j, pr):
        if lo <= ti <= hi:
            r = rows.get(int(ti / DT))
            if r is None:
                continue
            print(f"{ti:5.2f} ang real {ai:4.0f} sim {r[4]:4.0f} | key {r[0]} gnd {r[1]} cont {r[2]} turn {r[3]} "
                  f"v({r[5]:5.0f},{r[6]:5.0f}) pos({r[7]:5.0f},{r[8]:5.0f}) | jeans real {jj.round()} sim {pp.round()}")


if __name__ == "__main__":
    main()
