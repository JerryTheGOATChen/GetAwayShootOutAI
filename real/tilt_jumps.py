"""How often a run uses the move the sim gets wrong (real.probe_fit, runs/rejump_probe.png): a jump soon after a lean
that started from a tilt (on landing). In the sim that jump goes along the body's tilt; in the real game it goes
along a lean that ramps up from upright, so it is much more upright for the first half second.

    py -m real.tilt_jumps runs/03-speed/ppo_tower.pt runs/pb_Tower.json ...
"""
import math
import sys

from real.robustness import run_ticks
from sim.world import DT, Game

TILTED = 20.0  # deg: a lean started at a bigger tilt than this counts
SOON = 0.5  # s: jumps this soon after the lean started


def tilt_jumps(path):
    g = Game("Tower")
    jumps, tricks, gained = 0, 0, 0.0
    start_rot = 0.0
    orig_start, orig_jump = g._start_turn, g._jump

    def _start_turn(right):
        nonlocal start_rot
        was = g.is_turning
        orig_start(right)
        if g.is_turning and not was:
            start_rot = g.rotation

    def _jump(right):
        nonlocal jumps, tricks, gained
        if g.is_turning and right == g.turn_right and g.grounded:
            jumps += 1
            if abs(start_rot) > TILTED and g.turning_time < SOON:
                tricks += 1
                ramp = min(0.5 * g.mv.max_turn_speed / g.mv.lerp_turn_time * g.turning_time ** 2, g.mv.max_turn_angle)
                body = abs(g.rotation)
                gained += 1236 * (math.sin(math.radians(body)) - math.sin(math.radians(ramp)))
        orig_jump(right)

    g._start_turn, g._jump = _start_turn, _jump
    for a in run_ticks(path):
        g.step(a)
        if g.finished:
            break
    return jumps, tricks, gained, g.time_s


if __name__ == "__main__":
    for path in sys.argv[1:]:
        jumps, tricks, gained, t = tilt_jumps(path)
        print(f"{path}: {jumps} jumps, {tricks} of them soon after a lean started from a tilt "
              f"(about {gained / max(tricks, 1):.0f} px/s more sideways each than the real game would give); run {t:.2f} s")
