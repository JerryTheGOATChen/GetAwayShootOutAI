"""How well a run survives the real game's timing, tested in the sim: its key presses replayed blind (as
real.replay plays them into the game), with the start off by up to 2 ticks either way and each press a further
tick late at a coin flip. Reports how long each replay stays on the exact run, and how often it still finishes.

"Stays on the run" is measured the way real.trials_report measures it in the real game, where only the screen can
be seen: both runs filmed with the game's camera, and the player's jeans over 100 px from where the exact run has
them, three grabs (1/30 s apart) in a row. The camera follows the player, so this is more lenient than the
distance in the level, which is also reported ("in the level").

    py -m real.robustness runs/03-speed/ppo_tower.pt runs/04-jitter/ppo_tower.pt
    py -m real.robustness runs/pb_Tower.json --trials 200      # your run, for comparison
    py -m real.robustness runs/04-jitter/ppo_tower.pt --seeing    # the agent sees and reacts, keys still late

--seeing is what a live agent would face: the agent plays with its keys landing late as above (0-2 ticks for the
whole run, plus a tick at a coin flip, as in training batch 04-jitter), but it sees where it is and can react.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from sim.world import DT, Game

PX = 112.4  # screen (world) px per Box2D unit
OFF_BY = 100  # px from the exact run that counts as gone its own way
START_ERROR = 2  # ticks either way: the start is spotted to within about 2 frames
EXTRA = 0.5  # chance of each press landing one tick later still (60 Hz key reads, 50 Hz physics)


def run_ticks(path: str) -> list[int]:
    """A run's actions, one per physics tick: a recorded run (.json), or a trained agent's exact run in the sim (or
    its recorded run, <name>_run.json beside it, where the agent itself is gone)."""
    recorded = Path(path).with_name(Path(path).stem + "_run.json")
    if path.endswith(".pt") and not Path(path).exists() and recorded.exists():
        path = str(recorded)
    if path.endswith(".json"):
        run = json.loads(Path(path).read_text())
        return [a for a in run["actions"] for _ in range(run.get("action_repeat", 1))]
    from common.policy import load_agent
    from common.workers import agent_env
    from sim.env import GetawayEnv
    policy, info = load_agent(path)
    env = GetawayEnv(**agent_env(info, stall_limit_s=None))
    obs, _ = env.reset()
    ticks = []
    while True:
        a = policy.act(obs)
        ticks += [a] * env.action_repeat
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            return ticks


def late(ticks: list[int], start: int, rng) -> list[int]:
    """The same presses with each change moved `start` ticks (+ later, - earlier), plus a tick at a coin flip."""
    out, held, at = [], 0, 0
    changes = [(i, a) for i, a in enumerate(ticks) if a != (ticks[i - 1] if i else 0)]
    for i, a in changes:
        lands = max(at, i + start + int(rng.random() < EXTRA), 0)
        out += [held] * (lands - len(out))
        held, at = a, lands
    out += [held] * (len(ticks) + 50 - len(out))  # keep the last key a little past the end
    return out


_game = None


def play(ticks: list[int], angles=False):
    """The player's position every tick (px) and the tick it finished on (-1: didn't), and its body angle."""
    global _game
    if _game is None:
        _game = Game("Tower")
    g = _game
    g.reset()
    path, ang = [], []
    for a in ticks:
        g.step(a)
        p = g.player.position
        path.append((p.x * PX, p.y * PX))
        ang.append(g.player.angle)
        if g.finished:
            break
    return (np.asarray(path), g.finish_tick) + ((np.asarray(ang),) if angles else ())


JEANS = np.array([0.0, 70.0])  # px above the feet, on the body
GRAB_HZ = 30


def screen(path, angles, n) -> np.ndarray:
    """The jeans' screen position at each grab (30 a second) over the first n ticks, filmed with the game camera."""
    from real.trials_report import filmed
    t = np.arange(0, n * DT, 1 / GRAB_HZ)
    rel, cos, sin, _ = filmed(path[:n], angles[:n], t)
    return rel + np.stack([JEANS[0] * cos - JEANS[1] * sin, -(JEANS[0] * sin + JEANS[1] * cos)], 1)


def trial(args) -> tuple[float, float, bool, float]:
    ticks, exact, exact_ang, k = args
    rng = np.random.default_rng(k)
    start = int(rng.integers(-START_ERROR, START_ERROR + 1))
    path, finish, ang = play(late(ticks, start, rng), angles=True)
    n = min(len(path), len(exact))
    off = np.nonzero(np.hypot(*(path[:n] - exact[:n]).T) > OFF_BY)[0]
    in_level = (off[0] if len(off) else n) * DT
    err = np.hypot(*(screen(path, ang, n) - screen(exact, exact_ang, n)).T) > OFF_BY
    run3 = np.nonzero(err[:-2] & err[1:-1] & err[2:])[0]
    on_screen = run3[0] / GRAB_HZ if len(run3) else n * DT
    return on_screen, in_level, finish >= 0, finish * DT


def survival(ticks, trials: int, pool) -> dict:
    exact, finish, ang = play(ticks, angles=True)
    results = pool.map(trial, [(ticks, exact, ang, k) for k in range(trials)])
    stays = np.array([r[0] for r in results])
    level = np.array([r[1] for r in results])
    done = [r[3] for r in results if r[2]]
    return {"exact_time_s": finish * DT if finish >= 0 else None, "stays_median_s": float(np.median(stays)),
            "stays_q25_s": float(np.percentile(stays, 25)), "stays_q75_s": float(np.percentile(stays, 75)),
            "in_level_median_s": float(np.median(level)),
            "whole_way": float(np.mean(stays >= len(exact) * DT - 0.05)), "finishes": len(done) / trials,
            "finish_median_s": float(np.median(done)) if done else None, "stays_s": stays.tolist(),
            "in_level_s": level.tolist()}


def seeing_trial(args) -> tuple[bool, float, int, float]:
    """One run of an agent that sees the sim and reacts, with late keys: (finished, time, deaths, progress)."""
    path, k = args
    from common.policy import load_agent
    from common.workers import agent_env
    from sim.env import GetawayEnv, Jitter
    policy, info = load_agent(path)
    env = GetawayEnv(**agent_env(info, stall_limit_s=None), jitter=Jitter(), seed=k)
    obs, _ = env.reset()
    while True:
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        if terminated or truncated:
            return info["finished"], info["time_s"], info["deaths"], info["progress"]


def seeing(path: str, trials: int, pool) -> dict:
    results = pool.map(seeing_trial, [(path, k) for k in range(trials)])
    times = [r[1] for r in results if r[0]]
    return {"finishes": len(times) / trials, "finish_median_s": float(np.median(times)) if times else None,
            "finish_best_s": min(times, default=None), "deaths_mean": float(np.mean([r[2] for r in results])),
            "progress_median": float(np.median([r[3] for r in results])), "times_s": times}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="trained agents (.pt) or recorded runs (.json)")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", help="save the results here as JSON")
    ap.add_argument("--seeing", action="store_true", help="the agent sees and reacts (keys still late)")
    args = ap.parse_args()
    out = {}
    with mp.get_context("spawn").Pool() as pool:
        for path in args.runs:
            if args.seeing:
                r = out[path] = seeing(path, args.trials, pool)
                print(f"{path}: seeing, with late keys it finishes {r['finishes']:.0%}"
                      + (f" (median {r['finish_median_s']:.2f} s, best {r['finish_best_s']:.2f} s)" if r["finishes"] else "")
                      + f", {r['deaths_mean']:.1f} deaths a run, median progress {r['progress_median']:.0%}", flush=True)
                continue
            r = survival(run_ticks(path), args.trials, pool)
            out[path] = r
            exact = f"{r['exact_time_s']:.2f} s" if r["exact_time_s"] else "no finish"
            print(f"{path}: exact {exact}; blind with timing errors it stays on its run for {r['stays_median_s']:.2f} s "
                  f"(middle half {r['stays_q25_s']:.2f}-{r['stays_q75_s']:.2f}; in the level {r['in_level_median_s']:.2f}), "
                  f"the whole way {r['whole_way']:.0%}, "
                  f"finishes {r['finishes']:.0%}"
                  + (f" (median {r['finish_median_s']:.2f} s)" if r["finish_median_s"] else ""), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
