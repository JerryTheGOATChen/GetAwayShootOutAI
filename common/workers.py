"""Running many copies of the sim across the CPU cores, and the one way both methods score an agent."""
from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from common.policy import Policy
from sim.env import OBS_SIZE, REWARD_PARTS, GetawayEnv, Jitter, Rewards, Wobble
from sim.world import DT

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "runs"  # one folder per batch, plus your personal bests (pb_<Map>.json)

# The batches of the experiment, each trained with both methods into runs/<name>/. py -m common.run_batches trains
# them all, one after another. Add a new one at the end to try something else (and record it in EXPERIMENTS.md).
# practice  half the episodes the methods learn from start at a random moment of your best run (runs/pb_<Map>.json,
# copied into the batch's folder as your_run.json when it starts, and replayed exactly) instead of at
# the start, so the hard parts further along get practised before an agent can reach them on its own.

# PPO: each practice episode, at a coin flip. 
# Genetic: each member also plays from one moment (the samefor the whole generation) and is ranked on the two rewards added up. 
# Test runs, the saved agents and the logs are always from the start. 
# True / False for both methods, or per method,
# where "whichever was faster" picks whatever gave that method the faster best time in batches 01 (without) and 02 (with).

# rewards   sim/env.py's Rewards, with any changes for the batch
# save_by   which agent each method keeps in <method>_tower.pt: "reward", the one with the highest reward in its
# test run, or "time", the fastest to the getaway (the highest reward until anything finishes). Learning and breeding always use the reward.
# jitter    (optional) sim/env.py's Jitter: PPO's practice episodes get key changes a tick or so late, like the real
#           game. Test runs stay exact. The genetic algorithm ignores it (it ranks members on single exact runs),
#           unless the batch has noisy_runs.
# methods   (optional) which methods run_batches trains for the batch (default both)
# wobble    (optional) sim/env.py's Wobble: PPO's practice episodes nudge the player a little every few ticks, as the
#           shadow sim an agent plays from in the real game (real.live) is nudged by each grab of the screen
# lift_speed_seen  (optional) agents see how the lifts really move (lift_vy); before batch 05 they always saw 0.
#           Saved in each agent's file, so the tools give every agent what it was trained with
# noisy_runs  (optional) the genetic algorithm ranks each member on this many runs with the batch's jitter and wobble
#           instead (half from the start, half from moments of your run; in a generation all members get the same
#           noise), so it breeds agents that cope with late keys and nudges as batch 05's PPO does
# steps     (optional) sim steps each method trains for (default: run_batches' STEPS, 50M)
# seed      (optional) the trainers' random seed (default 0): the same batch again with another seed shows how much
#           of a difference between batches is luck
# late      (optional) seconds: half of PPO's practice episodes start up to this late for the lifts (anywhere from 0,
#           at random; GetawayEnv.reset(late=...)): the lifts and the helicopter carry on that long first while the
#           player waits where they are, as when a real-game run falls behind or dies. The lifts keep their schedule
BATCHES = {
    "01-from-the-start": dict(practice=False, rewards=Rewards(), save_by="reward"),
    "02-your-run": dict(practice=True, rewards=Rewards(), save_by="reward"),
    # the finish pays +1 more for every second left on the 90 s clock: a second saved is worth 1.1 instead of 0.1
    "03-speed": dict(practice={"ppo": True, "genetic": "whichever was faster"},
                     rewards=Rewards(finish_per_second_left=1.0), save_by="time"),
    # batch 02 with late keys, so its run survives the real game's timing (only PPO trained)
    "04-jitter": dict(practice=True, rewards=Rewards(), save_by="reward", jitter=Jitter(), methods=("ppo",)),
    # batch 04 for playing closed loop in the real game (real.live): nudged like its shadow, and seeing the lifts move
    "05-shadow": dict(practice=True, rewards=Rewards(), save_by="reward", jitter=Jitter(), wobble=Wobble(),
                      lift_speed_seen=True, methods=("ppo",)),
    # batch 05 for the genetic algorithm: each member ranked on 4 runs with late keys and nudges. Each generation
    # costs about 2.5 times batch 02's, hence the longer training
    "06-shadow-genetic": dict(practice=True, rewards=Rewards(), save_by="reward", jitter=Jitter(), wobble=Wobble(),
                              lift_speed_seen=True, noisy_runs=4, methods=("genetic",), steps=120e6),
    # batch 05 again in the sim with the helicopter where the game has it (sim/world.py HOVERS, 2026-10-22): batches
    # 01-06 were trained with it 1.05-1.55 m too high
    "07-helicopter": dict(practice=True, rewards=Rewards(), save_by="reward", jitter=Jitter(), wobble=Wobble(),
                          lift_speed_seen=True, methods=("ppo",)),
    # batch 07, late for the lifts: in the real game (2026-10-21) batch 04 got to the second lift 2 s late and walked
    # into its empty shaft, never having been anywhere at another time than in its own runs. Up to 16 s late covers
    # the slowest lift's whole cycle (15.3 s). More to learn, so trained longer
    "08-late": dict(practice=True, rewards=Rewards(), save_by="reward", jitter=Jitter(), wobble=Wobble(),
                    lift_speed_seen=True, late=16.0, methods=("ppo",), steps=80e6),
    # batch 08 got worse everywhere (it kept dying at the second lift, even on time): batch 07, only a little late
    # (up to 4 s: the real game's runs fell 2 s behind)
    "09-a-little-late": dict(practice=True, rewards=Rewards(), save_by="reward", jitter=Jitter(), wobble=Wobble(),
                             lift_speed_seen=True, late=4.0, methods=("ppo",)),
    # batch 07 again with another seed: how far apart two runs of the same batch end up
    "10-helicopter-seed-1": dict(practice=True, rewards=Rewards(), save_by="reward", jitter=Jitter(), wobble=Wobble(),
                                 lift_speed_seen=True, methods=("ppo",), seed=1),
}

# The batch this process trains or shows: set by run_batches (the GETAWAY_BATCH environment variable), else the
# last one. Everything it saves goes in runs/<BATCH>/.
BATCH = os.environ.get("GETAWAY_BATCH") or list(BATCHES)[-1]
if BATCH not in BATCHES:
    raise SystemExit(f"unknown batch {BATCH!r}; the batches are {', '.join(BATCHES)} (common/workers.py)")
RUNS = RUNS_ROOT / BATCH
YOUR_RUN = RUNS / "your_run.json"  # the batch's own copy of your best run, for practice (see keep_your_run)
SAVE_BY = BATCHES[BATCH]["save_by"]

# What both methods train on.
# The stall limit ends an episode as a failure when route progress has not improved for that long. It has
# to be longer than the slowest lift's full cycle (15.3 s on Tower) so that waiting for a lift is never punished.
TRAIN_ENV = dict(map="Tower", action_repeat=2, time_limit_s=90.0, stall_limit_s=20.0,
                 rewards=BATCHES[BATCH]["rewards"], lift_speed_seen=BATCHES[BATCH].get("lift_speed_seen", False))

# Also keep the agent from these test runs (PPO) / generations (genetic), e.g. to show learning in a video:
# runs/<BATCH>/<method>_run<N>.pt. Now every power of 10 (1, 10, 100, 1000, ...). Set to () to turn off.
SNAPSHOTS = tuple(10 ** k for k in range(10))

# Every logged run has its total reward and where it came from (reward_progress, reward_time, reward_deaths,
# reward_finish, reward_stall: they add up to the total).
PART_COLUMNS = [f"reward_{p}" for p in REWARD_PARTS]
RESULT_COLUMNS = ["reward", "progress", "finished", "time_s", "deaths"] + PART_COLUMNS
LOG_COLUMNS = ["wall_s", "sim_steps"] + RESULT_COLUMNS + ["best_progress", "best_time_s"]


def make_env(practice: bool = False, seed: int | None = None) -> GetawayEnv:
    """The training env. practice: for the episodes a method learns from, with the batch's jitter if it has any
    (test runs are always exact)."""
    jitter = BATCHES[BATCH].get("jitter") if practice else None
    wobble = BATCHES[BATCH].get("wobble") if practice else None
    return GetawayEnv(**TRAIN_ENV, jitter=jitter, wobble=wobble, seed=seed)


def agent_env(info: dict, **settings) -> dict:
    """GetawayEnv settings for playing an agent as it was trained: the training settings, with what the agent
    itself saw (from its file: agents from before batch 05 saw the lifts' speed as 0)."""
    return {**TRAIN_ENV, "lift_speed_seen": bool(info.get("lift_speed_seen", False)), **settings}


def dont_overwrite(files, method: str, resume: bool, fresh: bool):
    """Stop before a fresh start would replace a batch's trained agent and logs (a test run without --resume once
    overwrote batch 02's PPO agent, which can't be trained again exactly)."""
    there = [f for f in files if Path(f).exists()]
    if there and not resume and not fresh:
        raise SystemExit(f"runs/{BATCH}/ already has {method} training ({', '.join(Path(f).name for f in there)}). "
                         f"Use --resume to carry on, or --fresh to start over and replace it.")


def keep_your_run(batch: str = BATCH):
    """Copy your best run into a batch's folder when it starts (run_batches does it for all of them at once), so
    that a new best set while they train (or later) never changes what a batch practises from."""
    pb, copy = RUNS_ROOT / f"pb_{TRAIN_ENV['map']}.json", RUNS_ROOT / batch / YOUR_RUN.name
    if pb.exists() and not copy.exists():
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pb, copy)


def practises(method: str) -> bool:
    """Does this method practise from your run in this batch? (See practice in BATCHES.)"""
    setting = BATCHES[BATCH]["practice"]
    setting = setting[method] if isinstance(setting, dict) else setting
    if setting != "whichever was faster":
        return setting
    return best_time("02-your-run", method) <= best_time("01-from-the-start", method)


def best_time(batch: str, method: str) -> float:
    """A method's fastest finish in a batch's test runs (infinity if it never finished, or hasn't trained)."""
    log = RUNS_ROOT / batch / f"{method}_log.csv"
    if not log.exists():
        return math.inf
    with open(log, newline="") as f:
        times = [float(r["best_time_s"]) for r in csv.DictReader(f) if r["best_time_s"]]
    return min(times, default=math.inf)


def your_run(method: str) -> list[int]:
    """The batch's copy of your best run, as actions on the training map, one per physics tick ([] if the method
    doesn't practise from it in this batch, or there is no run)."""
    if not (YOUR_RUN.exists() and practises(method)):
        return []
    run = json.loads(YOUR_RUN.read_text())
    assert run["action_repeat"] == 1, "a personal best is recorded with a decision every tick"
    return run["actions"]


_states_cache: dict = {}
PHYSICS = "2026-10-22"  # the sim's physics version, saved with your runs (sim/world.py MOVEMENT)


def your_run_moments(method: str) -> list[dict]:
    """Your run, as the moments it went through (Game.snapshot(), one per tick), played under the physics it was
    recorded in. Practice starts from these: your keys replayed under today's physics go somewhere else (the sim
    changed on 2026-10-20), but where you were along the route is still a real place to be."""
    run = your_run(method)
    if not run:
        return []
    key = (str(YOUR_RUN), method)
    if key not in _states_cache:
        from sim.world import MOVEMENT, MOVEMENT_2026_10_20, OLD_MOVEMENT, Game
        recorded = json.loads(YOUR_RUN.read_text())  # runs saved before 2026-10-20 don't say: the old physics
        g = Game(TRAIN_ENV["map"], {PHYSICS: MOVEMENT, "2026-10-20": MOVEMENT_2026_10_20}.get(recorded.get("physics"),
                                                                                              OLD_MOVEMENT))
        moments = [g.snapshot()]
        for a in run[:-1]:
            g.step(a)
            moments.append(g.snapshot())
        _states_cache[key] = moments
    return _states_cache[key]


def practice_start(run: list, rand) -> int:
    """A random moment of your run to start from, in physics ticks (rand() gives a number from 0 to 1)."""
    return int(rand() * (len(run) - 1))  # never the last tick, which finishes the run


def play_episode(policy: Policy, env: GetawayEnv, start=None, seed: int | None = None) -> dict:
    """One Tower episode with the agent's best action every step, from the start or from a moment of your run
    (start: one of your_run_moments). The sim has no randomness, so this is an exact score: the same weights
    always give the same result (in an env with jitter or wobble, the same seed)."""
    obs, info = env.reset(start=start, seed=seed)
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        total += reward
        steps += 1
        if terminated or truncated:
            break
    return {"return": total, "progress": info["progress"], "finished": bool(info["finished"]),
            "time_s": info["time_s"], "deaths": info["deaths"], "steps": steps, "reward_parts": info["reward_parts"]}


def better(result: dict, saved: dict | None) -> bool:
    """Should an agent with this result replace the saved one? See SAVE_BY. By time: finishing beats not
    finishing, then the faster finish wins; between two that don't finish, the higher reward wins."""
    if saved is None:
        return True
    if SAVE_BY == "time":
        if result["finished"] != saved["finished"]:
            return result["finished"]
        if result["finished"]:
            return result["time_s"] < saved["time_s"]
    return result["return"] > saved["return"]


def best_of(results: list[dict]) -> int:
    """The index of the result that would be saved out of several (a genetic generation), by SAVE_BY."""
    finishers = [i for i, r in enumerate(results) if r["finished"]]
    if SAVE_BY == "time" and finishers:
        return min(finishers, key=lambda i: results[i]["time_s"])
    return max(range(len(results)), key=lambda i: results[i]["return"])


class Table:
    """A CSV that rows are appended to as training goes. A fresh run starts it over; a resumed one adds to it."""

    def __init__(self, path: Path, columns: list[str], resume: bool):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (resume and path.exists()):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(columns)

    def rows(self, rows: list[list]):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


class Log(Table):
    """One row per evaluation, in the same columns for both methods."""

    def __init__(self, path: Path, resume: bool):
        super().__init__(path, LOG_COLUMNS, resume)

    def row(self, wall_s, sim_steps, result, best_progress, best_time_s):
        self.rows([[f"{wall_s:.1f}", sim_steps] + result_fields(result)
                   + [f"{best_progress:.4f}", "" if best_time_s is None else f"{best_time_s:.2f}"]])


def snapshot_path(method: str, run: int) -> Path | None:
    """Where to keep the agent from this test run / generation, if it is one of SNAPSHOTS."""
    return RUNS / f"{method}_run{run}.pt" if run in SNAPSHOTS else None


def result_fields(result: dict) -> list:
    """An episode's result as CSV fields, in the order of RESULT_COLUMNS."""
    parts = result["reward_parts"]
    return ([f"{result['return']:.4f}", f"{result['progress']:.4f}", int(result["finished"]),
             f"{result['time_s']:.2f}", result["deaths"]] + [f"{parts[p]:.4f}" for p in REWARD_PARTS])


def default_procs() -> int:
    return max(1, (os.cpu_count() or 2))


# --- genetic algorithm: score a whole population in parallel ----------------------------------------

_env = None
_noisy = None  # with the batch's jitter and wobble, for noisy_runs
_policy = None
_run = []


def _init_worker():
    global _env, _noisy, _policy, _run
    full_speed()
    torch.set_num_threads(1)  # one thread per worker, or the workers fight over the cores
    _env = make_env()
    _noisy = make_env(practice=True)
    _policy = Policy()
    _policy.eval()
    _run = your_run_moments("genetic")  # PPO's test runs only ever start at the start


def _score_weights(weights: np.ndarray, start: int = 0, seed: int | None = None) -> dict:
    torch.nn.utils.vector_to_parameters(torch.from_numpy(weights), _policy.parameters())
    return play_episode(_policy, _env if seed is None else _noisy, _run[start] if start else None, seed)


class ScorePool:
    """Plays one episode for each weight vector it is given, spread over the cores."""

    def __init__(self, procs: int | None = None):
        self.procs = procs or default_procs()
        self.pool = mp.get_context("spawn").Pool(self.procs, initializer=_init_worker)

    def score(self, population: torch.Tensor, start: int = 0, seed: int | None = None) -> list[dict]:
        """start: begin at that tick of your run (see practice_start) instead of at the start. seed: play with the
        batch's jitter and wobble, all with the same noise from this seed."""
        jobs = [(row.numpy(), start, seed) for row in population]
        return self.pool.starmap(_score_weights, jobs, chunksize=max(1, len(jobs) // (self.procs * 4)))

    def score_later(self, weights: torch.Tensor):
        """Score one agent in the background; call .ready() / .get() on what this returns."""
        return self.pool.apply_async(_score_weights, (weights.numpy().copy(),))

    def close(self):
        self.pool.terminate()
        self.pool.join()


# --- PPO: collect experience in parallel -----------------------------------------------------------

def _rollout_worker(conn, n_envs: int, seed: int):
    """Holds n_envs copies of the env and its own copy of the policy. On request it loads the latest
    weights, plays n_steps in every env, and sends the whole stretch back at once (messaging every
    single step between processes is slow on Windows)."""
    full_speed()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    run = your_run_moments("ppo")
    late_s = BATCHES[BATCH].get("late", 0.0)
    envs = [make_env(practice=True, seed=seed * 100 + i) for i in range(n_envs)]
    policy = Policy()
    policy.eval()
    returns = [0.0] * n_envs
    starts = [0] * n_envs  # the tick of your run each env's episode started from (0: the start)
    lates = [0] * n_envs  # and how many ticks late for the lifts it started

    def reset(i):
        starts[i] = practice_start(run, rng.random) if run and rng.random() < 0.5 else 0
        late = int(rng.random() * late_s / DT) if late_s and rng.random() < 0.5 else 0
        obs = envs[i].reset(start=run[starts[i]] if starts[i] else None, late=late)[0]
        lates[i] = envs[i].late
        return obs

    obs = np.stack([reset(i) for i in range(n_envs)])
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            weights, n_steps = msg
            torch.nn.utils.vector_to_parameters(torch.from_numpy(weights), policy.parameters())
            O = np.zeros((n_steps, n_envs, OBS_SIZE), np.float32)
            A = np.zeros((n_steps, n_envs), np.int64)
            R = np.zeros((n_steps, n_envs), np.float32)
            TE = np.zeros((n_steps, n_envs), np.bool_)
            TR = np.zeros((n_steps, n_envs), np.bool_)
            cut_obs = []  # (step, env, observation) where an episode was cut off rather than ended
            episodes = []
            for t in range(n_steps):
                O[t] = obs
                with torch.no_grad():
                    probs = torch.softmax(policy(torch.from_numpy(obs)), dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(1).numpy()
                A[t] = actions
                for i, env in enumerate(envs):
                    o, r, te, tr, info = env.step(int(actions[i]))
                    R[t, i], TE[t, i], TR[t, i] = r, te, tr
                    returns[i] += r
                    if te or tr:
                        if tr:
                            cut_obs.append((t, i, o))
                        episodes.append({"start_s": starts[i] * DT, "late_s": lates[i] * DT,
                                         "return": returns[i], "progress": info["progress"],
                                         "finished": bool(info["finished"]), "time_s": info["time_s"],
                                         "deaths": info["deaths"], "reward_parts": info["reward_parts"]})
                        returns[i] = 0.0
                        o = reset(i)
                    obs[i] = o
            conn.send((O, A, R, TE, TR, cut_obs, episodes, obs.copy()))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


class RolloutPool:
    """n_envs copies of the training env spread over worker processes, for PPO."""

    def __init__(self, n_envs: int, procs: int | None = None, seed: int = 0):
        procs = min(procs or default_procs(), n_envs)
        ctx = mp.get_context("spawn")
        self.sizes = [n_envs // procs + (1 if i < n_envs % procs else 0) for i in range(procs)]
        self.conns, self.procs = [], []
        for k, size in enumerate(self.sizes):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_rollout_worker, args=(child, size, seed * 1000 + k), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def collect(self, weights: np.ndarray, n_steps: int):
        """Returns obs [T,N,53], actions, rewards, terminated, truncated [T,N], the cut-off observations
        as (step, env, obs), the finished episodes' results, and the observation after the last step [N,53]."""
        for conn in self.conns:
            conn.send((weights, n_steps))
        parts = [conn.recv() for conn in self.conns]
        cut_obs, episodes, start = [], [], 0
        for size, part in zip(self.sizes, parts):
            cut_obs += [(t, start + i, o) for t, i, o in part[5]]
            episodes += part[6]
            start += size
        stacked = [np.concatenate([part[k] for part in parts], axis=1) for k in range(5)]
        last_obs = np.concatenate([part[7] for part in parts], axis=0)
        return (*stacked, cut_obs, episodes, last_obs)

    def close(self):
        for conn in self.conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()


def keep_awake():
    """Keep Windows from going to sleep while this process trains (a sleep froze a run for 25 min). Windows lets
    go of it when the process ends. Closing a laptop's lid still sleeps."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    full_speed()


def full_speed():
    """Stop Windows from slowing this process down as a background job ("efficiency mode"). Started from a window
    that isn't in front (e.g. by a script), training ran at half speed until this was turned off."""
    if os.name == "nt":
        import ctypes

        class State(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE  # 64-bit handles: undeclared, ctypes would cut them to 32 bits
        k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        state = State(1, 0x1 | 0x4, 0)  # execution speed and timer resolution: not throttled
        k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))


class Stopwatch:
    """Wall-clock training time that carries on from where a resumed run left off."""

    def __init__(self, already: float = 0.0):
        self.already = already
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return self.already + time.perf_counter() - self.start
