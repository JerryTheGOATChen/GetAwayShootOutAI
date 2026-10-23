"""Method 1: PPO (reinforcement learning), written in PyTorch.

    py -m ppo.train                    # train on Tower, 50M sim steps
    py -m ppo.train --steps 100e6      # a longer run
    py -m ppo.train --resume           # carry on from the last checkpoint

One network plays many copies of the game at once. After each stretch of play it nudges its weights
so that actions which led to more reward than expected become more likely. If the batch practises
from your run, half its practice episodes start from a random moment of your best run.

Usually run by py -m common.run_batches, which picks the batch (BATCHES in common/workers.py). Everything is
saved to the batch's folder, runs/<batch>/, as it goes, so training can be stopped (Ctrl+C) and resumed:
    ppo_tower.pt        the best agent so far, by the batch's save_by
    ppo_run<N>.pt       the agent from the Nth test run, for the test runs in SNAPSHOTS
    ppo_checkpoint.pt   the full training state, for --resume
    ppo_log.csv         every test run (reward and its breakdown, progress, time, deaths), for py -m common.compare
    ppo_episodes.csv    every practice episode, the same columns and where in your run it started (start_s,
                        0 = from the start; late_s, how late for the lifts, if the batch has late), raw, for graphs
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from common.policy import HIDDEN, Policy, save_agent
from common.workers import (BATCH, BATCHES, RESULT_COLUMNS, RUNS, TRAIN_ENV, Log, RolloutPool, ScorePool, Stopwatch,
                            Table, better, dont_overwrite, keep_awake, keep_your_run, result_fields, snapshot_path)
from sim.env import OBS_SIZE

AGENT = RUNS / "ppo_tower.pt"
CHECKPOINT = RUNS / "ppo_checkpoint.pt"
LOG = RUNS / "ppo_log.csv"
EPISODES = RUNS / "ppo_episodes.csv"


def value_network() -> nn.Module:
    """Predicts how much reward is still to come from a given situation. Only used while training."""
    return nn.Sequential(nn.Linear(OBS_SIZE, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, 1))


def init_weights(policy: Policy, value: nn.Module) -> None:
    for module, last_gain in ((policy.net, 0.01), (value, 1.0)):
        layers = [m for m in module if isinstance(m, nn.Linear)]
        for layer in layers:
            nn.init.orthogonal_(layer.weight, gain=last_gain if layer is layers[-1] else 2 ** 0.5)
            nn.init.zeros_(layer.bias)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=float, default=50e6, help="total sim steps to train for")
    ap.add_argument("--fresh", action="store_true", help="start over, replacing this batch's files for this method")
    ap.add_argument("--resume", action="store_true", help="continue from the batch's ppo_checkpoint.pt")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=128, help="steps per env between updates")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5, help="updates between scored test runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    total_steps = int(args.steps)
    dont_overwrite([CHECKPOINT, LOG, AGENT], "PPO", args.resume, args.fresh)
    keep_awake()
    keep_your_run()

    torch.manual_seed(args.seed)
    policy, value = Policy(), value_network()
    init_weights(policy, value)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=args.lr, eps=1e-5)
    steps, updates, tests, already = 0, 0, 0, 0.0
    best = {"progress": 0.0, "time_s": None, "saved": None}  # saved: the result of the agent in AGENT

    resumed = args.resume and CHECKPOINT.exists()
    if resumed:
        ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["policy"])
        value.load_state_dict(ck["value"])
        optimizer.load_state_dict(ck["optimizer"])
        steps, updates, tests, already, best = ck["steps"], ck["updates"], ck["tests"], ck["wall_s"], ck["best"]
        print(f"resumed at {steps:,} steps ({already / 60:.1f} min of training so far)")
    elif args.resume:
        print("no checkpoint found, starting fresh")

    log = Log(LOG, resumed)
    late = "late" in BATCHES[BATCH]  # (only these batches' episodes say how late they started)
    raw = Table(EPISODES, ["sim_steps", "update", "start_s"] + ["late_s"] * late + RESULT_COLUMNS, resumed)
    clock = Stopwatch(already)
    workers = RolloutPool(args.envs, seed=args.seed + updates)
    scorer = ScorePool(1)  # scores the current agent in the background without pausing training
    pending = None  # (async result, steps when it was submitted, the weights that were scored)
    recent = deque(maxlen=200)
    last_save = time.perf_counter()
    batch = args.envs * args.rollout
    print(f"PPO on {TRAIN_ENV['map']}: {args.envs} envs x {args.rollout} steps = {batch:,} per update, "
          f"training to {total_steps:,} steps")

    def save_checkpoint():
        tmp = CHECKPOINT.with_suffix(".tmp")
        torch.save({"policy": policy.state_dict(), "value": value.state_dict(), "optimizer": optimizer.state_dict(),
                    "steps": steps, "updates": updates, "tests": tests, "wall_s": clock.elapsed(), "best": best}, tmp)
        tmp.replace(CHECKPOINT)

    def collect_score(wait: bool):
        nonlocal pending, tests
        if pending is None or not (wait or pending[0].ready()):
            return
        result, at_steps, weights = pending[0].get(), pending[1], pending[2]
        pending = None
        tests += 1
        scored = Policy()
        torch.nn.utils.vector_to_parameters(weights, scored.parameters())
        info = dict(method="ppo", trained_steps=at_steps, test_run=tests, action_repeat=TRAIN_ENV["action_repeat"],
                    lift_speed_seen=TRAIN_ENV["lift_speed_seen"], **result)
        if better(result, best["saved"]):
            best["saved"] = result
            save_agent(AGENT, scored, **info)
        if snapshot := snapshot_path("ppo", tests):
            save_agent(snapshot, scored, **info)
        best["progress"] = max(best["progress"], result["progress"])
        if result["finished"] and (best["time_s"] is None or result["time_s"] < best["time_s"]):
            best["time_s"] = result["time_s"]
        log.row(clock.elapsed(), at_steps, result, best["progress"], best["time_s"])
        done = f"FINISHED in {result['time_s']:.2f}s" if result["finished"] else f"{result['progress'] * 100:5.1f}% of the route"
        from_start = [e["progress"] for e in recent if e["start_s"] == 0 and e["late_s"] == 0]
        train = np.mean(from_start) * 100 if from_start else 0.0
        print(f"{at_steps / 1e6:7.2f}M steps  {clock.elapsed() / 60:6.1f} min  test run: {done}  "
              f"(reward {result['return']:+.2f}, deaths {result['deaths']})  practice avg {train:4.1f}%  "
              f"best time {'-' if best['time_s'] is None else format(best['time_s'], '.2f') + 's'}")

    try:
        while steps < total_steps:
            # learning rate falls in a straight line to zero over the whole run
            for group in optimizer.param_groups:
                group["lr"] = args.lr * max(0.0, 1.0 - steps / total_steps)

            weights = torch.nn.utils.parameters_to_vector(policy.parameters()).detach()
            O, A, R, TE, TR, cut_obs, episodes, last_obs = workers.collect(weights.numpy(), args.rollout)
            recent.extend(episodes)
            steps += batch
            updates += 1
            raw.rows([[steps, updates, f"{e['start_s']:.2f}"] + [f"{e['late_s']:.2f}"] * late + result_fields(e)
                      for e in episodes])

            obs = torch.from_numpy(O)
            actions = torch.from_numpy(A)
            rewards = torch.from_numpy(R)
            with torch.no_grad():
                values = value(obs).squeeze(-1)
                next_value = value(torch.from_numpy(last_obs)).squeeze(-1)
                old_logp = torch.distributions.Categorical(logits=policy(obs)).log_prob(actions)
                # an episode that was cut off (time or stall limit) did not really end: count what the
                # value network expects would have followed
                for t, i, o in cut_obs:
                    rewards[t, i] += args.gamma * value(torch.from_numpy(o)).item()
            ended = torch.from_numpy(TE | TR).float()

            # generalised advantage estimation: how much better each action turned out than expected
            adv = torch.zeros_like(rewards)
            running = torch.zeros(args.envs)
            for t in reversed(range(args.rollout)):
                after = next_value if t == args.rollout - 1 else values[t + 1]
                alive = 1.0 - ended[t]
                delta = rewards[t] + args.gamma * after * alive - values[t]
                running = delta + args.gamma * args.gae * alive * running
                adv[t] = running
            returns = adv + values

            b_obs, b_act = obs.reshape(-1, OBS_SIZE), actions.reshape(-1)
            b_logp, b_adv, b_ret = old_logp.reshape(-1), adv.reshape(-1), returns.reshape(-1)
            size = batch // args.minibatches
            for _ in range(args.epochs):
                order = torch.randperm(batch)
                for k in range(0, batch, size):
                    idx = order[k:k + size]
                    dist = torch.distributions.Categorical(logits=policy(b_obs[idx]))
                    ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    value_loss = 0.5 * (value(b_obs[idx]).squeeze(-1) - b_ret[idx]).pow(2).mean()
                    loss = policy_loss + 0.5 * value_loss - args.entropy * dist.entropy().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), 0.5)
                    optimizer.step()

            collect_score(wait=False)
            if updates % args.eval_every == 0 and pending is None:
                scored = torch.nn.utils.parameters_to_vector(policy.parameters()).detach().clone()
                pending = (scorer.score_later(scored), steps, scored)
            if time.perf_counter() - last_save > 120:
                save_checkpoint()
                last_save = time.perf_counter()
        collect_score(wait=True)
        print("done.")
    except KeyboardInterrupt:
        print("\nstopped. Progress is saved; run again with --resume to continue.")
    finally:
        save_checkpoint()
        workers.close()
        scorer.close()


if __name__ == "__main__":
    main()
