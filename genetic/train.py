"""Method 2: a genetic algorithm, written in PyTorch.

    py -m genetic.train                 # train on Tower, 50M sim steps
    py -m genetic.train --steps 100e6   # a longer run
    py -m genetic.train --resume        # carry on from the last checkpoint

A population of networks each plays the map once (and, if the batch practises from your run, once more from
a random moment of your best run, the same moment for everyone in a generation). The ones with the most reward
are kept; the rest are replaced by copies of the survivors with small random changes to their weights. No
gradients are used at all.

A batch with noisy_runs (BATCHES) ranks the members on that many runs with its late keys and nudges instead,
half from the start and half from moments of your run, all members of a generation with the same noise. Each also
plays once from the start exactly, for the log; the agent saved and logged is the generation's best-ranked member.

Usually run by py -m common.run_batches, which picks the batch (BATCHES in common/workers.py). Everything is
saved to the batch's folder, runs/<batch>/, as it goes, so training can be stopped (Ctrl+C) and resumed:
    genetic_tower.pt        the best agent so far, by the batch's save_by
    genetic_run<N>.pt       the best member of generation N, for the generations in SNAPSHOTS
    genetic_checkpoint.pt   the whole population, for --resume
    genetic_log.csv         each generation's best (reward and its breakdown, progress, time, deaths), for py -m common.compare
    genetic_population.csv  every member of every generation (same columns, from the start), which member it was
                            bred from, and its practice from your run (practice_start_s, practice_reward), raw;
                            with noisy_runs, where each noisy run started and its reward (0.00: from the start)
"""
from __future__ import annotations

import argparse
import time

import torch

from common.policy import Policy, save_agent
from common.workers import (BATCH, BATCHES, RESULT_COLUMNS, RUNS, TRAIN_ENV, Log, ScorePool, Stopwatch, Table, best_of,
                            better, dont_overwrite, keep_awake, keep_your_run, practice_start, result_fields,
                            snapshot_path, your_run)
from sim.world import DT

AGENT = RUNS / "genetic_tower.pt"
CHECKPOINT = RUNS / "genetic_checkpoint.pt"
LOG = RUNS / "genetic_log.csv"
POPULATION = RUNS / "genetic_population.csv"


def flat_weights(policy: Policy) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(policy.parameters()).detach().clone()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=float, default=50e6, help="total sim steps to train for")
    ap.add_argument("--fresh", action="store_true", help="start over, replacing this batch's files for this method")
    ap.add_argument("--resume", action="store_true", help="continue from the batch's genetic_checkpoint.pt")
    ap.add_argument("--population", type=int, default=128)
    ap.add_argument("--survivors", type=int, default=16, help="how many of the best are kept as parents")
    ap.add_argument("--mutation", type=float, default=0.05, help="size of the random change to each weight")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    total_steps = int(args.steps)
    dont_overwrite([CHECKPOINT, LOG, AGENT], "genetic", args.resume, args.fresh)
    keep_awake()
    keep_your_run()

    rng = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    # every member starts as a freshly initialised copy of the shared network
    population = torch.stack([flat_weights(Policy()) for _ in range(args.population)])
    steps, generation, already = 0, 0, 0.0
    best = {"progress": 0.0, "time_s": None, "saved": None}  # saved: the result of the agent in AGENT
    champion = None  # the unchanged best member's result; the sim is exact, so it never needs replaying
    parents = [None] * args.population  # each member's parent in the previous generation (none at the start)

    resumed = args.resume and CHECKPOINT.exists()
    if resumed:
        ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        population, steps, generation, already = ck["population"], ck["steps"], ck["generation"], ck["wall_s"]
        best, champion, parents = ck["best"], ck["champion"], ck["parents"]
        rng.set_state(ck["rng"])
        print(f"resumed at generation {generation}, {steps:,} steps ({already / 60:.1f} min of training so far)")
    elif args.resume:
        print("no checkpoint found, starting fresh")

    log = Log(LOG, resumed)
    raw = Table(POPULATION, ["generation", "sim_steps", "member", "parent"] + RESULT_COLUMNS
                + ["practice_start_s", "practice_reward"], resumed)
    run = your_run("genetic")
    noisy = BATCHES[BATCH].get("noisy_runs", 0)
    clock = Stopwatch(already)
    pool = ScorePool()
    last_save = time.perf_counter()
    print(f"genetic algorithm on {TRAIN_ENV['map']}: population {args.population}, {args.survivors} survive, "
          f"{population.shape[1]:,} weights each, training to {total_steps:,} steps, "
          f"{'practising from your run' if run else 'every episode from the start'}"
          + (f", ranked on {noisy} runs with late keys and nudges" if noisy else ""))

    def save_checkpoint():
        tmp = CHECKPOINT.with_suffix(".tmp")
        torch.save({"population": population, "steps": steps, "generation": generation, "wall_s": clock.elapsed(),
                    "best": best, "champion": champion, "parents": parents, "rng": rng.get_state()}, tmp)
        tmp.replace(CHECKPOINT)

    try:
        while steps < total_steps:
            # 1. everyone plays from the start (member 0 is last generation's unchanged best: reuse its result)
            if champion is None:
                results = pool.score(population)
            else:
                results = [champion] + pool.score(population[1:])
            steps += sum(r["steps"] for r in results[(0 if champion is None else 1):])
            if noisy:
                # ...then is ranked on runs with late keys and nudges, every other one from a moment of your run
                seeds = torch.randint(0, 2 ** 31 - 1, (noisy,), generator=rng).tolist()
                starts = [practice_start(run, lambda: torch.rand(1, generator=rng).item()) if run and k % 2 else 0
                          for k in range(noisy)]
                shaken = [pool.score(population, start, seed) for start, seed in zip(starts, seeds)]
                steps += sum(r["steps"] for runs in shaken for r in runs)
                scores = torch.tensor([sum(runs[i]["return"] for runs in shaken) for i in range(len(results))])
                practice_fields = [[" ".join(f"{start * DT:.2f}" for start in starts),
                                    " ".join(f"{runs[i]['return']:.4f}" for runs in shaken)] for i in range(len(results))]
            else:
                # ...and from a moment of your run
                if run:
                    start = practice_start(run, lambda: torch.rand(1, generator=rng).item())
                    practice = pool.score(population, start)
                    steps += sum(r["steps"] for r in practice)
                scores = torch.tensor([r["return"] + (practice[i]["return"] if run else 0.0) for i, r in enumerate(results)])
                practice_fields = [[f"{start * DT:.2f}", f"{practice[i]['return']:.4f}"] if run else ["", ""]
                                   for i in range(len(results))]
            generation += 1
            raw.rows([[generation, steps, i, "" if p is None else p] + result_fields(r) + practice_fields[i]
                      for i, (p, r) in enumerate(zip(parents, results))])

            # 2. rank them; the top few survive
            order = scores.argsort(descending=True)
            top = order[:args.survivors]

            # the generation's best run from the start, by the batch's save_by: the candidate for the saved agent, and logged
            # (with noisy_runs, the best-ranked member's run: an exact run alone would pick out lucky ones again)
            pick = int(order[0]) if noisy else best_of(results)
            winner = results[pick]
            agent = Policy()
            torch.nn.utils.vector_to_parameters(population[pick].clone(), agent.parameters())
            info = dict(method="genetic", trained_steps=steps, generation=generation,
                        action_repeat=TRAIN_ENV["action_repeat"], lift_speed_seen=TRAIN_ENV["lift_speed_seen"], **winner)
            if better(winner, best["saved"]):
                best["saved"] = winner
                save_agent(AGENT, agent, **info)
            if snapshot := snapshot_path("genetic", generation):
                save_agent(snapshot, agent, **info)
            best["progress"] = max(best["progress"], max(r["progress"] for r in results))
            finish_times = [r["time_s"] for r in results if r["finished"]]
            if finish_times and (best["time_s"] is None or min(finish_times) < best["time_s"]):
                best["time_s"] = min(finish_times)
            log.row(clock.elapsed(), steps, winner, best["progress"], best["time_s"])
            done = f"FINISHED in {winner['time_s']:.2f}s" if winner["finished"] else f"{winner['progress'] * 100:5.1f}% of the route"
            average = sum(r["progress"] for r in results) / len(results) * 100
            print(f"gen {generation:4d}  {steps / 1e6:7.2f}M steps  {clock.elapsed() / 60:6.1f} min  best: {done}  "
                  f"(reward {winner['return']:+.2f}, deaths {winner['deaths']})  population avg {average:4.1f}%  "
                  f"best time {'-' if best['time_s'] is None else format(best['time_s'], '.2f') + 's'}"
                  + (f"  noisy runs: best avg reward {scores.max().item() / noisy:+.2f}" if noisy else ""))

            # 3. next generation: the best one unchanged, the rest are mutated copies of random survivors
            survivors = population[top]
            picks = torch.randint(0, args.survivors, (args.population - 1,), generator=rng)
            noise = torch.randn(args.population - 1, population.shape[1], generator=rng)
            population = torch.cat([survivors[:1], survivors[picks] + args.mutation * noise])
            parents = [int(top[0])] + top[picks].tolist()
            champion = results[int(order[0])]

            if time.perf_counter() - last_save > 120:
                save_checkpoint()
                last_save = time.perf_counter()
        print("done.")
    except KeyboardInterrupt:
        print("\nstopped. Progress is saved; run again with --resume to continue.")
    finally:
        save_checkpoint()
        pool.close()


if __name__ == "__main__":
    main()
