"""Chart of PPO vs the genetic algorithm from a batch's training logs.

    py -m common.compare                     -> runs/<last batch>/compare.png
    py -m common.compare 01-from-the-start   -> the chart for a particular batch
    py -m common.compare --live              a window that redraws itself while training runs
    py -m common.compare --all               every batch on one chart -> runs/compare_all.png
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.workers import BATCH, BATCHES, RUNS_ROOT, TRAIN_ENV

METHODS = [("PPO (reinforcement learning)", "ppo_log.csv", "tab:blue"),
           ("Genetic algorithm", "genetic_log.csv", "tab:orange")]


def read(path):
    d = {"steps": [], "minutes": [], "progress": [], "time": []}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                row = (int(r["sim_steps"]) / 1e6, float(r["wall_s"]) / 60, float(r["best_progress"]) * 100,
                       float(r["best_time_s"]) if r["best_time_s"] else None)
            except (TypeError, ValueError):
                continue  # a row the trainer is still in the middle of writing
            for key, v in zip(d, row):
                d[key].append(v)
    return d


def load(folder):
    logs = [(name, read(folder / file), colour) for name, file, colour in METHODS if (folder / file).exists()]
    return [log for log in logs if log[1]["steps"]]


def human_time(folder):
    """Your best time on the training map in the sim, the line to beat: the batch's copy of your run if it has
    one, else your current best (sim.play saves it)."""
    for pb in (folder / "your_run.json", RUNS_ROOT / f"pb_{TRAIN_ENV['map']}.json"):
        if pb.exists():
            return json.loads(pb.read_text())["time_s"]
    return None


def draw(axes, logs, human):
    for col, (key, label) in enumerate((("steps", "sim steps used (millions)"), ("minutes", "training time (minutes)"))):
        top, bottom = axes[0][col], axes[1][col]
        for name, d, colour in logs:
            top.plot(d[key], d["progress"], color=colour, label=name)
            xs = [x for x, t in zip(d[key], d["time"]) if t is not None]
            ts = [t for t in d["time"] if t is not None]
            if xs:
                bottom.plot(xs, ts, color=colour, label=name)
        top.set_ylabel("furthest reached (% of the Tower route)")
        top.set_ylim(0, 105)
        if human:
            bottom.axhline(human, color="gray", linestyle="--", label=f"your best ({human:.2f} s)")
        bottom.set_ylabel("best finish time (s)")
        bottom.set_xlabel(label)
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
    axes[0][0].set_title("How far each method gets")
    axes[0][1].set_title("...against wall-clock time")


def draw_all(out):
    """Every batch's PPO (solid) and genetic algorithm (dashed) against sim steps, one colour per batch."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    human = None
    for batch, colour in zip(BATCHES, ("tab:gray", "tab:blue", "tab:red", "tab:green", "tab:purple")):
        folder = RUNS_ROOT / batch
        human = human or human_time(folder)
        for name, d, _ in load(folder):
            style = "-" if name.startswith("PPO") else "--"
            label = f"{batch} {'PPO' if style == '-' else 'genetic'}"
            top.plot(d["steps"], d["progress"], style, color=colour, label=label)
            xs = [x for x, t in zip(d["steps"], d["time"]) if t is not None]
            if xs:
                bottom.plot(xs, [t for t in d["time"] if t is not None], style, color=colour, label=label)
    if human:
        bottom.axhline(human, color="k", linestyle=":", label=f"your best ({human:.2f} s)")
    top.set_ylabel("furthest reached (% of the Tower route)")
    top.set_ylim(0, 105)
    top.set_title("All batches: PPO solid, genetic algorithm dashed")
    bottom.set_ylabel("best finish time (s)")
    bottom.set_xlabel("sim steps used (millions)")
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


def live(folder, every: float):
    plt.switch_backend("TkAgg")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.canvas.manager.set_window_title(f"Training: PPO vs genetic algorithm ({folder.name})")
    while plt.fignum_exists(fig.number):
        for ax in axes.flat:
            ax.clear()
        draw(axes, load(folder), human_time(folder))
        fig.tight_layout()
        plt.pause(every)  # keeps the window responsive while waiting; closing the window ends the loop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", nargs="?", default=BATCH, help=f"which batch folder in runs/ (default: {BATCH})")
    ap.add_argument("--live", action="store_true", help="open a window and keep redrawing it as the logs grow")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between redraws in --live")
    ap.add_argument("--all", action="store_true", help="every batch on one chart, runs/compare_all.png")
    args = ap.parse_args()
    if args.all:
        return draw_all(RUNS_ROOT / "compare_all.png")
    folder = RUNS_ROOT / args.batch
    if args.live:
        return live(folder, args.every)

    logs = load(folder)
    if not logs:
        raise SystemExit(f"no training logs in runs/{args.batch}/ yet. Run py -m ppo.train and py -m genetic.train first.")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    draw(axes, logs, human_time(folder))
    fig.tight_layout()
    out = folder / "compare.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)
    for name, d, _ in logs:
        best = min((t for t in d["time"] if t is not None), default=None)
        print(f"  {name}: {d['steps'][-1]:.1f}M steps in {d['minutes'][-1]:.0f} min, reached {max(d['progress']):.0f}% of the route, "
              f"best finish {'none yet' if best is None else format(best, '.2f') + ' s'}")


if __name__ == "__main__":
    main()
