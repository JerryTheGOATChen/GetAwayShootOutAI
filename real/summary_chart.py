"""The real-game test on one chart: how long each run's blind replay stays on its sim run, in the sim with timing
errors (real.robustness) and in the real game (real.trials_report), and how long real trials stay with each other.

    py -m real.summary_chart --sim real/robustness_blind.json --real "real/trials/<date time>" [more folders]
        --out runs/real_game.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIM, REAL, TOGETHER = "#2a78d6", "#eb6834", "#1baf7a"  # reference palette slots 1-3
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
def name(run: str) -> str:
    """A run's label: your PB, or a trained agent's batch, method and time in the sim."""
    path = Path(run)
    if path.name.startswith("pb_"):
        return "Your PB (42.42 s, old sim)"
    batch, method = path.parent.name, path.stem.split("_")[0]
    label = f"{batch} {'PPO' if method == 'ppo' else method}"
    if path.suffix == ".pt" and path.exists():
        from common.policy import load_agent
        info = load_agent(path)[1]
        if info.get("finished"):
            label += f" ({info['time_s']:.2f} s)"
    return label


def load_real(folders):
    out = {}
    for folder in folders:
        for run, r in json.loads((Path(folder) / "report.json").read_text()).items():
            o = out.setdefault(run, {"stays_s": [], "together_s": []})
            o["stays_s"] += r["stays_s"]
            o["together_s"] += r["together_s"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, help="real.robustness --out JSON (blind)")
    ap.add_argument("--real", nargs="+", required=True, help="real.trials folders with a report.json")
    ap.add_argument("--out", default="runs/real_game.png")
    args = ap.parse_args()
    sim = json.loads(Path(args.sim).read_text())
    real = load_real(args.real)
    runs = sorted(set(sim) | set(real), key=lambda r: (not Path(r).name.startswith("pb_"), r))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(10, 1.1 * len(runs) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rows = []
    for i, run in enumerate(runs):
        y = len(runs) - 1 - i
        rows.append(y)
        bands = [(sim.get(run, {}).get("stays_s", []), SIM, +0.25, 10),
                 (real.get(run, {}).get("stays_s", []), REAL, 0.0, 36),
                 (real.get(run, {}).get("together_s", []), TOGETHER, -0.25, 36)]
        for values, colour, dy, size in bands:
            if not values:
                continue
            v = np.asarray(values)
            ax.scatter(v, y + dy + rng.uniform(-0.07, 0.07, len(v)), s=size, color=colour, alpha=0.55 if size < 20 else 0.9,
                       edgecolors=SURFACE, linewidths=1.0, zorder=3)
            med = float(np.median(v))
            ax.plot([med, med], [y + dy - 0.11, y + dy + 0.11], color=INK, linewidth=2, zorder=4)
            ax.text(med + 0.4, y + dy + 0.02, f"{med:.1f} s", color=INK_2, fontsize=8, va="center", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    labels = [name(r) + ("" if r in real else "\nreal game: not run yet") for r in runs]
    ax.set_yticks(rows, labels, color=INK, fontsize=10)
    ax.set_xlabel("seconds the blind replay stays on track (jeans within 100 px on screen)", color=INK_2)
    ax.set_xlim(0, 45)
    ax.set_ylim(-0.6, len(runs) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=l) for c, l in
               ((SIM, "sim, with real-game timing errors (100 replays)"),
                (REAL, "real game vs its sim run (each trial)"),
                (TOGETHER, "real game vs real game (each pair of trials)"))]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2, label="median"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    ax.set_title("Agents' runs played blind into the real game", color=INK, fontsize=13, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
