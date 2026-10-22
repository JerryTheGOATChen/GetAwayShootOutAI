"""Chart of the landing re-jump probes (real/probes/rejump_*.json): which way the jump goes, against how long after
the landing it comes, in the real game and in the sim. Numbers from py -m real.probe_fit on the probes' trials."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIM, REAL, INK, INK_2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# turning time at the jump (s) and the jump's angle from upright (deg), real game (trials 2026-10-20 00-31-03)
TURN_S = np.array([2, 5, 8, 14, 27]) * 0.02
REAL_DEG = np.array([9, 10, 14, 29, 52])
SIM_DEG = np.array([56, 56, 56, 56, 56])

def main():
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    t = np.linspace(0, 0.6, 100)
    ax.plot(t, np.minimum(200 * t ** 2, 53.5), color=INK_2, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(0.47, 36, "the lean from standing", color=INK_2, fontsize=8.5, ha="right")
    ax.scatter(TURN_S, SIM_DEG, s=60, color=SIM, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.scatter(TURN_S, REAL_DEG, s=60, color=REAL, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.text(0.30, 59, "sim: along the body, still tilted from the landing", color=INK_2, fontsize=8.5)
    ax.text(0.03, 17, "real game", color=INK_2, fontsize=8.5)
    ax.set_xlabel("time from the landing (holding the lean key) to the jump, s", color=INK_2)
    ax.set_ylabel("jump direction, degrees from upright", color=INK_2)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 65)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.set_title("Jumping again right after a landing", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig("runs/rejump_probe.png", dpi=130, facecolor=SURFACE)
    print("saved runs/rejump_probe.png")


if __name__ == "__main__":
    main()
