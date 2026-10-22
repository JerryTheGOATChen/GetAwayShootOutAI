"""Chart: how long the sim follows the real game for each run, before and after the 2026-10-20 rebuild, against how
long real trials of the run follow each other (the most any sim could). Numbers from real.physics_check /
real.trials_report on the trials in real/trials/ (see EXPERIMENTS.md, "Rebuilding the sim from the real game")."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BEFORE, AFTER, REAL, INK, INK_2, GRID, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# run: (old sim and old start timing, rebuilt sim and timing, real vs real: median pair)
RUNS = {"Your PB (42.42 s)": (2.6, 14.4, 13.9),
        "02 genetic": (5.9, 13.2, 11.0),
        "02 PPO": (1.5, 7.8, 6.5),
        "03 speed PPO": (1.5, 5.9, 5.1),
        "04 jitter PPO": (1.6, 4.3, 3.5)}


def main():
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    names = list(RUNS)
    for i, name in enumerate(names):
        y = len(names) - 1 - i
        before, after, real = RUNS[name]
        ax.plot([before, after], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.scatter([real], [y], marker="|", s=420, color=REAL, linewidths=3, zorder=2)
        ax.scatter([before], [y], s=70, color=BEFORE, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.scatter([after], [y], s=70, color=AFTER, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(after + 0.35, y, f"{after:.1f} s", color=INK_2, fontsize=8.5, va="center")
    ax.set_yticks(range(len(names)), names[::-1], color=INK, fontsize=10)
    ax.set_xlabel("seconds a real-game trial stays on its sim run (jeans within 100 px)", color=INK_2)
    ax.set_xlim(0, 16)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=BEFORE, markersize=7, label="old sim"),
               plt.Line2D([], [], marker="o", linestyle="", color=AFTER, markersize=7, label="rebuilt sim"),
               plt.Line2D([], [], marker="|", linestyle="", color=REAL, markersize=12, markeredgewidth=3,
                          label="real trials vs each other (the limit)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    ax.set_title("How long the sim follows the real game", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig("runs/sim_vs_real.png", dpi=130, facecolor=SURFACE)
    print("saved runs/sim_vs_real.png")


if __name__ == "__main__":
    main()
