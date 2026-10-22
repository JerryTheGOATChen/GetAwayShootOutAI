"""The picture for the write-up: one of your recorded runs, tracked from the screen (real.locate), drawn over the
sim's own view of the level, with the same keys played blind in the sim and the shadow kept on the run
(real.shadow).

    py -m real.shadow_chart                              # 2026-10-02 23-41-14 -> runs/shadow_vs_blind.png
    py -m real.shadow_chart --video "2026-10-02 23-45-41"

Needs the run's track (py -m real.locate --video ...) and the recording itself (its key overlay gives the keys).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import real.shadow as rs

ROOT = Path(__file__).resolve().parent.parent
PPM = 20  # picture px per m
REAL, SHADOW, BLIND = (255, 255, 255), (60, 120, 220), (150, 90, 210)


def paths(stem: str) -> dict:
    """Where Steve went: tracked from the screen (real), and every tick of the keys played blind and of the shadow."""
    track = json.loads((ROOT / "real" / "tracks" / f"{stem}.json").read_text())
    keys = rs.overlay_keys(track["video"], track["first"], track["rows"][-1]["frame"] + 1)
    out = {}
    step = rs.Shadow.step
    try:
        for name, correct in (("blind", False), ("shadow", True)):
            pts = []

            def logged(self, action, pts=pts):
                step(self, action)
                pts.append((self.g.player.position.x, self.g.player.position.y, self.g.alive))
            rs.Shadow.step = logged
            rs.video_test(track, keys, track["first"] + 7, correct=correct)
            out[name] = pts
    finally:
        rs.Shadow.step = step
    out["real"] = [(*r["steve"]["feet_world"], True) for r in track["rows"] if r.get("steve") and all(r["used"])]
    return out


def draw(p: dict, out: Path, title: str):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    import sim.play as play
    from sim.world import PX_PER_M
    every = np.array([(x, y) for k in p for x, y, _ in p[k]])
    (x0, y0), (x1, y1) = every.min(0) - (5, 4), every.max(0) + (5, 4)
    play.W, play.H = int((x1 - x0) * PPM), int((y1 - y0) * PPM)
    play.ZOOMS = (PPM / (play.H / play.GAME_VIEW_H * PX_PER_M),)
    app = play.App(argparse.Namespace(agent=None, race=False, loop=False, overlay=False, map="Tower", replay=None))
    view = app.views[0]
    app.cam = [(x0 + x1) / 2, (y0 + y1) / 2]
    view.draw_scene()
    sc = view.screen

    def line(pts, colour, width, edge=None):
        segments, seg, last = [], [], None
        for x, y, alive in pts:  # a death, or a stretch the screen didn't pin down, breaks the line
            if seg and (not alive or np.hypot(x - last[0], y - last[1]) > 1.0):
                segments.append(seg)
                seg = []
            if alive:
                seg.append(view.to_screen(x, y))
            last = (x, y)
        segments.append(seg)
        for s in segments:
            if len(s) > 1:
                if edge:
                    pygame.draw.lines(sc, edge, False, s, width + 2)
                pygame.draw.lines(sc, colour, False, s, width)
    line(p["real"], REAL, 5, (20, 20, 30))
    line(p["shadow"], SHADOW, 2)
    line(p["blind"], BLIND, 3)
    bx, by, _ = p["blind"][-1]
    ex, ey = view.to_screen(bx, by)
    pygame.draw.line(sc, BLIND, (ex - 9, ey - 9), (ex + 9, ey + 9), 4)
    pygame.draw.line(sc, BLIND, (ex - 9, ey + 9), (ex + 9, ey - 9), 4)
    rows = [(REAL, "your run, tracked from the screen"), (SHADOW, "the shadow: the sim, corrected from the screen"),
            (BLIND, "the same keys played blind in the sim")]
    wide = max(app.font.size(t)[0] for _, t in rows) + 76
    box = pygame.Surface((wide, 40 + 24 * len(rows)), pygame.SRCALPHA)
    box.fill((0, 0, 0, 160))
    top = view.to_screen(0, 22.6)[1]  # in the open sky left of the tower
    sc.blit(box, (16, top))
    sc.blit(app.font.render(title, True, (255, 255, 255)), (28, top + 8))
    for i, (colour, text) in enumerate(rows):
        y = top + 40 + 24 * i
        pygame.draw.line(sc, colour, (28, y + 8), (68, y + 8), 5)
        sc.blit(app.font.render(text, True, (230, 230, 230)), (80, y))
    pygame.image.save(sc, str(out))
    pygame.quit()
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="2026-10-02 23-41-14", help="the recording's name (its track in real/tracks/)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "shadow_vs_blind.png"))
    args = ap.parse_args()
    draw(paths(args.video), Path(args.out), f"Tower, your run of {args.video}")


if __name__ == "__main__":
    main()
