"""Video clips: an agent in the real game, its run in the sim (in the sim's own look), and the two side by side.

    py -m real.clip blind runs/03-speed/ppo_tower.pt     # play the agent's sim run blind in the real game, filmed
    py -m real.clip sim runs/03-speed/ppo_tower.pt       # the agent's run in the sim, filmed in the viewer
    py -m real.clip pair "real/clips/<clip>.clip.json" runs/03-speed/ppo_tower.pt   # side by side, from the start
    py -m real.clip pair "real/live/<date time>/run1.clip.json" --shadow    # a closed-loop run beside its shadow
    py -m real.clip gif "real/clips/<clip>.mp4" --from 0 --to 8             # a short GIF of a clip, for the README

Have the game's Tower time trial open full screen (1920x1080) for `blind`, as for real.trials. Clips go in
real/clips/; each film of the real game has a .clip.json next to it saying when the run's clock started in it,
which `pair` lines the sim up with. real.live --clip films its closed-loop runs the same way, and --shadow draws
the shadow sim the agent played from (from the run's log) in place of a sim run.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "clips"
FPS = 30
SIZE = (960, 540)  # each side of a clip
LEAD_S = 1.0  # a pair starts this long before the run's clock does (the drop-in)


class Clip(threading.Thread):
    """Writes screen grabs to an mp4 at a steady FPS on its own thread: whoever grabs only hands each grab over
    (one that comes while the last is still waiting is skipped). Each frame shows the newest grab at its time."""

    def __init__(self, path: Path, fps: int = FPS, size=SIZE):
        super().__init__(daemon=True)
        self.path, self.fps, self.size = Path(path), fps, size
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.t0: float | None = None  # perf_counter of the first frame
        self.frames = 0
        self.start()

    def add(self, t: float, img: np.ndarray):
        """A grab taken at t (perf_counter)."""
        try:
            self.queue.put_nowait((t, cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)))
        except queue.Full:
            pass

    def run(self):
        out = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"avc1"), self.fps, self.size)
        last = None
        while (item := self.queue.get()) is not None:
            t, img = item
            if self.t0 is None:
                self.t0 = t
            while last is not None and self.t0 + self.frames / self.fps < t:
                out.write(last)
                self.frames += 1
            last = img
        if last is not None:
            out.write(last)
            self.frames += 1
        out.release()

    def close(self, **about):
        """Finish the file, and save what it shows next to it (about: e.g. start, the run's clock in perf_counter)."""
        self.queue.put(None)
        self.join()
        self.path.with_suffix(".clip.json").write_text(json.dumps({"video": self.path.name, "t0": self.t0, "fps": self.fps,
                                                              "frames": self.frames, **about}))


# --- the sim's side ---------------------------------------------------------------------------------------

def sim_frames(agent: str, fps: int = FPS, lead_s: float = 0.0, seconds: float = 90.0, size=SIZE):
    """The agent's run in the sim, drawn by the viewer (sim.play) off screen: BGR frames, fps a second, from lead_s
    before the start (showing the start) until it is over, plus a second for the result."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from sim.play import App
    from sim.world import DT
    app = App(argparse.Namespace(agent=[agent], race=False, loop=False, overlay=False, map="Tower", replay=None))
    view = app.views[0]
    ticks, after = 0, 0
    for k in range(int((lead_s + seconds) * fps)):
        want = round((k / fps - lead_s) / DT)
        while ticks < want and not view.done:
            view.advance()
            app.follow(DT)
            ticks += 1
        if view.done:
            after += 1
            if after > fps:
                break
        app.draw()
        img = pygame.surfarray.array3d(app.screen).transpose(1, 0, 2)[..., ::-1]
        yield cv2.resize(np.ascontiguousarray(img), size, interpolation=cv2.INTER_AREA)
    pygame.quit()


def shadow_frames(log_path: Path, fps: int = FPS, lead_s: float = 0.0, size=SIZE):
    """A closed-loop run's shadow (real.live's log: the shadow every other tick), drawn by the viewer as the sim:
    what the agent saw."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from sim.play import App
    from sim.world import DT
    log = json.loads(Path(log_path).read_text())
    film = log["film"]
    app = App(argparse.Namespace(agent=None, race=False, loop=False, overlay=False, map="Tower", replay=None))
    view = app.views[0]
    if log.get("agent"):
        from common.policy import load_agent
        view.agent_info = load_agent(log["agent"])[1]
    i, tick = 0, 0
    for k in range(int((lead_s + film[-1][0] * DT) * fps) + fps):
        want = round((k / fps - lead_s) / DT)
        while tick < want:  # the camera eases along tick by tick, as in the viewer
            tick += 1
            while i + 1 < len(film) and film[i + 1][0] <= tick:
                i += 1
            if film[i][0] <= tick:
                view.g.restore(film[i][2])
                view.action = film[i][1]
            app.follow(DT)
        view.done = tick >= film[-1][0]
        view.result = "GETAWAY!" if view.g.finished else "(stopped)"
        app.draw()
        img = pygame.surfarray.array3d(app.screen).transpose(1, 0, 2)[..., ::-1]
        yield cv2.resize(np.ascontiguousarray(img), size, interpolation=cv2.INTER_AREA)
    pygame.quit()


# --- putting them together --------------------------------------------------------------------------------

def label(img: np.ndarray, text: str):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x, y = img.shape[1] - w - 24, img.shape[0] - 20
    cv2.rectangle(img, (x - 10, y - h - 10), (x + w + 10, y + 10), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def pair(real_json: Path, agent: str | None, out: Path, real_label: str = "the real game", sim_label: str = "the sim"):
    """The film of the real game beside the agent's run in the sim (agent None: beside the shadow the agent played
    from, for a closed-loop run), both from LEAD_S before the run's start."""
    about = json.loads(Path(real_json).read_text())
    cap = cv2.VideoCapture(str(Path(real_json).with_name(about["video"])))
    fps = about["fps"]
    first = int(round((about["start"] - LEAD_S - about["t0"]) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(first, 0))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (SIZE[0] * 2, SIZE[1]))
    real = None
    n = 0
    log = Path(real_json).with_name(Path(about["video"]).stem + ".json")
    frames = sim_frames(agent, fps, LEAD_S) if agent else shadow_frames(log, fps, LEAD_S)
    for sim in frames:
        ok, frame = cap.read()
        if ok:
            real = frame
        if real is None:
            break
        both = np.hstack([real, sim])
        label(both[:, :SIZE[0]], real_label)
        label(both[:, SIZE[0]:], sim_label)
        writer.write(both)
        n += 1
    writer.release()
    cap.release()
    print(f"{n} frames ({n / fps:.1f} s) -> {out}")


def gif(video: Path, out: Path, start_s: float, end_s: float, width: int = 720, fps: int = 15, speed: float = 1.0):
    """A stretch of a clip as a looping GIF, `speed` times as fast as it happened."""
    from PIL import Image
    cap = cv2.VideoCapture(str(video))
    src = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src))
    frames, k = [], 0
    while k < (end_s - start_s) * src:
        ok, img = cap.read()
        if not ok:
            break
        if k % max(1, round(src * speed / fps)) == 0:
            h = int(img.shape[0] * width / img.shape[1])
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA),
                                                       cv2.COLOR_BGR2RGB)).quantize(128))
        k += 1
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


# --- filming a blind run ----------------------------------------------------------------------------------

def blind(agent: str, length_s: float) -> Path:
    """Restart the run, play the agent's sim run blind (as real.trials does) and film it."""
    from real import window
    from real.replay import stop_pressed
    from real.trials import Watcher, be_on_time, landing, play, restart, run_events
    be_on_time()
    events, length, _ = run_events(agent)
    length = min(length, length_s)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{time.strftime('%Y-%m-%d %H-%M-%S')} blind {Path(agent).parent.name} {Path(agent).stem}.mp4"
    hwnd = window.find_game()
    if hwnd is None or not window.bring_to_front(hwnd):
        raise SystemExit("the game isn't in front")
    watch = Watcher()
    watch.start()
    clip = Clip(path)
    watch.clip = clip
    start = None
    try:
        t_restart = time.perf_counter()
        restart()
        while start is None and time.perf_counter() - t_restart < 12:
            time.sleep(0.03)
            if stop_pressed():
                raise KeyboardInterrupt
            start = landing(watch.since(t_restart))
        if start is None:
            raise SystemExit("no drop-in seen")
        play([e for e in events if e[0] < length], start)
        while time.perf_counter() < start + length + 1.5:
            time.sleep(0.01)
    finally:
        watch.running = False
        watch.join(timeout=2)
        clip.close(start=start, agent=agent, kind="blind", length_s=length)
    print(f"filmed -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    b = sub.add_parser("blind", help="film the agent's sim run played blind in the real game")
    b.add_argument("agent")
    b.add_argument("--length", type=float, default=40.0, help="seconds of the run to play at most")
    s = sub.add_parser("sim", help="film the agent's run in the sim")
    s.add_argument("agent")
    p = sub.add_parser("pair", help="a film of the real game beside the agent's run in the sim")
    p.add_argument("real", help="the film's .clip.json")
    p.add_argument("agent", nargs="?")
    p.add_argument("--shadow", action="store_true", help="a closed-loop run (real.live --clip): beside its shadow")
    p.add_argument("--out")
    g = sub.add_parser("gif", help="a short GIF of a clip")
    g.add_argument("video")
    g.add_argument("--from", dest="start", type=float, default=0.0)
    g.add_argument("--to", dest="end", type=float, default=8.0)
    g.add_argument("--width", type=int, default=720)
    g.add_argument("--speed", type=float, default=1.0, help="play this many times as fast")
    g.add_argument("--out")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.what == "blind":
        blind(args.agent, args.length)
    elif args.what == "sim":
        out = OUT / f"sim {Path(args.agent).parent.name} {Path(args.agent).stem}.mp4"
        w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, SIZE)
        for img in sim_frames(args.agent):
            w.write(img)
        w.release()
        print(f"-> {out}")
    elif args.what == "pair":
        real = Path(args.real)
        if not args.agent and not args.shadow:
            ap.error("name the agent whose sim run to show, or --shadow")
        out = Path(args.out) if args.out else real.with_name(real.name.replace(".clip.json", " pair.mp4"))
        pair(real, None if args.shadow else args.agent, out,
             sim_label="the shadow: what the agent sees" if args.shadow else "the sim")
    else:
        video = Path(args.video)
        gif(video, Path(args.out) if args.out else video.with_suffix(".gif"), args.start, args.end, args.width, speed=args.speed)


if __name__ == "__main__":
    main()
