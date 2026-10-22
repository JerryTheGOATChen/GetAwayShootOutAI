"""Play runs blind into the real game, many times, and record where the player went: for measuring how long each
run stays on the path it took in the sim (real.trials_report compares them).

    py -m real.trials runs/02-your-run/ppo_tower.pt runs/04-jitter/ppo_tower.pt runs/pb_Tower.json --trials 10

Have the game's Tower time trial open full screen, in a run (any point of it: it restarts it from the pause menu)
with Standard Steve. The runs take turns, so a slow change in the game or the PC affects them all alike. Each trial
restarts the run, waits for the drop-in landing, plays the run's key presses at the times it made them in the sim,
and records the player's jeans (their screen position, ~30 a second) until the run's sim time is up. Keys and
clicks only ever go to the game: if another window comes to the front, the trial is dropped. X stops.

Saved in real/trials/<date time>/: one JSON per trial, a few small frames of each for checking by eye.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from real import window
from real.replay import CLOCK_AFTER_LANDING, KEY_LEAD, SCAN, key, schedule, sim_run, stop_pressed
from real.screen import Grabber
from sim.world import DT

OUT = Path(__file__).resolve().parent / "trials"
PAUSE_BUTTON, RESTART_BUTTON, OUT_OF_THE_WAY = (51, 96), (956, 804), (1915, 600)
JEANS_BGR = np.array([203, 123, 75])  # Standard Steve's jeans
JEANS_TOLERANCE = np.array([30, 30, 35])
SKIN_BGR = np.array([170, 222, 241])  # his head (and hand)
SKIN_TOLERANCE = np.array([14, 14, 14])


def run_events(path: str):
    """(time after the start, action) for each key change, how long the run lasts, and its keys one per tick (as
    the sim plays it now)."""
    from real.robustness import run_ticks
    ticks = run_ticks(path)
    return schedule(ticks, DT), len(ticks) * DT, ticks


# --- watching the screen --------------------------------------------------------------------------------

def jeans_blobs(img, colour=JEANS_BGR, tolerance=JEANS_TOLERANCE, half=None) -> list[tuple[float, float, int]]:
    """Every patch of jeans colour (or another colour) on the screen: (x, y, area), in full-size screen pixels,
    largest first. Worked out at half size, for speed (half: the grab at half size already, img[::2, ::2])."""
    small = half if half is not None else np.ascontiguousarray(img[::2, ::2])
    mask = cv2.inRange(small, np.clip(colour - tolerance + 1, 0, 255).astype(np.uint8),
                       np.clip(colour + tolerance - 1, 0, 255).astype(np.uint8))
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [(float(centres[i][0] * 2), float(centres[i][1] * 2), int(stats[i][4] * 4)) for i in range(1, n)
             if stats[i][4] >= 15]
    return sorted(blobs, key=lambda b: -b[2])


class Watcher(threading.Thread):
    """Grabs the screen as fast as it can (~30 a second) and keeps the jeans patches of every grab."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list[tuple[float, list, list]] = []  # (time, jeans blobs, skin blobs)
        self.frames: list[tuple[float, np.ndarray]] = []  # a few small copies, for checking by eye
        self.keep_every = 0.0  # seconds between kept frames (0: none)
        self.clip = None  # a real.clip.Clip filming every grab
        self.running = True
        self._last_kept = 0.0

    def run(self):
        grabber = Grabber()
        try:
            while self.running:
                t0 = time.perf_counter()
                img = grabber.grab()
                t = (t0 + time.perf_counter()) / 2
                self.samples.append((t, jeans_blobs(img), jeans_blobs(img, SKIN_BGR, SKIN_TOLERANCE)[:4]))
                if self.clip is not None:
                    self.clip.add(t, img)
                if self.keep_every and t - self._last_kept >= self.keep_every:
                    self.frames.append((t, cv2.resize(img, (480, 300), interpolation=cv2.INTER_AREA)))
                    self._last_kept = t
        finally:
            grabber.close()

    def since(self, t0):
        return [s for s in self.samples if s[0] >= t0]


def landing(samples, h=1200, newest=8) -> float | None:
    """When the game's clock started, from the drop-in (see real.replay.wait_for_landing): the player was off screen
    for a while, fell over a quarter of the screen in under half a second, and stopped. Only patches in the middle
    of the screen count. Checks the newest few grabs (it is called every few hundredths of a second)."""
    seen = []
    for t, blobs, *_ in samples:
        mid = [b for b in blobs if 0.3 * 1920 < b[0] < 0.7 * 1920 and b[2] >= 300]
        seen.append((t, mid[0][1] if mid else None))
    for k in range(max(1, len(seen) - newest), len(seen)):
        t, y = seen[k]
        window_ = [s for s in seen[:k] if s[0] > t - 3.0]
        visible = [(ts, ys) for ts, ys in window_ if ys is not None]
        if y is None or not visible or abs(y - visible[-1][1]) >= 2:
            continue
        fell = any(ys < y - 0.25 * h for ts, ys in visible if ts > t - 0.4)
        gone = sum(b[0] - a[0] for a, b in zip(window_ + [(t, y)], (window_ + [(t, y)])[1:])
                   if a[1] is None and a[0] > t - 2.5)
        if fell and gone > 0.3:
            moving = [s for s in visible if abs(s[1] - y) >= 2]
            first_still = next(ts for ts, ys in visible + [(t, y)] if ts > moving[-1][0])
            reached = first_still
            if len(moving) >= 2:
                (ta, ya), (tb, yb) = moving[-2:]
                v = (yb - ya) / (tb - ta)
                if v > 0:
                    reached = min(tb + (y - yb) / v, first_still)
            return reached + CLOCK_AFTER_LANDING
    return None


# --- one trial ------------------------------------------------------------------------------------------

def restart():
    window.click(*PAUSE_BUTTON)
    time.sleep(0.7)
    window.click(*RESTART_BUTTON)
    window.user32.SetCursorPos(*OUT_OF_THE_WAY)


def play(events, start) -> list[tuple[float, int]]:
    """Press the run's keys at their times after the start. Returns when each change was actually sent."""
    held, sent = None, []
    try:
        for at, a in events:
            target = start + at - KEY_LEAD
            while (now := time.perf_counter()) < target - 0.003:
                time.sleep(0.001)
                if stop_pressed():
                    raise KeyboardInterrupt
            while time.perf_counter() < target:
                pass
            if not window.game_in_front():
                raise window.NotInFront("the game left the front mid-run")
            if held in SCAN:
                key(held, False)
            if a in SCAN:
                key(a, True)
            held = a
            sent.append((time.perf_counter() - start + KEY_LEAD, a))  # the time it was meant for, give or take
        return sent
    finally:
        if held in SCAN:
            key(held, False)


def trial(watch: Watcher, events, length_s, keep_frames: bool, offset: float = 0.0) -> dict | None:
    hwnd = window.find_game()
    if hwnd is None or not window.bring_to_front(hwnd):
        print("  the game isn't in front; waiting", flush=True)
        time.sleep(5)
        return None
    t_restart = time.perf_counter()
    restart()
    start = None
    while start is None and time.perf_counter() - t_restart < 12:
        time.sleep(0.03)
        if stop_pressed():
            raise KeyboardInterrupt
        start = landing(watch.since(t_restart))
    if start is None:
        print("  no drop-in seen", flush=True)
        return None
    start += offset  # where the game's clock really starts, if not at the estimate (times are kept from here)
    watch.keep_every = 0.5 if keep_frames else 0.0
    sent = play([e for e in events if e[0] < length_s], start)
    while time.perf_counter() < start + length_s + 1.5:
        time.sleep(0.01)
    watch.keep_every = 0.0
    return {"start": start, "offset": offset, "sent": sent,
            "samples": [(t - start, b[:4], *rest) for t, b, *rest in watch.since(start - 3.0)]}


def be_on_time():
    """Run at high priority with a 1 ms timer, and stop Windows from slowing this process or sleeping the screen."""
    from ctypes import wintypes
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE  # 64-bit handles: undeclared, ctypes would cut them to 32 bits
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.SetPriorityClass(k.GetCurrentProcess(), 0x80)  # HIGH_PRIORITY_CLASS
    ctypes.windll.winmm.timeBeginPeriod(1)
    k.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)  # continuous, system and display on

    class PPTS(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
    s = PPTS(1, 0x1 | 0x4, 0)
    k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))
    sys.setswitchinterval(0.001)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="trained agents (.pt) or recorded runs (.json)")
    ap.add_argument("--trials", type=int, default=10, help="trials of each run")
    ap.add_argument("--max-s", type=float, default=40.0, help="stop recording a trial after this long")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds after the estimated start (drop-in landing + %g s) that the run starts" % CLOCK_AFTER_LANDING)
    args = ap.parse_args()
    be_on_time()
    folder = OUT / time.strftime("%Y-%m-%d %H-%M-%S")
    folder.mkdir(parents=True)
    runs = {}
    for path in args.runs:
        events, length, ticks = run_events(path)
        runs[path] = (events, min(length, args.max_s), ticks)
        print(f"{path}: {len(events)} key changes over {length:.2f} s", flush=True)
    watch = Watcher()
    watch.start()
    done = dict.fromkeys(runs, 0)
    try:
        while min(done.values()) < args.trials:
            for path, (events, length, ticks) in runs.items():
                if done[path] >= args.trials:
                    continue
                n = done[path]
                try:
                    result = trial(watch, events, length, keep_frames=n < 3, offset=args.offset)
                except window.NotInFront as e:
                    print(f"  dropped: {e}", flush=True)
                    result = None
                watch.samples = watch.samples[-200:]  # only the recent past is needed between trials
                if result is None:
                    continue
                name = f"{Path(path).parent.name}_{Path(path).stem}_{n:02d}"
                (folder / f"{name}.json").write_text(json.dumps({"run": path, "length_s": length, "ticks": ticks, **result}))
                for i, (t, img) in enumerate(watch.frames):
                    cv2.imwrite(str(folder / f"{name}_{t - result['start']:05.1f}s.jpg"), img,
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
                watch.frames = []
                done[path] += 1
                print(f"{time.strftime('%H:%M:%S')} {name}: done", flush=True)
    except KeyboardInterrupt:
        print("stopped.")
    finally:
        watch.running = False
    print(f"saved in {folder}")


if __name__ == "__main__":
    main()
