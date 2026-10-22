"""Play a trained agent in the real Getaway Shootout, open loop: its key presses from the sim, at the same timing.

    py -m real.replay runs/03-speed/ppo_tower.pt              # press W / E in the game
    py -m real.replay runs/03-speed/ppo_tower.pt --watch      # only check the start is detected; no keys
    py -m real.replay runs/03-speed/ppo_tower.pt --offset 0.03

How to use: open the game's Tower speedrun full screen, run this, click into the game so it has the keyboard,
and start (or restart) a run. The clock starts the moment your character lands from the drop-in, so this watches
the screen for that landing and then plays the agent's presses from it. X stops it. After each run it waits for
the next one. The drop-in has to be on screen: if you miss it, just restart the run.

The agent can't see the real game: it plays the run it played in the sim. Small differences (a key press landing a
frame later, since the game reads keys 60 times a second and its physics runs 50; a lift that leaves a little
earlier) add up, and the run goes its own way from there. In tests it drifted from about 2 s in.
--offset shifts every press later (positive) or earlier (negative), in seconds, to line up the start.
"""
from __future__ import annotations

import argparse
import ctypes
import time
from ctypes import wintypes

import numpy as np
from PIL import ImageGrab

from common.policy import load_agent
from common.workers import BATCHES, TRAIN_ENV, agent_env
from sim.env import GetawayEnv
from sim.world import DT, LEFT, RIGHT

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()  # screen coordinates in real pixels, not the scaled ones

# --- the keyboard ---------------------------------------------------------------------------------------

SCAN = {LEFT: 0x11, RIGHT: 0x12}  # W leans left, E leans right (scan codes, which games read reliably)
KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, VK_STOP = 0x0002, 0x0008, 0x58  # X stops (no Fn needed, the game doesn't use it)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("i",)
    _fields_ = [("type", wintypes.DWORD), ("i", _I)]


def key(action: int, down: bool):
    flags = KEYEVENTF_SCANCODE | (0 if down else KEYEVENTF_KEYUP)
    inp = INPUT(type=1, ki=KEYBDINPUT(0, SCAN[action], flags, 0, None))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def stop_pressed() -> bool:
    """X stops (not Esc: the game uses Esc for its pause menu, where you restart)."""
    return bool(user32.GetAsyncKeyState(VK_STOP) & 0x8000)


# --- the agent's run in the sim -------------------------------------------------------------------------

def sim_run(path):
    """The agent's actions (one per 0.04 s decision) and its checkpoint times, from its run in the sim."""
    policy, info = load_agent(path)
    batch = BATCHES.get(path.replace("\\", "/").split("/")[-2])
    env = GetawayEnv(**agent_env(info, stall_limit_s=None, **({"rewards": batch["rewards"]} if batch else {})))
    obs, _ = env.reset()
    actions = []
    while True:
        actions.append(policy.act(obs))
        obs, _, terminated, truncated, result = env.step(actions[-1])
        if terminated or truncated:
            break
    return actions, [t * DT for t in env.game.splits], result


def schedule(actions, step_s):
    """(time after the start, action) wherever the held key changes."""
    events, held = [], None
    for i, a in enumerate(actions):
        if a != held:
            events.append((i * step_s, a))
            held = a
    return events


# --- spotting the start ---------------------------------------------------------------------------------

# The game takes keys (the sim's tick 0) from about 0.16 s after the jeans stop falling. The clock shows zero
# sooner (about 2.5 frames after the landing, in a recording of a full run: 2026-10-02 23-41-14), but a lean
# pressed then only starts 0.12 s later: in probes leaning from the start (real/probes/stand_*), sending the
# keys 0.12 s later made each lean start on time and match the sim's (2026-10-20).
CLOCK_AFTER_LANDING = 0.16
# and the game acts on a key about 0.04 s (2 ticks) after it is sent, so keys go that much early: real-game trials
# followed their sim runs 2-10 s longer with it (real.physics_check, keys_shifted=2, 2026-10-20)
KEY_LEAD = 0.04

def jeans_y(img):
    """Screen row of the player's jeans (the drop-in starts mid screen), or None."""
    a = img.astype(np.int16)
    m = (abs(a[..., 0] - 75) < 35) & (abs(a[..., 1] - 123) < 30) & (abs(a[..., 2] - 203) < 30)
    if m.sum() < 300:
        return None
    return float(np.nonzero(m)[0].mean())


def wait_for_landing():
    """Watch the middle of the screen until the player drops in and lands; returns when the game's clock started
    (time.perf_counter), estimated between screen grabs from the speed of the fall.

    Only the drop-in counts: the player was off screen for a while just before, then fell over a quarter of the
    screen in under half a second, and stopped. During play the player is always on screen, so a fall then is
    ignored."""
    w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    box = (w // 2 - w // 5, 0, w // 2 + w // 5, h)
    seen = []  # (time, jeans row or None) over the last few seconds
    while not stop_pressed():
        t0 = time.perf_counter()
        y = jeans_y(np.asarray(ImageGrab.grab(bbox=box)))
        t = (t0 + time.perf_counter()) / 2
        seen = [s for s in seen if s[0] > t - 3.0] + [(t, y)]
        visible = [(ts, ys) for ts, ys in seen[:-1] if ys is not None]
        if y is None or not visible or abs(y - visible[-1][1]) >= 2:
            continue  # not stopped
        fell = any(ys < y - 0.25 * h for ts, ys in visible if ts > t - 0.4)
        # time spent off screen in the 2.5 s before: each missing grab counts until the next grab
        gone = sum(b[0] - a[0] for a, b in zip(seen, seen[1:]) if a[1] is None and a[0] > t - 2.5)
        if fell and gone > 0.3:
            # it landed between the last grab still falling and the first one on the ground: estimate where from
            # the falling speed
            moving = [s for s in visible if abs(s[1] - y) >= 2]
            first_still = next(ts for ts, ys in visible + [(t, y)] if ts > moving[-1][0])
            reached = first_still
            if len(moving) >= 2:
                (ta, ya), (tb, yb) = moving[-2:]
                v = (yb - ya) / (tb - ta)  # px/s
                if v > 0:
                    reached = min(tb + (y - yb) / v, first_still)
            return reached + CLOCK_AFTER_LANDING
    return None


# --- playing it ---------------------------------------------------------------------------------------

def play(events, start, offset):
    held = None
    try:
        for at, a in events:
            target = start + offset + at - KEY_LEAD
            while time.perf_counter() < target:
                if stop_pressed():
                    return False
            if held in SCAN:
                key(held, False)
            if a in SCAN:
                key(a, True)
            held = a
        return True
    finally:
        if held in SCAN:
            key(held, False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="a trained agent, e.g. runs/03-speed/ppo_tower.pt")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds to shift every press (+ later, - earlier)")
    ap.add_argument("--watch", action="store_true", help="only report each detected start; press no keys")
    args = ap.parse_args()

    actions, splits, result = sim_run(args.agent)
    events = schedule(actions, TRAIN_ENV["action_repeat"] * DT)
    print(f"in the sim this agent {'finishes in %.2f s' % result['time_s'] if result['finished'] else 'does not finish'}"
          f" ({len(events)} key changes). Checkpoints at: {', '.join(f'{s:.2f}' for s in splits)} s")
    print("Click into the game (full screen) and start or restart a Tower speedrun; the drop-in must be on screen. X stops.")
    while True:
        start = wait_for_landing()
        if start is None:
            break
        print(f"run started at {time.strftime('%H:%M:%S')}" + (" (watch only)" if args.watch else ": playing the agent"))
        if not args.watch and not play(events, start, args.offset):
            break
        if args.watch:
            time.sleep(3)
    print("stopped.")


if __name__ == "__main__":
    main()
