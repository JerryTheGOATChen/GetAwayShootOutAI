"""Play an agent in the real game closed loop. The agent plays from a shadow sim (real.shadow) that is kept on the
real run by watching the screen (real.locate), so it reacts to what really happens instead of replaying a sim run
blind (which the real game can't repeat for more than about 14 s).

    py -m real.live runs/04-jitter/ppo_tower.pt            # restart the run, play it, record it
    py -m real.live --watch                                # no keys sent: follows a run you play (W / E), to
                                                           # check the tracking; start or restart a run yourself
    py -m real.live runs/05-shadow/ppo_tower.pt --clip     # and film each run (real.clip)
    py -m real.live --calibrate                            # measure the screen first (restarts, sends no keys)
    py -m real.live runs/08-late/ppo_tower.pt --late 3     # stand still 3 s first: the lifts come at other times

Have the game's Tower time trial open full screen (1920x1080), with Standard Steve, in a run (it restarts it from the
pause menu). Keys and clicks only ever go to the game: if another window comes to the front it stops. X stops.

--calibrate restarts the run and leaves Steve standing at the start for a few seconds: from full-size grabs of him
there it finds the screen's scale (px per m, where the art matches best) and how far above his feet the camera
rests, and saves them in real/live/screen.json for every run after (on a 1920x1200 screen they differ from the
1920x1080 recordings real.locate was fitted on: about 120 px/m against 113.3).

Each run is saved in real/live/<date time>/log.json: every grab (where the camera, Steve and the lifts were seen),
how far the shadow was from it, and every key, so a run can be gone through afterwards.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from real import window
from real.locate import (CAM_ABOVE_FEET, DEATH_SCORE, MATCH_PPM, SCREEN_PX_PER_M, Level, Tracker, grey, hidden,
                          lift_schedule, respawn_points, sim_object_guess, steve_on_screen)
from real.replay import KEY_LEAD, SCAN, key, stop_pressed
from real.screen import Grabber
from real.shadow import Shadow
from real.trials import OUT_OF_THE_WAY, PAUSE_BUTTON, RESTART_BUTTON, be_on_time, jeans_blobs, landing
from sim.world import DT, LEFT, NONE, RIGHT

OUT = Path(__file__).resolve().parent / "live"
SCREEN = OUT / "screen.json"  # the screen's scale and the camera's rest, from --calibrate
GRAB_LAG = 0.02  # s: a grab shows the game about this long before it was taken (frame + composition)
SEARCH_FIRST = 1.5  # m: the camera search around the start for the first grabs
STAY_FINISHED_S = 3.0  # the shadow finished: keep playing this long in case the grabs say the real run hasn't
VK_W, VK_E = 0x57, 0x45


class Eyes(threading.Thread):
    """Grabs the screen as fast as it can. Before the run starts, keeps Steve's jeans (to spot the drop-in);
    after, locates everything in each grab (real.locate) and hands it over."""

    def __init__(self, px_per_m: float = SCREEN_PX_PER_M, cam_above: float = CAM_ABOVE_FEET):
        super().__init__(daemon=True)
        self.px_per_m, self.cam_above = px_per_m, cam_above
        self.level = Level("Tower")
        self.schedule = lift_schedule("Tower")
        from sim.world import Game
        self.anchors = respawn_points(Game("Tower"))  # where the tracker looks for Steve when lost (after a death)
        self.steve_hint = None  # where the shadow has Steve's feet (world m), set by the main loop
        self.samples: list[tuple[float, list]] = []
        self.seen: list[tuple[float, dict]] = []  # (grab time, what real.locate saw), taken by the main loop
        self.lock = threading.Lock()
        self.tracker: Tracker | None = None
        self.clock: float | None = None  # perf_counter when the run's clock started
        self.clip = None  # a real.clip.Clip filming every grab
        self.running = True
        self.error: Exception | None = None

    def begin(self, start: float, cam):
        """The run's clock started at `start` (perf_counter) with the camera at cam: from now on, locate things."""
        tracker = Tracker(self.level, cam, self.px_per_m, anchors=self.anchors, cam_above=self.cam_above)
        with self.lock:
            self.tracker, self.clock = tracker, start

    def run(self):
        grabber = Grabber()
        last = None
        try:
            while self.running:
                t0 = time.perf_counter()
                img = grabber.grab()
                t = (t0 + time.perf_counter()) / 2
                if self.clip is not None:
                    self.clip.add(t, img)
                with self.lock:
                    tracker, start = self.tracker, self.clock
                if tracker is None:
                    self.samples.append((t, jeans_blobs(img)))
                    continue
                since = t - GRAB_LAG - start
                row = tracker.update(img, (t - last) if last else 1 / 30,
                                     sim_object_guess(self.level, self.schedule, max(since, 0.0)), self.steve_hint)
                last = t
                with self.lock:
                    self.seen.append((t, row))
        except Exception as e:  # noqa: BLE001 - reported by the main loop
            self.error = e
        finally:
            grabber.close()

    def take(self) -> list[tuple[float, dict]]:
        with self.lock:
            out, self.seen = self.seen, []
        return out


def restart():
    window.click(*PAUSE_BUTTON)
    time.sleep(0.7)
    window.click(*RESTART_BUTTON)
    window.user32.SetCursorPos(*OUT_OF_THE_WAY)


def wait_for_start(eyes: Eyes, since: float, limit_s: float) -> float | None:
    while time.perf_counter() - since < limit_s:
        time.sleep(0.03)
        if stop_pressed():
            raise KeyboardInterrupt
        start = landing([(t, b) for t, b in eyes.samples if t >= since])
        if start is not None:
            return start
    return None


def screen_scale() -> tuple[float, float]:
    """(px per m, camera rest above the feet in m) for this screen, from --calibrate (else real.locate's, fitted on
    1920x1080 recordings)."""
    from real.screen import screen_size
    if SCREEN.exists():
        cal = json.loads(SCREEN.read_text())
        if list(cal["screen"]) == list(screen_size()):
            return cal["px_per_m"], cal["cam_above_feet"]
        print(f"real/live/screen.json is for a {cal['screen']} screen, not this one: run --calibrate")
    else:
        print("the screen isn't calibrated (py -m real.live --calibrate): using the recordings' scale")
    return SCREEN_PX_PER_M, CAM_ABOVE_FEET


def measure_screen(frames: list, feet) -> dict:
    """The screen's scale and the camera's rest, from full-size grabs of Steve standing with his feet at `feet`."""
    from real.screen import screen_size
    level = Level("Tower")
    guess = (feet[0], feet[1] + CAM_ABOVE_FEET)
    scales = np.arange(110.0, 130.01, 0.5)
    scores, cams = [], []
    for pxm in scales:
        tracker = Tracker(level, guess, pxm)
        sc, cs = [], []
        for img in frames:
            s = MATCH_PPM / pxm
            small = grey(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            look = hidden(small.shape[1], small.shape[0], steve_on_screen(img, pxm), s)
            cam, score, _, used, _ = tracker._match(small, look, guess, None, 0.8)
            sc.append(score if all(used) else 0.0)
            cs.append(cam)
        scores.append(np.mean(sc))
        cams.append(np.mean(cs, axis=0))
    i = int(np.argmax(scores))
    pxm = float(scales[i])
    if 0 < i < len(scales) - 1:  # between the steps
        l, m, r = scores[i - 1], scores[i], scores[i + 1]
        pxm += 0.5 * 0.5 * (l - r) / (l - 2 * m + r) if l - 2 * m + r < 0 else 0.0
    cam = cams[i]
    cal = {"screen": list(screen_size()), "px_per_m": round(pxm, 2), "cam_above_feet": round(float(cam[1] - feet[1]), 3),
           "cam_x_off": round(float(cam[0] - feet[0]), 3), "frames": len(frames), "measured": time.strftime("%Y-%m-%d %H:%M"),
           "match": round(float(scores[i]), 4), "scores": {f"{s:.1f}": round(float(v), 4) for s, v in zip(scales, scores)}}
    return cal


def calibrate(seconds: float = 3.0):
    """Restart the run, send no keys, and measure the screen from grabs of Steve standing at the start (see the top)."""
    from sim.world import Game
    g = Game("Tower")
    g.reset()
    feet = (g.player.position.x, g.player.position.y)
    hwnd = window.find_game()
    if hwnd is None or not window.bring_to_front(hwnd):
        raise SystemExit("the game isn't in front")
    eyes = Eyes()
    eyes.start()
    since = time.perf_counter()
    try:
        restart()
        start = wait_for_start(eyes, since, 12)
    finally:
        eyes.running = False
        eyes.join(timeout=2)
    if start is None:
        raise SystemExit("no drop-in seen")
    grabber, frames = Grabber(), []
    while time.perf_counter() < start + 1.0 + seconds:  # the camera has settled a second after the start
        img = grabber.grab()
        if time.perf_counter() > start + 1.0:
            frames.append(img.copy())
        time.sleep(0.25)
    grabber.close()
    cal = measure_screen(frames, feet)
    OUT.mkdir(exist_ok=True)
    SCREEN.write_text(json.dumps(cal, indent=1))
    print(f"screen {cal['screen']}: {cal['px_per_m']} px/m (art match {cal['match']:.3f}), camera rests "
          f"{cal['cam_above_feet']} m above the feet and {cal['cam_x_off']:+.3f} m across -> {SCREEN}")


def your_keys() -> int:
    held = ctypes.windll.user32.GetAsyncKeyState
    w, e = held(VK_W) & 0x8000, held(VK_E) & 0x8000
    return LEFT if w and not e else RIGHT if e and not w else NONE


def run(policy, eyes: Eyes, start: float, length_s: float, watch: bool, info: dict | None = None,
        late_s: float = 0.0) -> dict:
    """The closed loop, from the clock starting at `start` (perf_counter). late_s: the agent stands still that long
    first."""
    from sim.env import GetawayEnv
    shadow = Shadow()
    env = GetawayEnv(map="Tower", action_repeat=2, time_limit_s=90.0,
                     lift_speed_seen=bool((info or {}).get("lift_speed_seen", False)))
    env.reset()
    env.game = shadow.g
    paths = [o["path"] for o in eyes.level.objects]
    log = {"start": start, "grab_lag": GRAB_LAG, "key_lead": KEY_LEAD, "late_s": late_s, "grabs": [], "keys": [],
           "shadow": [],
           "film": []}  # film: the shadow every other tick (Game.snapshot), to draw what the agent saw (real.clip)
    held = NONE  # the key the game has
    a = shadow.g.held
    # the game's death notice (real.locate.death_marker): it stays up about 3 s, so only its first two grabs count,
    # and only once it has gone again (a notice left over from the last run fades in the first half second)
    notice = {"hits": 0, "first": None, "gone": 99, "armed": True}
    finished_for = 0
    try:
        for k in range(int(length_s / DT)):
            send_at = start + k * DT - KEY_LEAD
            while time.perf_counter() < send_at - 0.004:
                time.sleep(0.001)
            if stop_pressed():
                raise KeyboardInterrupt
            if eyes.error:
                raise eyes.error
            for t, row in eyes.take():  # what the grabs since the last tick saw
                tick = int(math.floor((t - GRAB_LAG - start) / DT))
                entry = {"t": t - start, "tick": tick, **row}
                if row.get("death_marker", 0.0) > DEATH_SCORE and tick > 25:
                    notice["first"] = tick if notice["hits"] == 0 else notice["first"]
                    notice["hits"] += 1
                    notice["gone"] = 0
                    if notice["hits"] == 2 and notice["armed"]:  # the real Steve died: so does the shadow
                        entry["death"] = shadow.die(notice["first"])
                        notice["armed"] = False
                else:
                    notice["hits"] = 0
                    notice["gone"] += 1
                    notice["armed"] = notice["armed"] or notice["gone"] >= 10
                st = row.get("steve")
                if st is not None and 0 <= tick <= shadow.tick and any(row["used"]):
                    lifts = {paths[int(i)]: y for i, (x, y) in row["objects"].items()}  # and the helicopter
                    entry["shadow"] = shadow.correct(tick, st["feet_world"], st["lean"], lifts, tuple(row["used"]))
                log["grabs"].append(entry)
            if watch:
                a = your_keys()
            elif k % 2 == 0:
                a = policy.act(env.observe()) if k * DT >= late_s else NONE
            while time.perf_counter() < send_at:
                pass
            if a != held and not watch:
                if not window.game_in_front():
                    raise window.NotInFront("the game left the front mid-run")
                if held in SCAN:
                    key(held, False)
                if a in SCAN:
                    key(a, True)
                held = a
                log["keys"].append((time.perf_counter() - start + KEY_LEAD, k, a))
            elif watch and a != held:
                held = a
                log["keys"].append((k * DT, k, a))
            shadow.step(a)
            eyes.steve_hint = (shadow.g.player.position.x, shadow.g.player.position.y) if shadow.g.alive else None
            p = shadow.g.player.position
            log["shadow"].append((k + 1, round(p.x, 3), round(p.y, 3), round(shadow.g.player.angle, 3)))
            if k % 2:
                log["film"].append((k + 1, a, shadow.g.snapshot()))
            finished_for = finished_for + 1 if shadow.g.finished else 0
            if finished_for * DT >= STAY_FINISHED_S:  # (a grab showing the real run elsewhere would have undone it)
                break
    finally:
        if held in SCAN and not watch:
            key(held, False)
    log["finished_in_shadow"] = shadow.g.finished
    log["time_s"] = shadow.g.time_s
    log["progress"] = 1.0 - shadow.g.dist_to_finish() / shadow.g.route_length
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", nargs="?")
    ap.add_argument("--watch", action="store_true", help="send no keys; follow a run you play")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--length", type=float, default=80.0, help="seconds to play at most")
    ap.add_argument("--clip", action="store_true", help="film each run (real.clip) into real/live/<date time>/")
    ap.add_argument("--calibrate", action="store_true", help="measure the screen (see the top), then stop")
    ap.add_argument("--late", type=float, default=0.0, help="the agent stands still this long (s) first")
    args = ap.parse_args()
    if args.calibrate:
        be_on_time()
        return calibrate()
    if not args.watch and not args.agent:
        ap.error("name an agent, or --watch")
    px_per_m, cam_above = screen_scale()
    print(f"screen: {px_per_m} px/m, camera {cam_above} m above the feet")
    policy, info = None, {}
    if args.agent:
        from common.policy import load_agent
        policy, info = load_agent(args.agent)
        print(f"agent {args.agent}: {info.get('method')}, sim best {info.get('time_s')}")
    be_on_time()
    out = OUT / time.strftime("%Y-%m-%d %H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    from sim.world import Game
    g = Game("Tower")
    g.reset()
    cam0 = (g.player.position.x, g.player.position.y + cam_above)
    for n in range(args.runs):
        eyes = Eyes(px_per_m, cam_above)
        if args.clip:
            from real.clip import Clip
            eyes.clip = Clip(out / f"run{n + 1}.mp4")
        eyes.start()
        start = None
        try:
            since = time.perf_counter()
            if args.watch:
                print("start (or restart) a run in the game", flush=True)
                start = wait_for_start(eyes, since, 120)
            else:
                hwnd = window.find_game()
                if hwnd is None or not window.bring_to_front(hwnd):
                    print("the game isn't in front: stopping")
                    return
                restart()
                start = wait_for_start(eyes, since, 12)
            if start is None:
                print("no drop-in seen")
                continue
            eyes.begin(start, cam0)
            log = run(policy, eyes, start, args.length + args.late, args.watch, info, args.late)
        finally:
            eyes.running = False
            eyes.join(timeout=2)
            if eyes.clip is not None:
                eyes.clip.close(start=start, agent=args.agent, kind="live")
        log["agent"] = args.agent
        (out / f"run{n + 1}.json").write_text(json.dumps(log))
        errs = [math.hypot(*s["error"]) for g_ in log["grabs"] if (s := g_.get("shadow")) and "error" in s]
        errs.sort()
        print(f"run {n + 1}: shadow progress {log['progress'] * 100:.0f}%"
              + (f", finished in the shadow at {log['time_s']:.2f} s" if log["finished_in_shadow"] else "")
              + (f"; {len(errs)} grabs, shadow off by median {errs[len(errs) // 2]:.3f} m, "
                 f"95% {errs[int(len(errs) * 0.95)]:.3f} m" if errs else "; no grabs used") + f" -> {out}", flush=True)


if __name__ == "__main__":
    main()
