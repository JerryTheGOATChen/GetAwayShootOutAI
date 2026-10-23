"""The sim run alongside the real game and kept on it: a shadow of the real run that an agent can play from, closed
loop, instead of replaying a run blind (which the real game can't repeat for more than about 14 s).

Each grab of the real game (real.locate) says where Steve's feet are, how far he leans and where the lifts are. The
shadow goes back to the tick the grab shows, moves Steve part of the way toward what was seen (and speeds him up or
slows him down by a little, so the same error doesn't keep coming back), puts the lifts where they were seen, and
plays the ticks since then again with the same keys.

    py -m real.shadow --video "C:/Users/ogjec/Videos/Getawayshootout/2026-10-02 23-41-14.mp4" --first 342
        offline test on a recorded run: its keys are read from the OBS key overlay, the shadow plays them and is
        corrected from the recording's track (py -m real.locate --video ... first), and it reports how far the
        shadow was from each grab before being corrected, against the same keys played blind.

The closed-loop rehearsal of agents in the sim (sim_test) is run by py -m real.rehearse.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from sim.world import DT, LEFT, MOVEMENT, NONE, RIGHT, Game

GAIN_POS = 0.6  # of a position error, corrected at once
GAIN_VEL = 3.0  # 1/s: velocity change per metre of error, so it doesn't build up again
GAIN_ANGLE = 0.5
DEADBAND = 0.02  # m: smaller errors are noise (the art match is good to about 2 cm)
STANDING_DEADBAND = 0.10  # m up or down while the shadow stands: his measured feet sit 2-4 cm above floors
ANGLE_DEADBAND = math.radians(4)
TOO_FAR = 1.5  # m: a grab this far off is a bad grab, unless the next few agree (then the shadow was lost: re-sync)
RESYNC_AFTER = 4  # grabs in a row
WEAK = 0.3  # gain on an axis the art doesn't pin down (the camera model's guess: right on average, drifts a little)
KEEP_S = 2.0  # how far back the shadow can go


class Shadow:
    def __init__(self, map_name: str = "Tower", game: Game | None = None):
        self.g = game or Game(map_name)
        if game is None:
            self.g.reset()
        self.snaps = {self.g.tick: self.g.snapshot()}
        self.actions: dict[int, int] = {}
        self.keep = int(KEEP_S / DT)
        self.lift_index = {e["path"]: i for i, e in enumerate(self.g.elevators)}
        self.hover_index = {h["path"]: h for h in self.g.hovers}  # the helicopter
        self.far = []  # (tick, feet) of the last grabs in a row that were too far off

    @property
    def tick(self) -> int:
        return self.g.tick

    def step(self, action: int):
        """Play one tick with this action held."""
        k = self.g.tick
        self.actions[k] = action
        self.g.step(action)
        self.snaps[self.g.tick] = self.g.snapshot()
        for old in [t for t in self.snaps if t < self.g.tick - self.keep]:
            del self.snaps[old]
            self.actions.pop(old, None)

    def correct(self, tick: int, feet=None, lean_deg=None, lifts=None, use=(True, True)) -> dict | None:
        """A grab showed the game at this tick: Steve's feet (world m) and lean (deg, + right), lifts {path: y} (and
        the helicopter: its y, which shifts its animation from then on).
        use: whether the grab's x and y can be trusted. Returns the errors found (before correcting)."""
        now = self.g.tick
        if tick not in self.snaps or tick > now:
            return None
        g = self.g
        g.restore(self.snaps[tick])
        p = g.player
        out = {"tick": tick, "behind": now - tick}
        changed = False
        if feet is not None and g.alive:
            e = np.array(feet) - np.array([p.position.x, p.position.y])
            out["error"] = e.tolist()
            if g.grounded and abs(e[1]) < STANDING_DEADBAND:
                e[1] = 0.0
            e = np.where(np.abs(e) < DEADBAND, 0.0, e)
            gain = np.where(use, 1.0, WEAK)
            resync_v = None
            if np.hypot(*e) > TOO_FAR:
                self.far.append((tick, np.array(feet, float)))
                last = self.far[-RESYNC_AFTER:]
                agree = len(last) == RESYNC_AFTER and max(np.hypot(*(a - b)) for _, a in last for _, b in last) <= TOO_FAR
                if not agree:
                    out["ignored"] = True
                    e = np.zeros(2)
                else:  # the last few grabs agree on where he is: the shadow is lost, so jump there, at the speed
                    out["resync"] = True  # they show him moving (2026-10-22: the error's own correction, 5 m/s per m,
                    gain = np.ones(2) / GAIN_POS  # launched a shadow 3.4 m off at 17 m/s)
                    (t0, f0), (t1, f1) = last[0], last[-1]
                    resync_v = (f1 - f0) / ((t1 - t0) * DT) if t1 > t0 else None
                    self.far = []
            else:
                self.far = []
            if e.any():
                angle = p.angle
                if lean_deg is not None:
                    ea = -math.radians(lean_deg) - p.angle
                    out["angle_error"] = math.degrees(ea)
                    if ANGLE_DEADBAND < abs(ea) < math.radians(45):
                        angle += GAIN_ANGLE * ea
                v = p.GetLinearVelocityFromLocalPoint((0, 0))
                w = p.angularVelocity
                d = GAIN_POS * gain * e
                p.transform = ((p.position.x + d[0], p.position.y + d[1]), angle)
                c = p.worldCenter - p.position
                if out.get("resync"):
                    vx, vy = (v.x, v.y) if resync_v is None else resync_v
                else:
                    vx, vy = v.x + GAIN_VEL * gain[0] * e[0], v.y + GAIN_VEL * gain[1] * e[1]
                p.linearVelocity = (vx - w * c.y, vy + w * c.x)
                p.awake = True
                changed = True
                if out.get("resync"):  # the checkpoints step one a tick: let them catch up with the jump (the agent
                    for _ in range(len(g.waypoints)):  # steers by the next three)
                        before = g.cur_wp
                        g._update_waypoints()
                        if g.cur_wp == before:
                            break
        for path, y in (lifts or {}).items():
            h = self.hover_index.get(path)
            if h is not None:
                err = y - g.hover_height(h)
                out.setdefault("lift_errors", {})[path] = err
                if abs(err) > DEADBAND:
                    h["offset"] += err
                    g._update_hovers()
                    changed = True
                continue
            i = self.lift_index.get(path)
            if i is None:
                continue
            b = g.elevators[i]["body"]
            out.setdefault("lift_errors", {})[path] = y - b.position.y
            if abs(y - b.position.y) > DEADBAND:
                b.transform = ((b.position.x, y), b.angle)
                changed = True
        if changed:
            self.snaps[tick] = g.snapshot()
        for k in range(tick, now):  # play the ticks since the grab again
            g.step(self.actions[k])
            self.snaps[k + 1] = g.snapshot()
        return out


    def die(self, tick: int) -> bool:
        """The real Steve died at this tick (the game's death notice): the shadow dies then too, unless it was
        already dead, and respawns where and when the game does (the sim's own rule: the last checkpoint, 1.5 s
        later). Returns whether it had to."""
        now = self.g.tick
        tick = min(max(tick, min(self.snaps)), now)
        g = self.g
        g.restore(self.snaps[tick])
        if g.finished:  # the shadow got away but the real Steve didn't: undo its finish
            g.finished, g.finish_tick = False, -1
            g.splits = g.splits[:-1]
        killed = g.alive
        if killed:
            g._die()
            self.snaps[tick] = g.snapshot()
        for k in range(tick, now):
            g.step(self.actions[k])
            self.snaps[k + 1] = g.snapshot()
        self.far = []
        return killed


# --- offline test on a recording -----------------------------------------------------------------------------

def overlay_keys(video: str, first: int, last: int) -> list[tuple[int, int]]:
    """(frame, action) whenever the held key changes, read from an OBS key overlay (W and E light up yellow): a
    press is the box getting much yellower within two frames, a release the reverse."""
    import cv2
    boxes = {"W": (1600, 176), "E": (1740, 176)}  # under the letters, 1920x1080
    v = cv2.VideoCapture(video)
    v.set(cv2.CAP_PROP_POS_FRAMES, first)
    yellow = {k: [] for k in boxes}
    for _ in range(first, last):
        ok, img = v.read()
        if not ok:
            break
        for k, (x, y) in boxes.items():
            b, g, r = img[y:y + 12, x:x + 60].reshape(-1, 3).astype(float).mean(0)
            yellow[k].append((r + g) / 2 - b)
    events = []
    for k, ys in yellow.items():
        on = False
        for t in range(2, len(ys)):
            if not on and ys[t] > 50 and ys[t] - min(ys[t - 1], ys[t - 2]) > 45:
                on = True
                events.append((first + t, k, True))
            elif on and max(ys[t - 1], ys[t - 2]) - ys[t] > 45 and ys[t] < 55:
                on = False
                events.append((first + t, k, False))
    events.sort()
    held, order, out = {"W": False, "E": False}, [], []
    for f, k, down in events:
        held[k] = down
        order = [kk for kk in order if kk != k] + ([k] if down else [])
        a = NONE if not order else (LEFT if order[-1] == "W" else RIGHT)
        if not out or out[-1][1] != a:
            out.append((f, a))
    return out


def video_test(track: dict, keys, tick0_frame: float, frame_lag: float = 0.0, correct: bool = True) -> dict:
    """Play the keys in a shadow, correcting it from the track's grabs. tick0_frame: the video frame of the sim's
    tick 0 (keys act at the frame the overlay shows them). frame_lag: frames between the physics tick and the
    grab showing it."""
    fps = track["fps"]
    sh = Shadow()
    paths = [o["path"] for o in json.loads((Path(__file__).resolve().parent.parent / "sim" / "maps" /
                                           "Tower_art.json").read_text())["objects"]]
    grabs = [r for r in track["rows"] if "steve" in r and any(r["used"])]
    tick_of = lambda f: int(math.floor((f - tick0_frame - frame_lag) / fps / DT + 1e-6))  # noqa: E731
    key_ticks = [(int(math.ceil((f - tick0_frame) / fps / DT - 1e-6)), a) for f, a in keys]
    a, ki, gi, results = RIGHT if not key_ticks or key_ticks[0][0] > 0 else NONE, 0, 0, []
    last_tick = tick_of(grabs[-1]["frame"]) + 1
    while sh.tick < last_tick:
        while ki < len(key_ticks) and key_ticks[ki][0] <= sh.tick:
            a = key_ticks[ki][1]
            ki += 1
        sh.step(a)
        while gi < len(grabs) and tick_of(grabs[gi]["frame"]) <= sh.tick:
            r = grabs[gi]
            gi += 1
            k = tick_of(r["frame"])
            if k < 0:
                continue
            st = r["steve"]
            lifts = {paths[int(i)]: y for i, (x, y) in r.get("objects", {}).items()}
            if correct:
                res = sh.correct(k, st["feet_world"], st["lean"], lifts, tuple(r["used"]))
            else:  # only measure
                snap_now = sh.g.snapshot()
                sh.g.restore(sh.snaps[k])
                res = {"tick": k, "error": (np.array(st["feet_world"]) -
                                            np.array([sh.g.player.position.x, sh.g.player.position.y])).tolist()}
                sh.g.restore(snap_now)
            if res and "error" in res:
                results.append(dict(t=k * DT, used=r["used"], **res))
    return dict(results=results, finished=sh.g.finished, time_s=sh.g.time_s, end=list(sh.g.player.position))


# --- closed-loop rehearsal in the sim ---------------------------------------------------------------------------

def sim_test(agent_path: str, runs: int = 10, mode: str = "shadow", seed: int = 0, grab_every: int = 2, lag: int = 2,
             noise: float = 0.02, lean_noise: float = 2.0, shadow_delay: int = 0, notice_lag: int = 6,
             late_s: float = 0.0, lift_shift_s: float = 0.0, movement=MOVEMENT) -> list[dict]:
    """An agent plays a "truth" sim standing in for the real game: its keys land 0-3 ticks late (Jitter), and it
    only sees the truth through grabs every grab_every ticks, lag ticks old, with noise (m, deg).
    mode: "shadow" (it plays from a shadow corrected by the grabs), "blind" (from a shadow never corrected) or
    "sees" (it sees the truth itself, exactly and at once: the best it could do). shadow_delay: ticks the shadow
    expects a key to take (live, keys are sent KEY_LEAD early to make up the game's average delay). When the truth
    dies the shadow is told notice_lag ticks later, as real.live learns it from the game's death notice.
    late_s: each run the agent first stands still for up to this long (at random), so it gets to the lifts late.
    lift_shift_s: each run the truth's lifts and helicopter run up to this much ahead of or behind the sim's
    schedule (at random), as the game's did by about 0.2 s; the shadow only knows from the grabs.
    movement: the physics both sims play in (sim/world.py's Movement)."""
    from common.policy import load_agent
    from sim.env import GetawayEnv, Jitter
    policy, info = load_agent(agent_path)
    jitter = Jitter()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        shift = rng.uniform(-lift_shift_s, lift_shift_s) if lift_shift_s else 0.0
        truth = Game("Tower", replace(movement, lift_head_start=movement.lift_head_start + shift))
        sh = Shadow(game=Game("Tower", movement))
        env = GetawayEnv(map="Tower", action_repeat=2, time_limit_s=90.0,
                         lift_speed_seen=bool(info.get("lift_speed_seen", False)))
        env.reset()
        env.game = truth if mode == "sees" else sh.g
        delay = int(rng.integers(jitter.delay + 1))
        idle = int(rng.uniform(0, late_s) / DT) if late_s else 0
        asked = applied = truth.held
        a = applied
        pending, seen, mine = [], {}, []  # keys on their way to the truth, the truth's past, and to the shadow
        notices = []  # ticks the truth died at, until the shadow is told
        sh_applied = sh_asked = a
        for k in range(int(90.0 / DT)):
            if k % 2 == 0:
                a = policy.act(env.observe()) if k >= idle else NONE
                if a != asked:
                    asked = a
                    lands = k + delay + int(rng.random() < jitter.extra)
                    pending.append((max(lands, pending[-1][0]) if pending else lands, a))
            while pending and pending[0][0] <= truth.tick:
                applied = pending.pop(0)[1]
            truth.step(applied)
            if "death" in truth.events:
                notices.append(truth.tick)
            if mode == "shadow" and notices and truth.tick - notices[0] >= notice_lag:
                sh.die(notices.pop(0))
            if mode != "sees":
                if a != sh_asked:
                    sh_asked = a
                    mine.append((k + shadow_delay, a))
                while mine and mine[0][0] <= k:
                    sh_applied = mine.pop(0)[1]
                sh.step(sh_applied)
            seen[truth.tick] = (truth.player.position.x, truth.player.position.y, truth.player.angle,
                                {e["path"]: e["body"].position.y for e in truth.elevators + truth.hovers}, truth.alive)
            if truth.finished:
                break
            if mode == "shadow" and k % grab_every == 0 and truth.tick - lag in seen:
                x, y, ang, lifts, alive = seen[truth.tick - lag]
                if alive:
                    sh.correct(truth.tick - lag, (x + rng.normal(0, noise), y + rng.normal(0, noise)),
                               -math.degrees(ang) + rng.normal(0, lean_noise), lifts)
        out.append(dict(finished=truth.finished, time_s=truth.time_s, deaths=truth.deaths, late_s=idle * DT,
                        lift_shift_s=shift,
                        progress=1.0 - truth.dist_to_finish() / truth.route_length,
                        end=[round(truth.player.position.x, 2), round(truth.player.position.y, 2)]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--first", type=int, help="frame the run's clock starts")
    ap.add_argument("--tick0", type=float, default=None, help="video frame of the sim's tick 0 (default first+7)")
    ap.add_argument("--sim-test", metavar="AGENT", help="closed-loop rehearsal of an agent in the sim")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()
    if args.sim_test:
        for mode in ("sees", "shadow", "blind"):
            res = sim_test(args.sim_test, args.runs, mode)
            done = [r for r in res if r["finished"]]
            print(f"{mode:6s}: finished {len(done)}/{len(res)}"
                  + (f", median {np.median([r['time_s'] for r in done]):.2f} s" if done else "")
                  + f"; progress median {np.median([r['progress'] for r in res]) * 100:.0f}%, deaths "
                  f"{sum(r['deaths'] for r in res)}", flush=True)
        return
    track = json.loads((Path(__file__).resolve().parent / "tracks" / f"{Path(args.video).stem}.json").read_text())
    last = track["rows"][-1]["frame"]
    keys = overlay_keys(args.video, args.first, last + 1)
    tick0 = args.tick0 if args.tick0 is not None else args.first + 7
    for name, corr in (("blind", False), ("shadow", True)):
        out = video_test(track, keys, tick0, correct=corr)
        if corr:
            (Path(__file__).resolve().parent / "tracks" / f"{Path(args.video).stem}_shadow.json").write_text(json.dumps(out))
        err = np.array([np.hypot(*r["error"]) for r in out["results"] if all(r["used"])])
        t = np.array([r["t"] for r in out["results"] if all(r["used"])])
        far = t[err > 0.3]
        print(f"{name:6s}: {len(err)} grabs; distance from each grab before correcting: median {np.median(err):.3f} m, "
              f"95% {np.percentile(err, 95):.3f} m; first over 30 cm at {far[0] if len(far) else float('nan'):.2f} s; "
              f"over 30 cm in {100 * np.mean(err > 0.3):.1f}%; finished: {out['finished']} ({out['time_s']:.2f} s)")


if __name__ == "__main__":
    main()
