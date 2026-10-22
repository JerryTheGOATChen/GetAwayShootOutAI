"""Which change to the sim's physics makes it follow the real trials longest (real.trials folders).

    py -m real.physics_check "real/trials/<date time>" [more folders] --variants base tilted_lean_ramps keys_shifted=1

Each variant is a sim with one change (see variant()); for every run in the folders it prints how long the real trials
stay on that sim's path (median, as real.trials_report measures it). keys_shifted=N moves every key press N ticks.
"""
import glob, json, sys
import numpy as np
from collections import defaultdict
from real.robustness import run_ticks
from real.trials_report import observed, fit, predict, stays, trial_ticks
from sim import world
from sim.world import Game

PX = 112.4


def variant(name):
    if "+" in name:  # several changes at once
        g = Game("Tower")
        for part in name.split("+"):
            g = _apply(g, part)
        return g
    return _apply(Game("Tower"), name)


def _apply(g, name):
    if name.startswith("keys_shifted"):
        g.keys_shifted = int(name.split("=")[1])
    if name == "lift_carries":
        # the lift moves by position (no velocity), and carries a player standing on it with it, as if parented:
        # the player keeps their own velocity, so nothing throws them up when it stops
        import math as _m
        def _update_elevators(g=g):
            for e in g.elevators:
                body = e["body"]
                old = body.position.copy()
                world.Game._step_elevator(e)
                v = body.linearVelocity.copy()
                body.linearVelocity = (0, 0)
                new = old + v * world.DT
                body.transform = (new, 0.0)
                delta = new - old
                if delta.lengthSquared == 0:
                    continue
                p = g.player
                for ce in p.contacts:
                    c = ce.contact
                    if not c.touching:
                        continue
                    other = c.fixtureB if c.fixtureA.body == p else c.fixtureA
                    if other.body != body or other.sensor:
                        continue
                    pts = c.worldManifold.points[:c.manifold.pointCount]
                    if pts and min(pt[1] for pt in pts) < p.worldCenter.y:
                        p.transform = (p.position + delta, p.angle)
                        break
        g._update_elevators = _update_elevators
    if name.startswith("lift_shift="):  # the lifts this many seconds further into their cycle at the start
        import dataclasses
        g.mv = dataclasses.replace(g.mv, lift_head_start=g.mv.lift_head_start + float(name.split("=")[1]))
    if name.startswith("first_key_late="):
        g.first_key_late = int(name.split("=")[1])
    if name.startswith("friction"):
        f = float(name.split("=")[1])
        for fx in g.player.fixtures:
            fx.friction = f
        orig_reset = g.reset
    if name.startswith("jump_keeps"):
        keep = float(name.split("=")[1])
        orig = g._jump
        def _jump(right, g=g, orig=orig):
            if g.is_turning and right == g.turn_right and g.grounded:
                v = g.player.linearVelocity
                g.player.linearVelocity = (v.x * keep, v.y * keep)
            orig(right)
        g._jump = _jump
    if name.startswith("landing_keeps"):
        keep = float(name.split("=")[1])
        orig_update = g._update_ground
        def _update_ground(g=g, orig=orig_update):
            was = g.grounded
            orig()
            if g.grounded and not was:
                v = g.player.linearVelocity
                g.player.linearVelocity = (v.x * keep, v.y)
        g._update_ground = _update_ground
    if name == "turn_from_upright":
        # a lean that starts on the ground starts from upright, whatever angle the body landed at
        orig_start = g._start_turn
        def _start_turn(right, g=g, orig=orig_start):
            was = g.is_turning
            orig(right)
            if g.is_turning and not was:
                g.player.transform = (g.player.position, 0.0)
        g._start_turn = _start_turn
    if name == "jump_along_new_lean":
        # the jump goes along the lean built up since this turn started, not the body's angle
        import math
        import Box2D as B
        orig_jump = g._jump
        def _jump(right, g=g, orig=orig_jump):
            if g.is_turning and right == g.turn_right and g.grounded:
                mv = g.mv
                t = g.turning_time
                a = 0.5 * mv.max_turn_speed / mv.lerp_turn_time * min(t, mv.lerp_turn_time) ** 2
                a = min(a, mv.max_turn_angle)
                ang = math.radians(-a if right else a)
                p = g.player
                up = B.b2Vec2(-math.sin(ang), math.cos(ang))
                p.ApplyLinearImpulse(up * mv.jump_force, p.worldCenter, True)
                g._airborne_ticks = 3
                p.fixedRotation = False
                g.is_turning = False
                g.turning_time = 0.0
                return
            orig(right)
        g._jump = _jump
    if name == "jump_along_turn":
        # the jump goes along the rotation this lean added since it started (same as the body's angle for a
        # lean started upright; ~upright for a lean started on landing at a tilt)
        import math
        import Box2D as B
        orig_start, orig_jump = g._start_turn, g._jump
        def _start_turn(right, g=g, orig=orig_start):
            was = g.is_turning
            orig(right)
            if g.is_turning and not was:
                g._turn_from = g.player.angle
        def _jump(right, g=g, orig=orig_jump):
            if g.is_turning and right == g.turn_right and g.grounded:
                ang = g.player.angle - getattr(g, "_turn_from", 0.0)
                p = g.player
                up = B.b2Vec2(-math.sin(ang), math.cos(ang))
                p.ApplyLinearImpulse(up * g.mv.jump_force, p.worldCenter, True)
                g._airborne_ticks = 3
                p.fixedRotation = False
                g.is_turning = False
                g.turning_time = 0.0
                return
            orig(right)
        g._start_turn, g._jump = _start_turn, _jump
    if name.startswith("inertia="):
        g.inertia_factor = float(name.split("=")[1])  # use with com=..., which sets the mass
    if name.startswith("com="):
        # the centre of mass this fraction of the body's height above the feet (the sim: 0.5, the box's middle)
        import Box2D as B
        from sim.world import PLAYER_H, PLAYER_W
        f = float(name.split("=")[1])
        orig_build = g._build_player
        def set_com(g=g):
            b = g.player
            m = b.mass
            i_centre = m * (PLAYER_W ** 2 + PLAYER_H ** 2) / 12.0 * getattr(g, "inertia_factor", 1.0)
            c = B.b2Vec2(0, f * PLAYER_H)
            b.massData = B.b2MassData(mass=m, center=c, I=i_centre + m * c.lengthSquared)
        def _build_player(g=g, orig=orig_build):
            orig()
            set_com(g)
        g._build_player = _build_player
        # Box2D recomputes the mass (back to the box's middle) whenever rotation is frozen or freed
        for meth in ("_start_turn", "_jump", "_place_player"):
            orig_m = getattr(g, meth)
            def wrapped(*a, orig=orig_m, g=g):
                r = orig(*a)
                set_com(g)
                return r
            setattr(g, meth, wrapped)
        set_com(g)
    if name.startswith("ground_ray="):
        import dataclasses
        g.mv = dataclasses.replace(g.mv, ground_ray=float(name.split("=")[1]))
    if name == "turn_needs_contact":
        orig_st = g._start_turn
        def _start_turn(right, g=g, orig=orig_st):
            if g.in_contact or g.is_turning:
                orig(right)
        g._start_turn = _start_turn
    if name.startswith("turn_after_landing="):
        n = int(name.split("=")[1])
        orig_st, orig_ug = g._start_turn, g._update_ground
        g._grounded_for = 0
        def _update_ground(g=g, orig=orig_ug):
            orig()
            g._grounded_for = g._grounded_for + 1 if g.grounded else 0
        def _start_turn(right, g=g, orig=orig_st):
            if g._grounded_for > n or g.is_turning:
                orig(right)
        g._update_ground, g._start_turn = _update_ground, _start_turn
    if name == "jump_ungrounds":
        # a jump leaves the ground at once: a key pressed in the same frame can't start a lean (in the air)
        orig_j = g._jump
        def _jump(right, g=g, orig=orig_j):
            was_grounded = g.grounded
            orig(right)
            if was_grounded and not g.is_turning and g._airborne_ticks == 3:
                g.grounded = False
        g._jump = _jump
    if name == "tilted_lean_ramps":
        # a lean started from a tilt (after a landing) jumps along a lean that ramps up from upright, as a lean from
        # standing does; a lean started upright jumps along the body, as before
        import math
        import Box2D as B
        orig_start, orig_jump = g._start_turn, g._jump
        def _start_turn(right, g=g, orig=orig_start):
            was = g.is_turning
            orig(right)
            if g.is_turning and not was:
                g._tilted_start = abs(g.rotation) > g.mv.max_balanced_angle
        def _jump(right, g=g, orig=orig_jump):
            if g.is_turning and right == g.turn_right and g.grounded and getattr(g, "_tilted_start", False):
                mv = g.mv
                t = min(g.turning_time, mv.lerp_turn_time)
                a = min(0.5 * mv.max_turn_speed / mv.lerp_turn_time * t * t, abs(g.rotation))
                ang = math.radians(-a if right else a)
                p = g.player
                up = B.b2Vec2(-math.sin(ang), math.cos(ang))
                p.ApplyLinearImpulse(up * mv.jump_force, p.worldCenter, True)
                g._airborne_ticks = 3
                p.fixedRotation = False
                g.is_turning = False
                g.turning_time = 0.0
                return
            orig(right)
        g._start_turn, g._jump = _start_turn, _jump
    return g


def path(g, ticks):
    g.reset()
    shift = getattr(g, "keys_shifted", 0)
    if shift:
        from real.robustness import late
        import real.robustness as R
        extra, R.EXTRA = R.EXTRA, 0.0
        ticks = late(ticks, shift, np.random.default_rng(0))
        R.EXTRA = extra
    first = getattr(g, "first_key_late", 0)
    if first and ticks and ticks[0] != 0:
        ticks = [0] * first + list(ticks[first:])  # the run's first key held from a few ticks later
    pos, ang = [], []
    for a in ticks:
        g.step(a)
        pos.append((g.player.position.x * PX, g.player.position.y * PX))
        ang.append(g.player.angle)
        if g.finished:
            break
    return np.asarray(pos), np.asarray(ang)


def main():
    folders = sys.argv[1:]
    runs = defaultdict(list)
    for folder in folders:
        for f in sorted(glob.glob(folder + "/*.json")):
            if not f.endswith("report.json"):
                d = json.load(open(f))
                runs[d["run"]].append(d)
    names = sys.argv[sys.argv.index("--variants") + 1:] if "--variants" in sys.argv else ["base", "turn_from_upright"]
    folders = folders[:folders.index("--variants")] if "--variants" in folders else folders
    ticks = {run: trial_ticks(trials[0]) for run, trials in runs.items()}
    for name in names:
        g = variant(name)
        if name.startswith("friction"):
            f = float(name.split("=")[1])
            orig_build = g._build_player
            def _build_player(g=g, orig=orig_build, f=f):
                orig()
                for fx in g.player.fixtures:
                    fx.friction = f
            g._build_player = _build_player
        row = []
        for run, trials in runs.items():
            pos, ang = path(g, ticks[run])
            obs = [observed(tr) for tr in trials]
            d, owns = fit(obs, pos, ang)
            st = [stays(t, np.hypot(*(xy - predict(t, pos, ang, d, own)).T), tr["length_s"])
                  for (t, xy), own, tr in zip(obs, owns, trials)]
            row.append(f"{run.split('/')[-2][:6]}/{run.split('/')[-1][:3]} {np.median(st):5.2f}")
        print(f"{name:18s}", " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
