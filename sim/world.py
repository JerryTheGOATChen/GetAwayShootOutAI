"""Getaway Shootout speedrun sim: the original maps (extracted from the 2019 build) on Box2D, the same
physics engine Unity's 2D physics uses, with the player controller ported from the developer's
tutorial project (github.com/neweichgames/getaway-shootout-tutorial-series, PlayerMovement.cs).

One player, no weapons, no bots. Fixed 50 Hz step like the game (Time.fixedDeltaTime = 0.02).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import Box2D as B

DT = 0.02
TICK_HZ = 50
GRAVITY = -9.81
VEL_ITERS, POS_ITERS = 8, 3
MAPS_DIR = Path(__file__).resolve().parent / "maps"
MAP_NAMES = ("Street", "Train", "Town", "Tower")

NONE, LEFT, RIGHT = 0, 1, 2  # action = which direction key is held

# Everything shown to people (viewer readouts, docs) is in the real game's screen pixels with the game
# window 1920 wide: the map art is 16 art pixels per Box2D unit and each art pixel is 7.02 screen px
# (1080p recordings agree within 1%). Box2D itself keeps working in its own units (metres), where its
# tolerances are tuned; this converts.
PX_PER_M = 112.4

DEFAULT_FRICTION = 0.4  # Unity's default PhysicsMaterial2D
PLAYER_W, PLAYER_H = 0.5625, 2.0
ZIP_ANCHOR = (-0.061, 2.0)  # the player's "Zipline" child: the point that hooks onto a zipline
# The lifts start moving when the level loads; the in-game timer starts a moment later, when the
# player drops in. The lifts are this many seconds into their cycle at the sim's tick 0 (when the game starts taking
# keys, about 0.12 s after its timer shows zero). The schedule is the same every run. Fitted to real-game trials of
# a lift ride (2026-10-20): the camera settling as the first lift stops matched best 0.30 s later into the cycle
# than the 1.25 s measured from timer zero in recordings, which only saw the timer.
ELEVATOR_HEAD_START = 1.55
ELEVATOR_EXTRA_HEAD_START = {"Map_Tower/Tower/Elevator (1)/MapObj_Elevator": 0.17}
# Each lift cab carries two death zones from the real map, one just under its floor and one just
# above its roof. They are switched off in the level file, and the lift script (compiled, so not
# readable) must turn them on as the cab arrives: in a recording, a player the first Tower lift came
# down on was pushed over alive for half a second and only died once the cab was between 98 and
# 83 px from its bottom stop, pinning them flat. So a lift kills what it squashes against the
# ground (or the block above the shaft), not what it is only pushing along.
ELEVATOR_CRUSH_DIST = 90 / PX_PER_M
# The getaway helicopter hovers. Its Animator (clip Heli_MoveVertical in the 2019 build, a 2 s loop) sets the
# Helicopter_Body's local position to (0, y), y easing from -0.25 up to +0.25 m and back, in place of the scene's
# (0, 1.3): so the body, with its floor, walls, rotor and the finish trigger, hangs 1.05-1.55 m below where the
# scene puts it (in the user's recordings: 1.09-1.58 m lower, at the same x). Like the lifts, it starts when the
# level loads: the first recording's bob was 1.58 s into the clip at tick 0, the lifts' head start is 1.55 s.
# Animated body path -> its local height in the scene, which the animation replaces.
HOVERS = {"Map_Tower/Tower/Helicopter/Helicopter_Body": 1.3}


def hover_y(t: float) -> float:
    """Heli_MoveVertical's local height t s into the clip (its keys, read from the build: two cubic pieces)."""
    x = t % 2.0
    if x < 1.0:
        return -x ** 3 + 1.5 * x ** 2 - 0.25
    x -= 1.0
    return x ** 3 - 1.5 * x ** 2 + 0.25


@dataclass(frozen=True)
class Movement:
    """The player controller's settings. The controller itself is PlayerMovement.cs from the tutorial project.

    The lean ramp is measured from a 60 fps recording of the current game with the keys on screen: holding
    a key from standing, the turning speed grows by 400 deg/s every second (four holds: 390-520) until the
    max lean, 56 deg, about 0.5 s in. That is a third of the 2019 build's stored ramp (turnSpeed 550 deg/s
    over lerpTurnSpeed 0.45 s). On the ground the max lean always comes first, so the top speed is never
    seen; it stays at the 2019 value. The body's drag (none), mass and gravity scale are the 2019 build's
    Rigidbody2D values and the rest are the tutorial's; the same recording's jumps confirm them (gravity,
    launch speeds and distances within a few per cent, see the README)."""
    balance_force: float = 0.3
    max_balanced_angle: float = 7.2  # deg
    max_turn_speed: float = 550.0  # deg/s (2019 build: turnSpeed; never reached on the ground)
    lerp_turn_time: float = 1.38  # s to reach max turn speed: 550 / 400 deg/s^2, measured
    max_turn_angle: float = 53.5  # deg
    jump_force: float = 11.0  # impulse along the body's up axis
    ground_ray: float = 0.2
    respawn_time: float = 1.5
    gravity_scale: float = 1.5
    linear_drag: float = 0.0  # Rigidbody2D.drag (Box2D linear damping), as in the 2019 build
    angular_damping: float = 0.05
    # The centre of mass sits this far up the body (as a fraction of its height), not in the middle of the box:
    # landing tilted, the real player first rocks back, then tips forward. Fitted to 70 real-game trials played from
    # the same keys as sim runs (real.physics_check, 2026-10-20): 0.3 kept the sim on them longest; 0.5 rocked
    # straight forward on every landing, and the runs left the real ones at the first landing re-jump.
    centre_of_mass: float = 0.3
    # A jump leaves the ground at once, so a key pressed in the same frame can't start a lean (see Game._jump).
    jump_leaves_ground: bool = True
    # The lifts carry a player standing on them as if parented, and how far into their cycle they are at tick 0
    # (see Game._update_elevators, ELEVATOR_HEAD_START). Lifts, not the player, but part of the same calibration.
    lift_carries: bool = True
    lift_head_start: float = ELEVATOR_HEAD_START
    # The helicopter moves as animated (HOVERS), on the lifts' clock; before 2026-10-22 it stayed where the scene
    # puts it, 1.05-1.55 m too high. Scenery, not the player, but part of the same calibration.
    helicopter_hovers: bool = True


MOVEMENT = Movement()
# The physics before 2026-10-20 (centre of mass in the middle of the box; a key pressed in a jump's own frame could
# start a lean in the air; lifts that throw you up when they stop, 0.3 s later in their cycle), which your
# runs/pb_<Map>.json runs until then were played in.
OLD_MOVEMENT = Movement(centre_of_mass=0.5, jump_leaves_ground=False, lift_carries=False, lift_head_start=1.25,
                        helicopter_hovers=False)
# The physics of 2026-10-20 to 2026-10-22: today's, with the helicopter still where the scene puts it. Every agent of
# batches 01-06 was trained in it.
MOVEMENT_2026_10_20 = Movement(helicopter_hovers=False)


class _GroundRay(B.b2RayCastCallback):
    def __init__(self):
        super().__init__()
        self.ignore = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body == self.ignore or fixture.sensor:  # Physics2D.queriesHitTriggers = false
            return -1.0
        self.hit = True
        return 0.0


class _InsidePoint(B.b2QueryCallback):
    """Is a point inside any solid fixture other than the player? Unity's raycasts count a collider the ray
    starts inside as hit (Physics2D.queriesStartInColliders, on by default); Box2D's skip it."""

    def __init__(self):
        super().__init__()
        self.ignore = None
        self.point = None
        self.hit = False

    def ReportFixture(self, fixture):
        if fixture.body != self.ignore and not fixture.sensor and fixture.TestPoint(self.point):
            self.hit = True
            return False
        return True


class _ClosestRay(B.b2RayCastCallback):
    def __init__(self):
        super().__init__()
        self.ignore = None
        self.fraction = 1.0
        self.kind = 0
        self.crush_active = None  # Game.crush_active: a lift's crush zone only counts while it is deadly

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body == self.ignore:
            return -1.0
        ud = fixture.userData or {}
        deadly = ud.get("tag") == "Death" or (ud.get("tag") == "Crush" and self.crush_active(ud))
        if fixture.sensor and not deadly:
            return -1.0
        self.fraction = fraction
        self.kind = 2 if deadly else 1
        return fraction


def _triangulate(pts):
    """Ear clipping for simple polygons -> list of triangles (Box2D needs convex pieces)."""
    pts = [tuple(p) for p in pts]
    signed = sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts)))
    if signed < 0:  # ear test below assumes counter-clockwise winding
        pts = pts[::-1]
    idx = list(range(len(pts)))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-9:
                continue
            if any(cross(a, b, pts[j]) >= 0 and cross(b, c, pts[j]) >= 0 and cross(c, a, pts[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append([a, b, c])
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append([pts[i] for i in idx])
    return tris


def _convex(pts):
    n = len(pts)
    if n < 3 or n > 8:
        return False
    sign = 0
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) < 1e-12:
            continue
        s = 1 if cr > 0 else -1
        if sign and s != sign:
            return False
        sign = s
    return True


def _poly_area(pts):
    return 0.5 * abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts))))


class Game:
    def __init__(self, map_name: str, movement: Movement = MOVEMENT):
        self.map_name = map_name
        self.data = json.loads((MAPS_DIR / f"{map_name}.json").read_text())
        self.mv = movement
        self.ziplines = [(tuple(z["zipline"]["top"]), tuple(z["zipline"]["bottom"]), z["zipline"]["catch_zone"])
                         for z in self.data["scripts"] if z["script"] == "Zipline" and z["active"]]
        wps = [s for s in self.data["scripts"] if s["script"] == "Waypoint" and s["active"]]
        wps.sort(key=lambda s: s["order"])  # scene order = route order
        self.waypoints = [tuple(w["pos"]) for w in wps]
        self.wp_spawnable = [w["spawnable"] for w in wps]
        # each waypoint is a line: you have passed it once you are on side_x of its x (and, for some,
        # above its height). Decoded from the game's Waypoint data.
        self.wp_zone = [(w["side_x"], w["above"], w["y_offset"]) for w in wps]
        # distance along the route from each waypoint to the last one (WaypointManager.Awake)
        self.wp_dist = [0.0] * len(self.waypoints)
        for i in range(len(self.waypoints) - 2, -1, -1):
            a, b = self.waypoints[i], self.waypoints[i + 1]
            self.wp_dist[i] = self.wp_dist[i + 1] + math.dist(a, b)
        self.route_length = self.wp_dist[0] + 0.0
        self.kill_y = self.data["bounds"][1] - 15.0
        self._gray = _GroundRay()
        self._inside = _InsidePoint()
        self._cray = _ClosestRay()
        self._cray.crush_active = self.crush_active
        self.reset()

    # --- construction -------------------------------------------------------

    def reset(self):
        self.world = B.b2World(gravity=(0, GRAVITY), doSleep=True)
        self._build_map()
        self._build_player()
        self.tick = 0
        self.finished = False
        self.finish_tick = -1
        self.deaths = 0
        self.dead_timer = 0
        self.held = NONE
        self.is_turning = False
        self.turn_right = True
        self.turning_time = 0.0
        self.unbalanced_time = 0.0
        self.in_contact = False
        self._airborne_ticks = 0
        self.grounded = False
        self.cur_wp = 0
        self.spawn_wp = 0
        self.splits: list[int] = []
        self.events: set[str] = set()
        self.zip = -1  # index of the zipline the player is riding, or -1
        self._zip_released = False
        self._place_player(self.waypoints[0])
        for e in self.elevators:
            head = self.mv.lift_head_start + ELEVATOR_EXTRA_HEAD_START.get(e["path"], 0.0)
            b = e["body"]
            for _ in range(int(round(head * TICK_HZ))):
                self._step_elevator(e)
                b.transform = (b.position + b.linearVelocity * DT, 0.0)
        self._update_hovers()
        self._update_ground()

    def _fixture_def(self, shape_pts, sensor, mat, density=0.0):
        fr = mat["friction"] if mat else DEFAULT_FRICTION
        rs = mat["bounciness"] if mat else 0.0
        return B.b2FixtureDef(shape=B.b2PolygonShape(vertices=shape_pts), isSensor=sensor, friction=fr,
                              restitution=rs, density=density)

    def _add_shape(self, body, sh, to_local, density=0.0, mat=None):
        mat = sh.get("material") or mat
        sensor = sh["trigger"]
        pts = [to_local(p) for p in sh["points"]]
        pieces = [pts] if _convex(pts) else _triangulate(pts)
        for piece in pieces:
            if _poly_area(piece) < 1e-4:
                continue
            fx = body.CreateFixture(self._fixture_def(piece, sensor, mat, density))
            fx.userData = {"tag": sh["tag"], "effector": self._effectors.get(sh["path"])}
            self._fixtures.append(fx)

    def _build_map(self):
        d = self.data
        self._fixtures = []
        self._door_paths = {x["path"] for x in d["scripts"] if x["script"] == "Door" and x["active"]}
        self._effectors = {e["path"]: e for e in d["effectors"]}
        self.effector_fixtures = []
        ground = self.world.CreateStaticBody()
        # elevators become kinematic bodies moving between their two points
        self.elevators = []
        elev_paths = {}
        for s in d["scripts"]:
            if s["script"] == "Elevator" and s["active"]:
                e = s["elevator"]
                body = self.world.CreateKinematicBody(position=tuple(s["pos"]))
                lo, hi = sorted(e["points"], key=lambda p: p[1])
                self.elevators.append({"body": body, "path": s["path"], "points": (lo, hi), "speed": e["speed"],
                                       "wait": e["wait"], "target": 1 if e["going_up"] else 0, "timer": 0.0})
                elev_paths[s["path"]] = body
        # animated scenery (the helicopter) becomes kinematic bodies, moved along their animation (HOVERS). Built
        # even when it doesn't move (helicopter_hovers off), so every game of a map has the same bodies for
        # snapshot() / restore()
        self.hovers = []
        hover_paths = {}
        for a in d["animators"]:
            if a["path"] in HOVERS and a["active"] and a["enabled"]:
                body = self.world.CreateKinematicBody(position=tuple(a["pos"]), userData="hover")
                self.hovers.append({"body": body, "path": a["path"], "scene": tuple(a["pos"]),
                                    "base": HOVERS[a["path"]], "offset": 0.0})
                hover_paths[a["path"]] = body
        # rigidbodies
        bodies = {}
        for bd in d["bodies"]:
            if not bd["active"] or bd["type"] == "static":
                continue
            maker = self.world.CreateDynamicBody if bd["type"] == "dynamic" else self.world.CreateKinematicBody
            body = maker(position=tuple(bd["pos"]), angle=math.radians(bd["angle"]),
                         linearDamping=bd["drag"], angularDamping=bd["angular_drag"], gravityScale=bd["gravity"])
            bodies[bd["id"]] = (body, bd)
        # shapes
        body_shapes = {}
        for sh in d["shapes"]:
            if not sh["active"] or not sh["enabled"] or sh["kind"] == "circle" or sh["tag"] == "PowerUp":
                continue
            if sh["path"] in self._door_paths:
                continue  # doors are cosmetic in the current game: you pass straight through
            if sh["body"] is not None and sh["body"] in bodies:
                body_shapes.setdefault(sh["body"], []).append(sh)
                continue
            elev = next((b for p, b in {**elev_paths, **hover_paths}.items()
                         if sh["path"] == p or sh["path"].startswith(p + "/")), None)
            if elev is not None:
                ox, oy = elev.position
                self._add_shape(elev, sh, lambda p, ox=ox, oy=oy: (p[0] - ox, p[1] - oy))
            else:
                self._add_shape(ground, sh, lambda p: (p[0], p[1]))
        # the cabs' own crush zones (inactive Death triggers parented to the cab; see ELEVATOR_CRUSH_DIST)
        for i, e in enumerate(self.elevators):
            ox, oy = e["body"].position
            for sh in d["shapes"]:
                if sh["active"] or sh["tag"] != "Death" or not sh["trigger"] or not sh["path"].startswith(e["path"] + "/"):
                    continue
                pts = [(p[0] - ox, p[1] - oy) for p in sh["points"]]
                fx = e["body"].CreateFixture(self._fixture_def(pts, True, None))
                # side: the stop this zone is deadly at (0 = under the cab, at the bottom; 1 = above it, at the top)
                fx.userData = {"tag": "Crush", "effector": None, "elevator": i, "side": 1 if sum(p[1] for p in pts) > 0 else 0}
                self._fixtures.append(fx)
        for bid, shapes in body_shapes.items():
            body, bd = bodies[bid]
            area = sum(_poly_area(s["points"]) for s in shapes if not s["trigger"]) or 1.0
            dens = bd["mass"] / area
            xf = body.transform
            for sh in shapes:
                self._add_shape(body, sh, lambda p, xf=xf: tuple(B.b2MulT(xf, B.b2Vec2(*p))), dens, bd["material"])
        # wheel joints (parked cars' suspension)
        by_rb = {bd["rb_path_id"]: bid for bid, (b, bd) in bodies.items()}
        for j in d["joints"]:
            if j["body"] not in bodies:
                continue
            body_a = bodies[j["body"]][0]
            cid = by_rb.get(j["connected"])
            body_b = bodies[cid][0] if cid is not None else ground
            anchor = body_a.GetWorldPoint(tuple(j["anchor"]))
            ang = math.radians(j["suspension"]["angle"])
            self.world.CreateWheelJoint(bodyA=body_b, bodyB=body_a, anchor=anchor, axis=(math.cos(ang), math.sin(ang)),
                                        frequencyHz=j["suspension"]["frequency"],
                                        dampingRatio=j["suspension"]["damping"], enableMotor=False)
        for fx in self._fixtures:
            if fx.userData["effector"]:
                self.effector_fixtures.append(fx)

    def _build_player(self):
        mv = self.mv
        self.player = self.world.CreateDynamicBody(position=(0, 0), angularDamping=mv.angular_damping,
                                                   linearDamping=mv.linear_drag, gravityScale=mv.gravity_scale)
        fx = self.player.CreateFixture(B.b2FixtureDef(
            shape=B.b2PolygonShape(box=(PLAYER_W / 2, PLAYER_H / 2, (0, PLAYER_H / 2), 0)),
            density=1.0 / (PLAYER_W * PLAYER_H), friction=DEFAULT_FRICTION, restitution=0.0))
        fx.userData = {"tag": "Player"}
        self.player.userData = "player"
        self._set_mass()

    def _set_mass(self):
        """The player's mass (1), with the centre of mass at mv.centre_of_mass of the height and the box's own
        inertia about it. Box2D puts the centre back in the middle of the box every time rotation is locked or
        freed, so _lock_rotation calls this again."""
        b = self.player
        m = b.mass
        centre = B.b2Vec2(0, self.mv.centre_of_mass * PLAYER_H)
        inertia = m * (PLAYER_W ** 2 + PLAYER_H ** 2) / 12.0 + m * centre.lengthSquared
        b.massData = B.b2MassData(mass=m, center=centre, I=inertia)

    def _lock_rotation(self, locked: bool):
        """RigidbodyConstraints2D.FreezeRotation on or off."""
        self.player.fixedRotation = locked
        self._set_mass()

    def _place_player(self, pos):
        p = self.player
        p.active = True
        self._lock_rotation(False)
        p.transform = (pos, 0.0)
        p.linearVelocity = (0, 0)
        p.angularVelocity = 0.0
        p.awake = True
        self.is_turning = False
        self.turning_time = 0.0
        self.unbalanced_time = 0.0

    # --- queries ------------------------------------------------------------

    @property
    def rotation(self) -> float:
        """Player rotation in degrees, -180..180, 0 = upright (PlayerBody.GetRotation)."""
        r = math.degrees(self.player.angle) % 360.0
        return r - 360.0 if r > 180.0 else r

    @property
    def alive(self) -> bool:
        return self.dead_timer == 0

    @property
    def time_s(self) -> float:
        return (self.finish_tick if self.finished else self.tick) * DT

    def dist_to_finish(self) -> float:
        """Route distance left: straight to the next checkpoint line's waypoint, then along the route."""
        nxt = min(self.cur_wp + 1, len(self.waypoints) - 1)
        return math.dist(self.waypoints[nxt], tuple(self.player.position)) + self.wp_dist[nxt]

    def _update_ground(self):
        p = self.player
        down = p.GetWorldVector((0, -1))
        self._gray.ignore = self._inside.ignore = p
        self._gray.hit = self._inside.hit = False
        for sx in (-0.275, 0.0, 0.275):
            o = p.GetWorldPoint((sx, -0.01))
            # at a steep lean the corner the body pivots on sits a hair inside the ground: like Unity, that counts
            self._inside.point = o
            self.world.QueryAABB(self._inside, B.b2AABB(lowerBound=(o.x - 1e-4, o.y - 1e-4), upperBound=(o.x + 1e-4, o.y + 1e-4)))
            if self._inside.hit:
                break
            self.world.RayCast(self._gray, o, o + down * self.mv.ground_ray)
            if self._gray.hit:
                break
        self.grounded = self._gray.hit or self._inside.hit

    def ray(self, origin, direction, length):
        """Closest hit fraction along a ray and what it hit (0 nothing, 1 solid, 2 death)."""
        self._cray.ignore = self.player
        self._cray.fraction, self._cray.kind = 1.0, 0
        end = (origin[0] + direction[0] * length, origin[1] + direction[1] * length)
        self.world.RayCast(self._cray, origin, end)
        return self._cray.fraction, self._cray.kind

    # --- PlayerMovement port ------------------------------------------------

    def _start_turn(self, right: bool):
        if self.is_turning or not self.grounded:
            return
        self._lock_rotation(True)
        self.turn_right = right
        self.is_turning = True

    def _jump(self, right: bool):
        if not self.is_turning or right != self.turn_right:
            return
        p = self.player
        if self.grounded:
            up = p.GetWorldVector((0, 1))
            p.ApplyLinearImpulse(up * self.mv.jump_force, p.worldCenter, True)
            # Box2D keeps reporting the ground contact for two more steps after take-off. In recordings
            # the body turns back upright at ~54 deg/s in the air, which is exactly one tick of balance
            # torque, so the contact must not count once the jump has happened.
            self._airborne_ticks = 3
            # and the player is off the ground at once: a key pressed in the same frame can't start a lean (which
            # would lock the body's rotation and turn it in the air). In real-game trials of switching keys at
            # the jump, the body never turned in the air (2026-10-20); the sim's used to swing 56 deg.
            if self.mv.jump_leaves_ground:
                self.grounded = False
        self._lock_rotation(False)
        self.is_turning = False
        self.turning_time = 0.0

    def _turn(self):
        mv = self.mv
        self.turning_time += DT
        rot = self.rotation
        d = -1 if self.turn_right else 1
        speed = min(self.turning_time / mv.lerp_turn_time, 1.0) * mv.max_turn_speed
        if (rot > mv.max_turn_angle * d) == self.turn_right:
            p = self.player
            p.transform = (p.position, math.radians(rot + speed * d * DT))
            # Rigidbody2D.SetRotation wakes the body; Box2D's SetTransform doesn't. Asleep (after half a second
            # standing still) its contacts aren't solved, so a lean into a wall stuck in it instead of pushing
            # the feet away, as recordings show
            p.awake = True

    def _balance(self):
        mv = self.mv
        rot = self.rotation
        if abs(rot) > mv.max_balanced_angle:
            if self.in_contact:
                self.unbalanced_time += DT
                mult = min(max(math.ceil(self.unbalanced_time / 1.2), 1), 4)
                power = min(max(-rot, -90.0), 90.0)
                self.player.ApplyTorque(power * mv.balance_force * mult, True)
        elif self.unbalanced_time > 0.0:
            self.player.angularVelocity = self.player.angularVelocity / 100.0
            self.unbalanced_time = 0.0

    # --- saving and restoring a moment -----------------------------------------

    _STATE = ("tick", "finished", "finish_tick", "deaths", "dead_timer", "held", "is_turning", "turn_right",
              "turning_time", "unbalanced_time", "in_contact", "_airborne_ticks", "grounded", "cur_wp", "spawn_wp",
              "zip", "_zip_released", "_zip_speed", "_zip_s")

    def snapshot(self) -> dict:
        """Everything about this moment: every body's place and motion, the lifts and the player controller. Velocities
        are kept at each body's origin (the player's feet), so a moment from a game with another centre of mass
        restores the same motion."""
        bodies = []
        for b in self.world.bodies:
            v = b.GetLinearVelocityFromLocalPoint((0, 0))
            bodies.append((b.position.x, b.position.y, b.angle, v.x, v.y, b.angularVelocity, b.awake, b.fixedRotation))
        lifts = [(e["target"], e["timer"]) for e in self.elevators]
        return {"bodies": bodies, "lifts": lifts, "hovers": [h["offset"] for h in self.hovers], "splits": list(self.splits),
                "state": {k: getattr(self, k) for k in self._STATE if hasattr(self, k)}}

    def restore(self, snap: dict):
        """Put the game back at a moment from snapshot() (of the same map, in a game with any Movement)."""
        for b, (x, y, angle, vx, vy, w, awake, fixed) in zip(self.world.bodies, snap["bodies"]):
            if b == self.player:
                self._lock_rotation(fixed)
            elif b.fixedRotation != fixed:
                b.fixedRotation = fixed
            b.transform = ((x, y), angle)
            b.angularVelocity = w
            c = b.worldCenter - b.position  # the velocity at the centre of mass, from the one at the origin
            b.linearVelocity = (vx - w * c.y, vy + w * c.x)
            b.awake = awake
        for e, (target, timer) in zip(self.elevators, snap["lifts"]):
            e["target"], e["timer"] = target, timer
        for h, offset in zip(self.hovers, snap.get("hovers", [])):
            h["offset"] = offset
        for k, v in snap["state"].items():
            setattr(self, k, v)
        self.splits = list(snap["splits"])
        self.events = set()
        self.player.active = self.dead_timer == 0  # a dead player's body is out of the world until respawning
        self._update_hovers()  # where its animation has it then (a moment from a game with other physics may differ)

    # --- step ---------------------------------------------------------------

    def step(self, action: int) -> None:
        """Advance one physics tick (0.02 s). action: 0 none, 1 hold left, 2 hold right."""
        self.events = set()
        if self.finished:
            return
        self.tick += 1
        p = self.player

        if self.dead_timer:
            self.dead_timer -= 1
            if not self.dead_timer:
                self._place_player(self.waypoints[self.spawn_wp])
                # The checkpoint tracker restarts from the respawn checkpoint. The tutorial's WaypointManager
                # doesn't, but the real game does: in a recording its progress bar slides back towards the
                # start during a long fall and turns straight back up to the respawn point on respawning.
                self.cur_wp = self.spawn_wp
                self.events.add("respawn")
            self.held = NONE
        else:
            # Update(): key release jumps, key held starts a turn (PlayerHumanController)
            if self.held != action and self.held in (LEFT, RIGHT):
                self._jump(self.held == RIGHT)
            if action == LEFT:
                self._start_turn(False)
            elif action == RIGHT:
                self._start_turn(True)
            self.held = action
            # FixedUpdate()
            if self.zip >= 0:
                pass  # hanging from a zipline
            elif self.is_turning:
                self._turn()
            else:
                self._balance()
            self.in_contact = False

        self._update_elevators()
        self._update_hovers()
        self._apply_effectors()
        self.world.Step(DT, VEL_ITERS, POS_ITERS)

        if self.dead_timer:
            return
        # OnCollisionStay2D / OnTriggerEnter2D
        for ce in p.contacts:
            c = ce.contact
            if not c.touching:
                continue
            other = c.fixtureB if c.fixtureA.body == p else c.fixtureA
            if not other.sensor:
                self.in_contact = True
            tag = (other.userData or {}).get("tag")
            if tag == "Death" or (tag == "Crush" and self.crush_active(other.userData)):
                self._die()
                return
            if tag == "Finish":
                self.finished = True
                self.finish_tick = self.tick
                self.splits.append(self.tick)
                return
        if self._airborne_ticks:
            self._airborne_ticks -= 1
            self.in_contact = False
        if p.position.y < self.kill_y:
            self._die()
            return
        self._update_zipline()
        self._update_ground()
        self._update_waypoints()

    def _die(self):
        if self.zip >= 0:
            self.zip = -1
            self.player.gravityScale = self.mv.gravity_scale
        self.deaths += 1
        self.events.add("death")
        self.dead_timer = int(round(self.mv.respawn_time * TICK_HZ))
        self.player.active = False
        self.is_turning = False

    def _in_zone(self, i, pos):
        """Waypoint.InZone: past the waypoint's line on its side (and above it, if it asks for that)."""
        side, above, yoff = self.wp_zone[i]
        wx, wy = self.waypoints[i]
        if side and (pos[0] - wx > 0) != (side > 0):
            return False
        if above and pos[1] - (wy + yoff) <= 0:
            return False
        return True

    def _update_waypoints(self):
        """WaypointManager.UpdateNavs: step forward when the next line is passed, back if we fall behind."""
        pos = tuple(self.player.position)
        cur = self.cur_wp
        if not self._in_zone(cur, pos):
            if cur > 0:
                self.cur_wp -= 1
        elif cur < len(self.waypoints) - 1 and self._in_zone(cur + 1, pos):
            self.cur_wp += 1
            if self.wp_spawnable[cur + 1] and cur + 1 > self.spawn_wp:
                self.spawn_wp = cur + 1
                self.splits.append(self.tick)

    def _update_zipline(self):
        """Latch onto a zipline the moment any part of the player touches it, slide to the bottom, drop off.

        Endpoints and catch distance are from the game data.
        The slide is a frictionless bead on a wire (keeps its speed along the line, gravity adds to it):
        a recorded full ride took 1.74 s, and this model gives 1.74 s for the same hook point and speed.
        """
        p = self.player
        if self.zip < 0:
            if not self.ziplines:
                return
            touching = False
            for i, (top, bot, catch) in enumerate(self.ziplines):
                if not self._touches_segment(top, bot, catch):
                    continue
                touching = True
                if self._zip_released:
                    break  # just dropped off the end: no re-latch until clear of the line
                ux, uy = bot[0] - top[0], bot[1] - top[1]
                length = math.hypot(ux, uy)
                ux, uy = ux / length, uy / length
                a = p.GetWorldPoint(ZIP_ANCHOR)
                v = p.linearVelocity
                self.zip = i
                self._zip_speed = v.x * ux + v.y * uy
                self._zip_s = min(max((a.x - top[0]) * ux + (a.y - top[1]) * uy, 0.0), length - 1e-3)
                self._lock_rotation(True)
                p.gravityScale = 0.0
                self.is_turning = False
                break
            if not touching:
                self._zip_released = False
            if self.zip < 0:
                return
        top, bot, _ = self.ziplines[self.zip]
        ux, uy = bot[0] - top[0], bot[1] - top[1]
        length = math.hypot(ux, uy)
        ux, uy = ux / length, uy / length
        self._zip_speed += GRAVITY * self.mv.gravity_scale * uy * DT
        self._zip_s += self._zip_speed * DT
        if self._zip_s < 0.0:  # slid back up to the top anchor: stop there
            self._zip_s, self._zip_speed = 0.0, max(self._zip_speed, 0.0)
        if self._zip_s >= length:
            self.zip = -1
            self._zip_released = True
            self._lock_rotation(False)
            p.gravityScale = self.mv.gravity_scale
            p.linearVelocity = (ux * self._zip_speed, uy * self._zip_speed)
            return
        ax, ay = top[0] + ux * self._zip_s, top[1] + uy * self._zip_s
        p.transform = ((ax - ZIP_ANCHOR[0], ay - ZIP_ANCHOR[1]), 0.0)
        p.linearVelocity = (ux * self._zip_speed, uy * self._zip_speed)

    def _touches_segment(self, a, b, margin) -> bool:
        """Does the segment a-b pass within `margin` of the player's body box? (Any touch latches.)"""
        p = self.player
        la, lb = p.GetLocalPoint(a), p.GetLocalPoint(b)
        x0, x1 = -PLAYER_W / 2 - margin, PLAYER_W / 2 + margin
        y0, y1 = -margin, PLAYER_H + margin
        t0, t1 = 0.0, 1.0
        for s, d, lo, hi in ((la.x, lb.x - la.x, x0, x1), (la.y, lb.y - la.y, y0, y1)):
            if abs(d) < 1e-12:
                if s < lo or s > hi:
                    return False
                continue
            ta, tb = (lo - s) / d, (hi - s) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
        return True

    def _update_elevators(self):
        """Each lift moves by position, carrying a player who stands on it, as if parented to it: the player keeps
        their own velocity, so a lift stopping doesn't throw them up (in the real game, the player stays on the
        floor when the lift stops and can lean straight away; in real-game trials the sim's old velocity-driven
        lift flung them into a 0.4 s hop, 2026-10-20)."""
        for e in self.elevators:
            if not self.mv.lift_carries:
                self._step_elevator(e)  # the old way: the lift moves by velocity
                e["velocity"] = tuple(e["body"].linearVelocity)
                continue
            body = e["body"]
            old = body.position.copy()
            self._step_elevator(e)
            delta = body.linearVelocity * DT
            body.linearVelocity = (0, 0)
            e["velocity"] = (delta.x / DT, delta.y / DT)  # how it moves (its body has none: it is moved by position)
            if delta.lengthSquared == 0:
                continue
            body.transform = (old + delta, 0.0)
            if self.dead_timer:
                continue
            p = self.player
            for ce in p.contacts:
                c = ce.contact
                if not c.touching:
                    continue
                other = c.fixtureB if c.fixtureA.body == p else c.fixtureA
                if other.body != body or other.sensor:
                    continue
                points = c.worldManifold.points[:c.manifold.pointCount]
                if points and min(pt[1] for pt in points) < p.worldCenter.y:  # standing on it, not beside it
                    p.transform = (p.position + delta, p.angle)
                    break

    def hover_height(self, h: dict, tick: int | None = None) -> float:
        """Where an animated body (HOVERS) is at a tick (now if None): its world y."""
        if not self.mv.helicopter_hovers:
            return h["scene"][1]
        t = self.mv.lift_head_start + (self.tick if tick is None else tick) * DT
        return h["scene"][1] - h["base"] + hover_y(t) + h["offset"]

    def _update_hovers(self):
        """Animated scenery is moved by its animation, by position (in the game it has no Rigidbody2D: the Animator
        moves its Transform, and the colliders with it). offset: a shift a shadow sim puts on it (real.shadow)."""
        for h in self.hovers:
            body = h["body"]
            body.transform = ((h["scene"][0], self.hover_height(h)), 0.0)
            body.linearVelocity = (0, 0)

    def crush_active(self, zone) -> bool:
        """Whether a cab's crush zone is deadly right now: the cab is on its way to that zone's stop
        and within ELEVATOR_CRUSH_DIST of it."""
        e = self.elevators[zone["elevator"]]
        if e["timer"] > 0 or e["target"] != zone["side"]:
            return False
        return abs(e["points"][e["target"]][1] - e["body"].position.y) <= ELEVATOR_CRUSH_DIST

    @staticmethod
    def _step_elevator(e):
        body = e["body"]
        if e["timer"] > 0:
            e["timer"] -= DT
            body.linearVelocity = (0, 0)
            return
        tx, ty = e["points"][e["target"]]
        px, py = body.position
        dx, dy = tx - px, ty - py
        dist = math.hypot(dx, dy)
        step = e["speed"] * DT
        if dist <= step:
            body.linearVelocity = (dx / DT, dy / DT)
            e["target"] ^= 1
            e["timer"] = e["wait"]
        else:
            body.linearVelocity = (dx / dist * e["speed"], dy / dist * e["speed"])

    def _apply_effectors(self):
        for fx in self.effector_fixtures:
            eff = fx.userData["effector"]
            for ce in fx.body.contacts:
                c = ce.contact
                if not c.touching or (c.fixtureA is not fx and c.fixtureB is not fx):
                    continue
                other = c.fixtureB if c.fixtureA is fx else c.fixtureA
                ob = other.body
                if ob.type != B.b2_dynamicBody or other.sensor:
                    continue
                if eff["type"] == "AreaEffector2D":
                    ang = math.radians(eff["ForceAngle"]) + (0.0 if eff["UseGlobalAngle"] else fx.body.angle)
                    f = eff["ForceMagnitude"]
                    ob.ApplyForceToCenter((math.cos(ang) * f, math.sin(ang) * f), True)
                else:  # SurfaceEffector2D: push toward the surface speed along the contact tangent
                    n = c.worldManifold.normal
                    nx, ny = (n[0], n[1]) if c.fixtureA is fx else (-n[0], -n[1])
                    tx, ty = ny, -nx
                    vt = ob.linearVelocity[0] * tx + ob.linearVelocity[1] * ty
                    j = eff["ForceScale"] * ob.mass * (eff["Speed"] - vt)
                    ob.ApplyLinearImpulse((tx * j, ty * j), ob.worldCenter, True)
