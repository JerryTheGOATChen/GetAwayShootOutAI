"""Gymnasium-style env over the Box2D sim. The viewer drives this same class, so what you watch is
exactly what the agent gets."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .world import DT, TICK_HZ, Game

N_RAYS = 16
RAY_LEN = 12.0  # Box2D units = 1349 px
RAY_ANGLES = [2 * math.pi * i / N_RAYS for i in range(N_RAYS)]
OBS_LABELS = (
    ["vx", "vy", "sin_rot", "cos_rot", "ang_vel", "grounded", "in_contact", "turning", "turn_right",
     "turn_time", "dead", "progress"]
    + [f"wp{i}_{a}" for i in range(3) for a in ("dx", "dy")]
    + [f"ray{i}_dist" for i in range(N_RAYS)] + [f"ray{i}_death" for i in range(N_RAYS)]
    + ["time", "lift_dy", "lift_vy"]  # the lifts run on a fixed schedule; these let an agent time them
)
OBS_SIZE = len(OBS_LABELS)
REWARD_PARTS = ("progress", "time", "deaths", "finish", "stall")  # where an episode's reward came from


@dataclass(frozen=True)
class Rewards:
    progress: float = 0.1  # per 112.4 px (one Box2D unit) closer to the finish along the waypoint route
    per_second: float = -0.1
    finish: float = 10.0
    finish_per_second_left: float = 0.0  # at the finish, this much more for every second left on the clock
    death: float = -1.0
    # Training only (with stall_limit_s): no progress for that long ends the episode as a failure. Treated as
    # "cut off, would have carried on", freezing in place cost so little that agents settled for it.
    stall: float = -5.0


@dataclass(frozen=True)
class Jitter:
    """Training only: the agent's key changes reach the game a little late, as they do in the real game (its
    keys are read once a frame at 60 Hz, the physics runs at 50 Hz, and the start can only be spotted to about a
    frame). Each episode every key change lands 0 to `delay` ticks late (the same for the whole episode), and
    each one has a further `extra` chance of one more tick. Without it the sim is exact to the tick, so agents
    learn tricks that only work with perfect timing."""
    delay: int = 2
    extra: float = 0.5


@dataclass(frozen=True)
class Wobble:
    """Training only: every few ticks the player is nudged a little (position, speed, lean), as an agent playing the
    real game through real.shadow sees its shadow nudged back onto the real run by each grab of the screen
    (measured on the 2026-10-02 recording: errors of a few cm, corrected about 30 times a second). Not up or down
    while standing (the shadow doesn't correct that either)."""
    every: int = 2  # ticks
    position: float = 0.03  # m, standard deviation, each way
    speed: float = 0.15  # m/s
    lean: float = 1.5  # degrees


class GetawayEnv:
    """Actions: 0 release/none, 1 hold left, 2 hold right. Releasing a held key jumps (like the game).

    map: "Street" | "Train" | "Town" | "Tower"
    action_repeat: physics ticks per decision (each tick is 0.02 s).
    """

    def __init__(self, map="Street", action_repeat: int = 2, time_limit_s: float = 90.0,
                 stall_limit_s: float | None = None, rewards: Rewards | None = None,
                 jitter: Jitter | None = None, seed: int | None = None, wobble: Wobble | None = None,
                 lift_speed_seen: bool = False):
        # Training aid: end the episode as a failure (Rewards.stall) when route progress has not improved for
        # this long. Keep it above the slowest lift's full cycle (15.3 s on Tower) so waiting is never punished.
        self.stall_ticks = int(stall_limit_s * TICK_HZ) if stall_limit_s else 0
        self.map = map
        self.action_repeat = action_repeat
        self.time_limit_ticks = int(time_limit_s * TICK_HZ)
        self.rewards = rewards or Rewards()
        self.jitter = jitter
        self.wobble = wobble
        # Lifts move by position (Movement.lift_carries), so their bodies have no velocity: agents trained before
        # 2026-10-20 saw lift_vy = 0 always. With this, lift_vy is how the nearest lift really moves.
        self.lift_speed_seen = lift_speed_seen
        self._rng = np.random.default_rng(seed)
        self._games: dict[str, Game] = {}
        self.game: Game | None = None
        self._stepping = False

    def reset(self, map=None, lead_in=(), start=None, seed: int | None = None):
        """Start an episode (on another map, if one is named). Returns the first observation and info.
        seed: restart the jitter's and wobble's random numbers, so the episode gets the same noise as another.

        lead_in: actions, one per physics tick, played before the episode begins, e.g. the start of a recorded
        run, so the episode starts from that exact moment of it. The sim has no randomness, so it is the same
        situation the recording was in. start: instead, a moment saved with Game.snapshot() (e.g. of a run played
        under older physics) to start from. Rewards, deaths and the stall limit count from there."""
        self.map = map or self.map
        g = self._games.get(self.map)
        if g is None:
            g = self._games[self.map] = Game(self.map)
        else:
            g.reset()
        self.game = g
        for a in lead_in:
            g.step(a)
        if start is not None:
            g.restore(start)
        # the key the game has (applied) and the one the agent last asked for, with the changes still on their way
        self._applied = self._asked = g.held
        self._on_the_way: list[tuple[int, int]] = []  # (tick it lands, action)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._delay = int(self._rng.integers(self.jitter.delay + 1)) if self.jitter else 0
        self._start_deaths = g.deaths
        self._dist = g.dist_to_finish()
        self._best_dist, self._best_tick = self._dist, g.tick
        self._stepping = False
        self.reward_parts = dict.fromkeys(REWARD_PARTS, 0.0)
        return self.observe(), self.info()

    def step(self, action):
        self.begin_step(action)
        while self.tick():
            pass
        return self.end_step()

    # The viewer calls these directly so it can draw every physics tick.
    def begin_step(self, action):
        assert not self._stepping
        self._stepping = True
        action = int(action)
        if action != self._asked:
            self._asked = action
            lands = self.game.tick + self._delay
            if self.jitter and self._rng.random() < self.jitter.extra:
                lands += 1
            if self._on_the_way:
                lands = max(lands, self._on_the_way[-1][0])  # keys arrive in the order they were pressed
            self._on_the_way.append((lands, action))
        self._left = self.action_repeat
        self._reward = 0.0
        self._terminated = False

    def tick(self) -> bool:
        g, r = self.game, self.rewards
        while self._on_the_way and self._on_the_way[0][0] <= g.tick:
            self._applied = self._on_the_way.pop(0)[1]
        g.step(self._applied)
        if self.wobble and g.alive and not g.finished and g.tick % self.wobble.every == 0:
            self._nudge(g, self.wobble)
        self._left -= 1
        self._earn("time", r.per_second * DT)
        if g.alive:
            d = g.dist_to_finish()
            self._earn("progress", (self._dist - d) * r.progress)
            self._dist = d
            if d < self._best_dist - 0.25:
                self._best_dist, self._best_tick = d, g.tick
        if "death" in g.events:
            self._earn("deaths", r.death)
        if "respawn" in g.events:
            self._dist = g.dist_to_finish()
        if g.finished:
            self._earn("finish", r.finish + r.finish_per_second_left * (self.time_limit_ticks - g.finish_tick) * DT)
            self._terminated = True
        elif self.stall_ticks and g.tick - self._best_tick >= self.stall_ticks:
            self._earn("stall", r.stall)
            self._terminated = True
        return self._left > 0 and not self._terminated and not self._out_of_time()

    def _nudge(self, g, w: Wobble):
        p, r = g.player, self._rng
        dx, dy = r.normal(0, w.position, 2)
        if g.grounded:
            dy = 0.0
        v = p.GetLinearVelocityFromLocalPoint((0, 0))
        omega = p.angularVelocity
        p.transform = ((p.position.x + dx, p.position.y + dy), p.angle + math.radians(r.normal(0, w.lean)))
        c = p.worldCenter - p.position
        vx, vy = v.x + r.normal(0, w.speed), v.y + r.normal(0, w.speed)
        p.linearVelocity = (vx - omega * c.y, vy + omega * c.x)
        p.awake = True

    def _earn(self, part, amount):
        self._reward += amount
        self.reward_parts[part] += amount

    def _out_of_time(self) -> bool:
        return self.game.tick >= self.time_limit_ticks

    def end_step(self):
        self._stepping = False
        trunc = not self._terminated and bool(self._out_of_time())
        return self.observe(), float(self._reward), self._terminated, trunc, self.info()

    def rays(self):
        """(fraction, kind) per ray from the player's center; kind 0 nothing, 1 solid, 2 death."""
        g = self.game
        c = g.player.worldCenter
        out = []
        for a in RAY_ANGLES:
            out.append(g.ray((c.x, c.y), (math.cos(a), math.sin(a)), RAY_LEN))
        return out

    def observe(self) -> np.ndarray:
        g = self.game
        p = g.player
        rot = math.radians(g.rotation)
        pos = (p.position.x, p.position.y)
        wps = []
        for i in range(3):
            w = g.waypoints[min(g.cur_wp + 1 + i, len(g.waypoints) - 1)]
            wps += [np.clip((w[0] - pos[0]) / 20.0, -3, 3), np.clip((w[1] - pos[1]) / 20.0, -3, 3)]
        rays = self.rays() if g.alive else [(1.0, 0)] * N_RAYS
        head = [p.linearVelocity.x / 10.0, p.linearVelocity.y / 10.0, math.sin(rot), math.cos(rot),
                np.clip(p.angularVelocity / 10.0, -3, 3), float(g.grounded), float(g.in_contact),
                float(g.is_turning), float(g.turn_right), min(g.turning_time / g.mv.lerp_turn_time, 2.0),
                g.dead_timer / (g.mv.respawn_time * TICK_HZ), 1.0 - g.dist_to_finish() / max(g.route_length, 1.0)]
        lift_dy = lift_vy = 0.0
        if g.elevators:
            e = min(g.elevators, key=lambda e: abs(e["body"].position.x - pos[0]))
            b = e["body"]
            lift_dy = float(np.clip((b.position.y - pos[1]) / 20.0, -2, 2))
            lift_vy = (e.get("velocity", (0.0, 0.0))[1] if self.lift_speed_seen else b.linearVelocity.y) / 4.0
        tail = [min(g.tick / self.time_limit_ticks, 1.0), lift_dy, lift_vy]
        return np.asarray(head + wps + [f for f, _ in rays] + [float(k == 2) for _, k in rays] + tail,
                          dtype=np.float32)

    def info(self) -> dict:
        g = self.game
        return {"time_s": g.time_s, "finished": g.finished, "deaths": g.deaths - self._start_deaths,
                "progress": 1.0 - self._best_dist / max(g.route_length, 1.0), "reward_parts": dict(self.reward_parts)}
