"""Viewer: play the sim yourself, or watch a replay / agent. Uses the exact env the agent trains on.

    py -m sim.play                       # Street
    py -m sim.play --map Tower
    py -m sim.play --agent runs/<batch>/ppo_tower.pt   # watch a trained agent (PPO or genetic)
    py -m sim.play --agent runs/<batch>/ppo_tower.pt --loop   # leave open during training: newest agent
    py -m sim.play --race                # the current batch's best agents on one screen (PPO blue, genetic red)
    py -m sim.play --replay runs/pb_Street.json

Controls (like the game): HOLD a direction to lean, RELEASE to jump.
  left = A / W / Left,  right = D / E / Right
  R restart   1-4 map   Tab model's-eye view   Z zoom out/in   F 4x speed   Space pause   Esc quit
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import Box2D as B
import pygame

from .env import OBS_LABELS, RAY_ANGLES, RAY_LEN, GetawayEnv
from .world import DT, LEFT, MAP_NAMES, NONE, PX_PER_M, RIGHT, TICK_HZ

W, H = 1280, 720
# The camera, measured from a 60 fps recording of the current game in a 1920x1080 window (game px):
GAME_VIEW_H = 1080  # px of the game the window shows top to bottom
ZOOMS = (1.0, 0.5)  # Z switches between the game's view and a wide one showing twice as much
CAM_ABOVE = 270  # px: it aims this far above the player's feet (the player sits two thirds of the way down)
# It trails the player: each frame it moves CAM_SPEED * deltaTime of the way there (Vector3.Lerp), the 2019
# build's CameraMovement speed. Filming the sim's replay of the recording with it puts the player within
# ~35 px of where the recording shows them, all through it.
CAM_SPEED = 5.0
RUNS = Path(__file__).resolve().parent.parent / "runs"  # your personal bests live here; batches in subfolders
LEFT_KEYS = (pygame.K_a, pygame.K_w, pygame.K_LEFT)
RIGHT_KEYS = (pygame.K_d, pygame.K_e, pygame.K_RIGHT)

COL_SOLID = (70, 72, 84)
COL_DYNAMIC = (214, 150, 50)
COL_KINEMATIC = (90, 140, 210)
COL_DEATH = (220, 60, 60)
COL_FINISH = (60, 190, 90)
COL_EFFECTOR = (120, 180, 250)
COL_SENSOR = (170, 170, 190)
BLUE, RED = (60, 120, 220), (215, 55, 55)
METHOD_COLOURS = {"ppo": BLUE, "genetic": RED}  # the same agent is always the same colour
SPARE_COLOURS = (BLUE, RED, (60, 170, 90), (230, 160, 40), (150, 90, 210), (40, 180, 190), (230, 100, 180),
                 (150, 110, 60))

HELP_HUMAN = ("HOLD to lean, RELEASE to jump   A/W/<- left  D/E/-> right   R restart  1-4 map  "
              "Tab model view  Z zoom  F fast  Space pause")
HELP_WATCH = "R restart   Tab model view   Z zoom   F fast   Space pause"
PLACES = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th")


class RecordedRun:
    """An agent's recorded run (a .json of its actions, e.g. runs/02-your-run/ppo_tower_run.json) played back in
    place of the agent. The sim is exact, so it plays out exactly as the agent did."""

    def __init__(self, path):
        run = json.loads(Path(path).read_text())
        self.actions = run["actions"]
        self.i = 0
        self.info = {"method": Path(path).stem.split("_")[0], "finished": True, "time_s": run["time_s"],
                     "trained_steps": 0, "recorded": True}

    def act(self, obs):
        a = self.actions[self.i] if self.i < len(self.actions) else 0
        self.i += 1
        return a


class View:
    """One player (you, a replay or an agent) and the game it is playing. Several can share the window:
    each runs its own copy of the map, and they are drawn over one another like a race."""

    def __init__(self, app, args, agent=None):
        self.app = app
        self.screen = app.screen
        self.w, self.h = app.screen.get_size()
        self.font, self.small, self.mid, self.big = app.font, app.small, app.mid, app.big
        self.colour = BLUE
        self.replay = json.loads(Path(args.replay).read_text()) if args.replay else None
        self.map = self.replay["map"] if self.replay else args.map
        self.policy = None
        self.agent_info = None
        self.agent_path = Path(agent) if agent else None
        self.agent_mtime = None
        if self.agent_path:
            # a trained agent plays under exactly the settings it was trained and scored with (its batch's rewards)
            from common.workers import BATCHES, TRAIN_ENV
            batch = BATCHES.get(self.agent_path.resolve().parent.name)
            self.map = TRAIN_ENV["map"]
            self.env = GetawayEnv(**{**TRAIN_ENV, **({"rewards": batch["rewards"]} if batch else {})})
        else:
            # you play with a decision every physics tick; a replay uses whatever it was recorded with
            self.env = GetawayEnv(map=self.map, action_repeat=self.replay["action_repeat"] if self.replay else 1)
        self.restart()

    @property
    def mode(self):
        i = self.agent_info
        if i is not None:
            when = (f", run {i['test_run']}" if "test_run" in i else
                    f", gen {i['generation']}" if "generation" in i else "")
            return f"AGENT ({i.get('method', '?')}{when})"
        return "REPLAY" if self.replay else "HUMAN"

    def load_agent(self):
        """(Re)loads the agent file if it has changed, so a viewer left open follows training as it improves."""
        from common.policy import load_agent
        if self.agent_path.suffix == ".json":
            self.policy = RecordedRun(self.agent_path)  # from the start again every time
            self.agent_info = self.policy.info
            return
        try:
            mtime = self.agent_path.stat().st_mtime
            if mtime == self.agent_mtime:
                return
            self.policy, self.agent_info = load_agent(self.agent_path)
            self.env.lift_speed_seen = bool(self.agent_info.get("lift_speed_seen", False))  # as it was trained
            self.agent_mtime = mtime
        except Exception:
            if self.policy is None:
                raise
            return  # caught the file mid-save: keep playing the one already loaded

    @property
    def name(self):
        """Short name for a race: the agent's batch and method."""
        i = self.agent_info
        return f"{self.agent_path.resolve().parent.name} {i.get('method', '?')}" if i is not None else self.mode

    @property
    def title(self):
        i = self.agent_info
        if i.get("recorded"):
            return "%s agent's recorded run (finishes in %.2f s)" % (i["method"], i["time_s"])
        result = ("finishes in %.2f s" % i["time_s"] if i.get("finished")
                  else "reaches %.0f%% of the route" % (i.get("progress", 0.0) * 100))
        return "%s agent after %.2fM training steps (%s)" % (i.get("method", "?"), i.get("trained_steps", 0) / 1e6, result)

    def restart(self):
        if self.agent_path:
            self.load_agent()
        self.obs, _ = self.env.reset(self.map)
        self.g = self.env.game
        self.stepping = False
        self.actions: list[int] = []
        self.decision = 0
        self.action = NONE
        self.total_reward = 0.0
        self.done = False
        self.result = ""
        try:
            self.pb = json.loads((RUNS / f"pb_{self.map}.json").read_text())
        except (OSError, ValueError):
            self.pb = None

    def human_action(self):
        keys = pygame.key.get_pressed()
        if any(keys[k] for k in LEFT_KEYS):
            return LEFT
        if any(keys[k] for k in RIGHT_KEYS):
            return RIGHT
        return NONE

    def choose(self):
        if self.replay:
            acts = self.replay["actions"]
            return acts[self.decision] if self.decision < len(acts) else NONE
        if self.policy:
            return int(self.policy.act(self.obs))
        return self.human_action()

    def advance(self):
        env = self.env
        if not self.stepping:
            self.action = self.choose()
            self.actions.append(self.action)
            env.begin_step(self.action)
            self.stepping = True
        if not env.tick():
            self.obs, r, term, trunc, _ = env.end_step()
            self.total_reward += r
            self.stepping = False
            self.decision += 1
            if term or trunc:
                self.done = True
                self.on_done()

    def on_done(self):
        g = self.g
        if not g.finished:
            if g.tick >= self.env.time_limit_ticks:
                self.result = "OUT OF TIME"
            else:
                self.result = "STUCK (no progress for %g s)" % (self.env.stall_ticks / TICK_HZ)
            return
        self.result = "GETAWAY!"
        if self.mode == "HUMAN" and (self.pb is None or g.time_s < self.pb["time_s"]):
            RUNS.mkdir(exist_ok=True)
            run = {"map": self.map, "action_repeat": self.env.action_repeat, "time_s": g.time_s,
                   "splits_s": [t * DT for t in g.splits], "actions": self.actions,
                   "physics": "2026-10-22"}  # the sim's physics version (common/workers.py PHYSICS)
            (RUNS / f"pb_{self.map}.json").write_text(json.dumps(run))
            self.result = "NEW PB!"

    def to_finish(self):
        return 0.0 if self.g.finished else self.g.dist_to_finish()

    # --- drawing ------------------------------------------------------------

    @property
    def ppm(self):
        """Window pixels per Box2D unit."""
        return ZOOMS[self.app.zoom] * H / GAME_VIEW_H * PX_PER_M

    def to_screen(self, x, y):
        cam = self.app.cam
        return (x - cam[0]) * self.ppm + self.w / 2, self.h / 2 - (y - cam[1]) * self.ppm

    def fixture_color(self, fx):
        ud = fx.userData or {}
        tag = ud.get("tag")
        if tag == "Death" or (tag == "Crush" and self.g.crush_active(ud)):
            return COL_DEATH, 2
        if tag == "Finish":
            return COL_FINISH, 0
        if ud.get("effector"):
            return COL_EFFECTOR, 2
        if fx.sensor:
            return COL_SENSOR, 1
        t = fx.body.type
        if fx.body.userData == "hover":  # animated scenery (the helicopter): drawn like the rest of the level
            return COL_SOLID, 0
        if t == B.b2_dynamicBody:
            return COL_DYNAMIC, 0
        if t == B.b2_kinematicBody:
            return COL_KINEMATIC, 0
        return COL_SOLID, 0

    def draw_scene(self):
        """The map as it is in this player's game (lifts, loose furniture, which checkpoints are passed)."""
        sc = self.screen
        sc.fill((182, 214, 236))
        g = self.g
        for body in g.world.bodies:
            if body.userData == "player":
                continue
            xf = body.transform
            for fx in body.fixtures:
                col, width = self.fixture_color(fx)
                pts = [self.to_screen(*(xf * v)) for v in fx.shape.vertices]
                if max(p[0] for p in pts) < 0 or min(p[0] for p in pts) > self.w:
                    continue
                pygame.draw.polygon(sc, col, pts, width)
        for top, bot, _ in g.ziplines:
            pygame.draw.line(sc, (20, 20, 20), self.to_screen(*top), self.to_screen(*bot), 3)
        for i, (wx, wy) in enumerate(g.waypoints):
            if i == 0:
                continue
            side, above, yoff = g.wp_zone[i]
            passed = i <= g.cur_wp
            spawn = g.wp_spawnable[i]
            col = (60, 190, 90) if passed else ((250, 250, 250) if spawn else (150, 110, 200))
            base = wy + yoff
            top_y, bot_y = (base + 5.0, base) if above else (base + 3.0, base - 3.0)
            x_px = self.to_screen(wx, 0)[0]
            if x_px < -50 or x_px > self.w + 50:
                continue
            a, b = self.to_screen(wx, top_y), self.to_screen(wx, bot_y)
            pygame.draw.line(sc, col, a, b, 4 if spawn else 1)
            if side:  # little tick on the side that counts as "passed"
                pygame.draw.line(sc, col, a, (a[0] + 12 * side, a[1]), 4 if spawn else 1)
            if above:
                pygame.draw.line(sc, col, b, (b[0] + 12 * (side or 1), b[1]), 1)

    def draw_player(self):
        p = self.g.player
        xf = p.transform
        body = [self.to_screen(*(xf * v)) for v in p.fixtures[0].shape.vertices]
        pygame.draw.polygon(self.screen, self.colour, body)
        pygame.draw.polygon(self.screen, (20, 20, 30), body, 1)
        head = self.to_screen(*(xf * B.b2Vec2(0.0, 1.72)))
        pygame.draw.circle(self.screen, (240, 200, 160), head, int(0.27 * self.ppm))
        legs = [self.to_screen(*(xf * B.b2Vec2(sx, 0.0))) for sx in (-0.15, 0.15)]
        hip = self.to_screen(*(xf * B.b2Vec2(0.0, 0.8)))
        for l in legs:
            pygame.draw.line(self.screen, (40, 50, 90), hip, l, 5)
        if self.g.is_turning:
            t = self.small.render(f"lean {self.g.turning_time:.2f}s", True, (0, 0, 0))
            self.screen.blit(t, (head[0] - 30, head[1] - 34))

    def draw_rays(self):
        c = self.g.player.worldCenter
        for (frac, kind), a in zip(self.env.rays(), RAY_ANGLES):
            ex, ey = c.x + math.cos(a) * RAY_LEN * frac, c.y + math.sin(a) * RAY_LEN * frac
            col = (255, 60, 60) if kind == 2 else ((40, 220, 90) if kind == 1 else (255, 255, 255))
            pygame.draw.line(self.screen, col, self.to_screen(c.x, c.y), self.to_screen(ex, ey), 1)
            if kind:
                pygame.draw.circle(self.screen, col, self.to_screen(ex, ey), 3)

    def draw_obs_panel(self):
        panel = pygame.Surface((250, self.h - 20), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        self.screen.blit(panel, (self.w - 260, 10))
        for i, (name, v) in enumerate(zip(OBS_LABELS, self.obs)):
            self.screen.blit(self.small.render(f"{name:<13}{v:+.2f}", True, (225, 225, 225)), (self.w - 250, 14 + i * 13))

    def draw_hud(self, slot=0):
        """This player's info box, action buttons and result. In a race each player gets its own box,
        tinted in its colour, in a row along the top (smaller boxes in rows of three for more than three)."""
        g = self.g
        sc = self.screen
        n = len(self.app.views)
        racing = n > 1
        self.draw_buttons(slot, n)
        if n > 3:
            return self.draw_compact_hud(slot)
        x0 = 10 + slot * 350
        box = pygame.Surface((340, 166), pygame.SRCALPHA)
        tint = tuple(c // 2 for c in self.colour) if self.agent_info is not None else (0, 0, 0)
        box.fill((*tint, 175 if self.agent_info is not None else 140))
        sc.blit(box, (x0, 10))
        if self.agent_info is not None:
            pygame.draw.rect(sc, self.colour, (x0, 10, 340, 166), 2)
        sc.blit(self.font.render(f"{g.map_name}  {self.mode}  deaths {g.deaths}", True, (255, 255, 255)), (x0 + 10, 16))
        t = g.time_s
        sc.blit(self.big.render(f"{int(t // 60)}:{t % 60:06.3f}", True,
                                (120, 255, 140) if g.finished else (255, 255, 255)), (x0 + 10, 36))
        pb_splits = (self.pb or {}).get("splits_s", [])
        y = 84
        for i, st in enumerate(g.splits[-3:]):
            idx = len(g.splits) - min(3, len(g.splits)) + i
            s = st * DT
            txt, col = f"split {idx + 1}: {s:6.2f}", (255, 255, 255)
            if idx < len(pb_splits):
                d = s - pb_splits[idx]
                txt += f"  {d:+.2f}"
                col = (120, 255, 140) if d <= 0 else (255, 120, 110)
            sc.blit(self.small.render(txt, True, col), (x0 + 10, y))
            y += 14
        pb = f"PB {self.pb['time_s']:.2f}" if self.pb else "PB --"
        sc.blit(self.small.render(f"{pb}   to finish {g.dist_to_finish() * PX_PER_M:.0f} px   reward {self.total_reward:+.1f}",
                                  True, (220, 220, 220)), (x0 + 10, 140))
        v = g.player.linearVelocity
        sc.blit(self.small.render(f"vx {v.x * PX_PER_M:+6.0f}  vy {v.y * PX_PER_M:+6.0f} px/s   lean {g.rotation:+5.1f} deg",
                                  True, (220, 220, 220)), (x0 + 10, 156))
        if slot == 0:
            sc.blit(self.small.render(HELP_HUMAN if self.mode == "HUMAN" else HELP_WATCH, True, (20, 20, 20)), (10, self.h - 16))
        if not self.done:
            return
        col = (120, 255, 140) if g.finished else (255, 120, 110)
        if racing:  # each player's result goes under its own box
            result = self.race_result()
            label = pygame.Surface((340, 30), pygame.SRCALPHA)
            label.fill((0, 0, 0, 175))
            sc.blit(label, (x0, 180))
            sc.blit(self.mid.render(result, True, col), (x0 + 10, 184))
        else:
            t = self.big.render(self.result, True, col)
            sc.blit(t, (self.w / 2 - t.get_width() / 2, 190))

    def race_result(self):
        g = self.g
        if not g.finished:
            return self.result
        ahead = sum(o.g.finished and o.g.time_s < g.time_s for o in self.app.views)
        return f"{PLACES[min(ahead, len(PLACES) - 1)]}  {g.time_s:.2f} s"

    def draw_buttons(self, slot, n):
        """The held-key buttons along the bottom, one pair per player."""
        cx = self.w * (slot + 1) // (n + 1)
        for a, lbl, off in ((LEFT, "<", -40), (RIGHT, ">", 40)):
            on = self.action == a
            lit = self.colour if n > 1 else (255, 210, 60)
            pygame.draw.rect(self.screen, lit if on else (60, 60, 60), (cx + off - 22, self.h - 58, 44, 44), border_radius=8)
            tx = self.big.render(lbl, True, (0, 0, 0) if on else (200, 200, 200))
            self.screen.blit(tx, (cx + off - tx.get_width() / 2, self.h - 60))

    def draw_compact_hud(self, slot):
        """A small box for big races: who it is, the clock, and the result once done."""
        g, sc = self.g, self.screen
        bw, bh = (self.w - 40) // 3, 52
        x0, y0 = 10 + (slot % 3) * (bw + 10), 10 + (slot // 3) * (bh + 8)
        box = pygame.Surface((bw, bh), pygame.SRCALPHA)
        box.fill((*(c // 2 for c in self.colour), 175))
        sc.blit(box, (x0, y0))
        pygame.draw.rect(sc, self.colour, (x0, y0, bw, bh), 2)
        sc.blit(self.font.render(f"{self.name}   deaths {g.deaths}", True, (255, 255, 255)), (x0 + 8, y0 + 5))
        t = g.time_s
        sc.blit(self.mid.render(f"{int(t // 60)}:{t % 60:06.3f}", True,
                                (120, 255, 140) if g.finished else (255, 255, 255)), (x0 + 8, y0 + 25))
        if self.done:
            res = self.mid.render(self.race_result().split(" (")[0], True, (120, 255, 140) if g.finished else (255, 120, 110))
            sc.blit(res, (x0 + bw - res.get_width() - 8, y0 + 25))


class App:
    """The window. Usually one player; with several agents they all run at once, drawn over each other."""

    def __init__(self, args):
        pygame.init()
        agents = list(args.agent or [])
        if args.race:  # the current batch's best agents (or a recorded run of one, where its network is gone)
            from common.workers import RUNS as BATCH_RUNS
            agents = [str(f) for m in ("ppo", "genetic")
                      for f in [BATCH_RUNS / f"{m}_tower.pt", BATCH_RUNS / f"{m}_tower_run.json"] if f.exists()][:2]
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 12)
        self.mid = pygame.font.SysFont("consolas", 20, bold=True)
        self.big = pygame.font.SysFont("consolas", 40, bold=True)
        self.loop = args.loop
        self.show_obs = args.overlay
        self.paused = self.fast = False
        self.zoom = 0
        self.cam = [0.0, 0.0]
        self.views = [View(self, args, agent) for agent in (agents or [None])]
        self.restart(reset_views=False)

    @property
    def done(self):
        return all(v.done for v in self.views)

    def focus(self) -> View:
        """Who the camera follows: whoever is furthest along the route and still playing."""
        playing = [v for v in self.views if not v.done] or self.views
        return min(playing, key=View.to_finish)

    def restart(self, reset_views=True):
        if reset_views:
            for v in self.views:
                v.restart()
        self.done_frames = 0
        used = []
        for v in self.views:
            colour = METHOD_COLOURS.get((v.agent_info or {}).get("method"))
            if colour is None or colour in used:
                colour = next((c for c in SPARE_COLOURS if c not in used), SPARE_COLOURS[len(used) % len(SPARE_COLOURS)])
            v.colour = colour
            used.append(colour)
        p = self.views[0].g.player.position
        self.cam = [p.x, p.y + CAM_ABOVE / PX_PER_M]
        titles = [v.title for v in self.views if v.agent_info is not None]
        pygame.display.set_caption("Getaway Shootout - " + ("  vs  ".join(titles) if titles else "speedrun sim"))

    def follow(self, dt):
        p = self.focus().g.player.position
        for axis, target in ((0, p.x), (1, p.y + CAM_ABOVE / PX_PER_M)):
            self.cam[axis] += (target - self.cam[axis]) * min(CAM_SPEED * dt, 1.0)

    def draw(self):
        focus = self.focus()
        focus.draw_scene()
        for v in sorted(self.views, key=lambda v: v is focus):  # the one the camera follows goes on top
            if v.g.alive:
                v.draw_player()
                if self.show_obs:
                    v.draw_rays()
        for slot, v in enumerate(self.views):
            v.draw_hud(slot)
        if self.show_obs:
            focus.draw_obs_panel()
        if self.done:
            again = "R / Space to go again" + ("" if focus.agent_path else ", 1-4 for another map")
            s = self.font.render(again, True, (0, 0, 0))
            self.screen.blit(s, (W / 2 - s.get_width() / 2, 238))

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_r:
                        self.restart()
                    elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        for v in self.views:
                            v.map = MAP_NAMES[e.key - pygame.K_1]
                            v.replay = None
                        self.restart()
                    elif e.key == pygame.K_TAB:
                        self.show_obs = not self.show_obs
                    elif e.key == pygame.K_f:
                        self.fast = not self.fast
                    elif e.key == pygame.K_z:
                        self.zoom = 1 - self.zoom
                    elif e.key == pygame.K_SPACE:
                        if self.done:
                            self.restart()
                        else:
                            self.paused = not self.paused
            if not self.paused:
                # the game runs physics at 50 Hz; the window at 50 fps keeps it 1:1 real time. Every player
                # steps the same number of ticks, so a race is a fair one.
                for _ in range(4 if self.fast else 1):
                    for v in self.views:
                        if not v.done:
                            v.advance()
                    self.follow(DT)
            if self.done and self.loop:
                self.done_frames += 1
                if self.done_frames > 2 * TICK_HZ:  # show the result for two seconds, then go again
                    self.restart()
            self.draw()
            pygame.display.flip()
            self.clock.tick(TICK_HZ)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="Street", choices=MAP_NAMES)
    ap.add_argument("--replay", help="play back your best run of a map (runs/pb_<Map>.json)")
    ap.add_argument("--agent", nargs="+", metavar="FILE",
                    help="a trained agent saved by ppo.train or genetic.train, or an agent's recorded run (.json); "
                         "give several to race them on one screen")
    ap.add_argument("--race", action="store_true",
                    help="race the best PPO agent (blue) against the best genetic agent (red) on one screen")
    ap.add_argument("--loop", action="store_true",
                    help="keep replaying; with --agent each replay uses the newest saved agent, so it follows training")
    ap.add_argument("--overlay", action="store_true", help="start with the model's-eye view on (same as pressing Tab)")
    args = ap.parse_args(argv)
    missing = [a for a in args.agent or [] if not Path(a).exists()]
    if missing:
        batches = sorted(p.name for p in RUNS.iterdir() if p.is_dir()) if RUNS.exists() else []
        ap.exit(1, f"no agent file {', '.join(missing)}\nagents are saved in their batch's folder, e.g. "
                   f"runs/<batch>/ppo_tower.pt. Batches: {', '.join(batches) or 'none yet'}\n")
    App(args).run()
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
