"""Where things are in the real game, from a screen grab: the camera, Steve, the lifts and the helicopter, in the
sim's world coordinates. Each grab is matched against the level's own art (tools/render_level.py renders it from
the game's build, so it lines up with the sim's colliders to the centimetre).

    py -m real.locate --video "C:/Users/ogjec/Videos/Getawayshootout/2026-10-02 23-41-14.mp4" --first 342 --last 3060
        tracks a recorded run (--first: the frame the clock starts) -> real/tracks/<video>.json and .png

Each grab:
- the camera is predicted with the game's camera model: it eases toward Steve's feet at speed 5 (Vector3.Lerp),
  resting with them CAM_ABOVE_FEET below the middle of the screen;
- then the grab is matched against the art around that guess, using only pixels where there is art (not the
  parallax city behind, the HUD or Steve). Each axis is taken from the match only when the art in view pins it down
  (flat ground says nothing about left and right; a brick shaft repeats up and down), otherwise from the prediction;
- the lift cars and the helicopter, which move, are found along their paths (up and down).
Steve's feet and lean follow from his jeans and head on screen. Measured on the 2026-10-02 recording: the camera's
match is good to about 2 cm a frame, and his feet sit within a few cm of the sim's floors when he stands.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from real.body import UPRIGHT
from real.trials import SKIN_BGR, SKIN_TOLERANCE, jeans_blobs

MAPS = Path(__file__).resolve().parent.parent / "sim" / "maps"
TRACKS = Path(__file__).resolve().parent / "tracks"
SCREEN_PX_PER_M = 113.3  # 1920x1080 full screen: the art matches best at 113-113.5
CAM_SPEED = 5.0  # fitted 5.13 across, 4.89 up and down (the 2019 CameraMovement value is 5)
CAM_ABOVE_FEET = 2.156  # m: where the camera comes to rest above Steve's feet (fitted)
JEANS_ABOVE_FEET = 0.427  # m along his body: the middle of his jeans, standing on the street
MATCH_PPM = 16  # the art's own resolution: matching happens at one art pixel per pixel
SEARCH = 0.4  # m around the predicted camera (more after a long gap between grabs: see Tracker.update)
MAX_SEARCH = 2.0
MIN_SCORE, MIN_MARGIN = 0.5, 0.01  # a match counts if it scores this, and beats other positions along an axis by this
RECOVER_AFTER, RECOVER_RADIUS = 8, 2.0  # lost: after this many grabs in a row, search this wide (m)
RECOVER_SCORE, RECOVER_MARGIN = 0.85, 0.03  # and only take a clear match (a strip of street fits anywhere)
# When Steve dies the game shows "Player 1" and a skull, in white, on the left of the screen for about 3 s. Cut from
# a 1920x1200 grab (white pixels of it). It matches 0.98-1.00 over walls and 0.55-0.75 over clouds; elsewhere white
# office furniture reached 0.54 (real/live/2026-10-21 22-53-24)
KILL_MARKER = cv2.imread(str(Path(__file__).resolve().parent / "kill_marker.png"), cv2.IMREAD_GRAYSCALE)
DEATH_SCORE = 0.6
GREY = np.array([0.5, 0.3, 0.2], np.float32)  # B, G, R weights: keeps the blue walls apart from the dark bricks


def grey(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[..., :3].astype(np.float32) @ GREY)


class Level:
    """A map's art at MATCH_PPM: the still level, plus the things that move without changing their look."""

    def __init__(self, map_name: str = "Tower"):
        info = json.loads((MAPS / f"{map_name}_art.json").read_text())
        k = MATCH_PPM / info["ppm"]
        self.x0, self.y0, self.x1, self.y1 = info["bounds"]

        def load(png):
            img = cv2.imread(str(MAPS / png), cv2.IMREAD_UNCHANGED).astype(np.float32)
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            return grey(img), np.ascontiguousarray(img[..., 3] / 255)
        self.grey, self.alpha = load(f"{map_name}_art.png")
        self.objects = []
        for o in info.get("objects", []):
            g, a = load(o["png"])
            self.objects.append(dict(o, grey=g, alpha=a, lift=o["path"].endswith("/MapObj_Elevator")))

    def expected(self, cam, w, h, objects=None):
        """The grey art and its coverage (0-1) that a camera at cam (world m) shows in a w x h grab at MATCH_PPM.
        objects: {index: (x, y)} where each moving object is (its scene position moved there)."""
        M = np.array([[1, 0, w / 2 - (cam[0] - self.x0) * MATCH_PPM], [0, 1, h / 2 - (self.y1 - cam[1]) * MATCH_PPM]],
                     np.float32)
        g = cv2.warpAffine(self.grey, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        a = cv2.warpAffine(self.alpha, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        for i, pos in (objects or {}).items():
            o = self.objects[i]
            og, oa, (ow, oh) = self._object_view(o, pos, cam, w, h)
            if og is None:
                continue
            g = og * oa + g * (1 - oa)
            a = np.maximum(a, oa)
        return g, a

    def _object_view(self, o, pos, cam, w, h):
        bx0, _, bx1, by1 = o["bounds"]
        left = w / 2 + (bx0 + pos[0] - o["scene_pos"][0] - cam[0]) * MATCH_PPM
        top = h / 2 - (by1 + pos[1] - o["scene_pos"][1] - cam[1]) * MATCH_PPM
        oh, ow = o["grey"].shape
        if left > w or top > h or left + ow < 0 or top + oh < 0:
            return None, None, (ow, oh)
        M = np.array([[1, 0, left], [0, 1, top]], np.float32)
        return (cv2.warpAffine(o["grey"], M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0),
                cv2.warpAffine(o["alpha"], M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0), (ow, oh))


def steve_on_screen(img: np.ndarray, px_per_m: float = SCREEN_PX_PER_M, near=None) -> dict | None:
    """Steve in a full-size grab: jeans and head (px), lean (deg, + = right) and where his feet are (px)."""
    half = np.ascontiguousarray(img[::2, ::2])
    jeans = [b for b in jeans_blobs(img, half=half) if b[2] >= 300 and b[1] > 150]
    if not jeans:
        return None
    j = min(jeans, key=lambda b: math.dist(b[:2], near)) if near is not None else jeans[0]
    # his head is within 2 m of his jeans: only look there (a third of the time of the whole screen)
    r = int(2.3 * px_per_m / 2)
    x0, y0 = max(int(j[0] / 2) - r, 0), max(int(j[1] / 2) - r, 0)
    around = np.ascontiguousarray(half[y0:int(j[1] / 2) + r, x0:int(j[0] / 2) + r])
    heads = [(x + 2 * x0, y + 2 * y0, n) for x, y, n in jeans_blobs(None, SKIN_BGR, SKIN_TOLERANCE, half=around)]
    heads = [b for b in heads if b[2] >= 1500 and b[1] > 150]
    h = min(heads, key=lambda b: math.dist(b[:2], j[:2])) if heads else None
    if h is not None and math.dist(h[:2], j[:2]) > 2.0 * px_per_m:
        h = None
    lean = (math.atan2(h[0] - j[0], j[1] - h[1]) - math.atan2(UPRIGHT[0], -UPRIGHT[1])) if h else 0.0
    up = JEANS_ABOVE_FEET * px_per_m
    return dict(jeans=list(j[:2]), head=list(h[:2]) if h else None, lean=math.degrees(lean),
                feet=[j[0] - up * math.sin(lean), j[1] + up * math.cos(lean)])


def death_marker(img: np.ndarray) -> float:
    """How clearly a full-size grab shows the game's death notice (see KILL_MARKER), 0-1."""
    H, W = img.shape[:2]
    region = img[int(0.66 * H):int(0.80 * H), :int(0.22 * W)]
    white = ((region.min(axis=2) > 200) * 255).astype(np.uint8)
    s = H / 1200  # the notice was cut from a 1200-high screen
    t = KILL_MARKER if s == 1 else cv2.resize(KILL_MARKER, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    if white.shape[0] < t.shape[0] or white.shape[1] < t.shape[1] or not white.any():
        return 0.0
    res = cv2.matchTemplate(white, t, cv2.TM_CCOEFF_NORMED)
    return float(np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).max())


def hidden(w, h, steve=None, s=1.0) -> np.ndarray:
    """1 where a grab (w x h, at s of full size) shows the level: not the HUD, the key overlay or Steve."""
    m = np.ones((h, w), np.float32)
    m[: int(0.13 * h)] = 0  # pause button, progress bar, clock
    m[: int(0.45 * h), int(0.66 * w):] = 0  # an OBS key overlay, when recording
    if steve is not None:
        pts = [p for p in (steve["jeans"], steve["head"], steve["feet"]) if p is not None]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, x1 = (min(xs) - 80) * s, (max(xs) + 80) * s
        y0, y1 = (min(ys) - 80) * s, (max(ys) + 40) * s
        m[max(int(y0), 0): max(int(math.ceil(y1)), 0), max(int(x0), 0): max(int(math.ceil(x1)), 0)] = 0
    return m


def _peak(res):
    """Best position in a match result, sub-pixel, and how clearly it beats the rest along x and along y."""
    iy, ix = np.unravel_index(np.argmax(res), res.shape)
    best = float(res[iy, ix])

    def sub(l, r):
        d = l - 2 * best + r
        return 0.5 * (l - r) / d if d < 0 else 0.0

    def margin(profile, i):
        rest = np.r_[profile[:max(i - 2, 0)], profile[i + 3:]]
        return float(best - rest.max()) if rest.size else 0.0
    fy = iy + (sub(res[iy - 1, ix], res[iy + 1, ix]) if 0 < iy < res.shape[0] - 1 else 0)
    fx = ix + (sub(res[iy, ix - 1], res[iy, ix + 1]) if 0 < ix < res.shape[1] - 1 else 0)
    return fx, fy, best, (margin(res[iy, :], ix), margin(res[:, ix], iy))


def _match(image, template, mask):
    res = cv2.matchTemplate(image, np.ascontiguousarray(template), cv2.TM_CCOEFF_NORMED,
                            mask=np.ascontiguousarray(mask))
    return np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)


class Tracker:
    """Follows the camera, Steve and the moving objects through a run, one grab at a time."""

    def __init__(self, level: Level, cam, px_per_m: float = SCREEN_PX_PER_M, anchors=(),
                 cam_above: float = CAM_ABOVE_FEET):
        """anchors: where Steve can appear out of nowhere (respawn points, his feet), to look for him there when lost.
        px_per_m, cam_above: the screen's scale and where the camera rests above his feet (real.live --calibrate)."""
        self.level, self.px_per_m, self.cam_above = level, px_per_m, cam_above
        self.cam = np.array(cam, float)
        self.objects = {i: tuple(o["scene_pos"]) for i, o in enumerate(level.objects)}
        self.offsets = {}  # object: how far it was from its given guess when last seen (m, up)
        self.steve = None
        self.anchors = [tuple(a) for a in anchors]
        self.bad = 0  # grabs in a row that pinned down neither axis

    def _match(self, small, look, centre, objects, radius, hide=None):
        """Match the grab against the art (with the objects drawn where given) around a camera at centre:
        (camera, score, (margin x, margin y), [x usable, y usable], cover)."""
        h, w = small.shape
        E, A = self.level.expected(centre, w, h, objects)
        r = max(int(math.ceil(radius * MATCH_PPM)), 1)
        mask = (A[r:h - r, r:w - r] > 0.8).astype(np.float32) * look[r:h - r, r:w - r]
        if hide is not None:
            mask *= hide[r:h - r, r:w - r]
        cover = float(mask.mean())
        if cover < 0.03:
            return np.array(centre, float), -1.0, (0.0, 0.0), [False, False], cover
        fx, fy, score, margins = _peak(_match(small, E[r:h - r, r:w - r], mask))
        cam = np.array([centre[0] - (fx - r) / MATCH_PPM, centre[1] + (fy - r) / MATCH_PPM])
        used = [score > MIN_SCORE and margins[0] > MIN_MARGIN, score > MIN_SCORE and margins[1] > MIN_MARGIN]
        return cam, score, margins, used, cover

    def _object_band(self, w, h, cam, guess):
        """0 where the moving objects may be (around their guesses, lifts all along their way), else 1."""
        m = np.ones((h, w), np.float32)
        for i, pos in guess.items():
            o = self.level.objects[i]
            bx0, by0, bx1, by1 = o["bounds"]
            dx, dy = pos[0] - o["scene_pos"][0], pos[1] - o["scene_pos"][1]
            reach = 3.0 if o["lift"] else 0.6
            x0 = int(w / 2 + (bx0 + dx - cam[0]) * MATCH_PPM) - 2
            x1 = int(w / 2 + (bx1 + dx - cam[0]) * MATCH_PPM) + 2
            y0 = int(h / 2 - (by1 + dy + reach - cam[1]) * MATCH_PPM)
            y1 = int(h / 2 - (by0 + dy - reach - cam[1]) * MATCH_PPM)
            m[max(y0, 0): max(y1, 0), max(x0, 0): max(x1, 0)] = 0
        return m

    def update(self, img: np.ndarray, dt: float = 1 / 60, object_guess: dict | None = None,
               steve_guess=None) -> dict:
        """img: a full-size BGR grab, dt s after the last one. object_guess: {index: (x, y)} to search around
        (e.g. the sim's lifts), else around where they were last seen. steve_guess: where Steve's feet should be
        (world m, e.g. the shadow's): with more than one Steve on screen (his body stays where he died), the one
        nearest it; else the one nearest where he was last seen."""
        H, W = img.shape[:2]
        pxm = self.px_per_m
        near = self.steve["jeans"] if self.steve else None
        if steve_guess is not None:
            near = [W / 2 + (steve_guess[0] - self.cam[0]) * pxm,
                    H / 2 - (steve_guess[1] + JEANS_ABOVE_FEET - self.cam[1]) * pxm]
        steve = steve_on_screen(img, pxm, near)
        # predict: the camera eased toward Steve's feet for dt (n game frames, each moving it CAM_SPEED/60 there).
        # If he stood still all along, it moved b/(1-b) times his offset now; if he moved at a steady speed, about
        # CAM_SPEED * dt times it. The two agree for grabs close together; after a long gap (the grabbing stalled
        # 0.4 s once, with Steve in a lift, and the first guess put the camera 3.7 m off) aim between them and
        # search wide enough for either
        pred, radius = self.cam.copy(), SEARCH
        if steve is not None:
            fx, fy = steve["feet"]
            off = np.array([(fx - W / 2) / pxm, (H / 2 - fy) / pxm + self.cam_above])
            b = 1 - (1 - CAM_SPEED / 60) ** (dt * 60)
            hi = b / (1 - b)
            lo = min(CAM_SPEED * dt, hi)
            pred = self.cam + (lo + hi) / 2 * off
            radius = min(SEARCH + (hi - lo) / 2 * float(np.hypot(*off)), MAX_SEARCH)
        s = MATCH_PPM / pxm
        small = grey(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        h, w = small.shape
        look = hidden(w, h, steve, s)
        # given guesses (the sim's lifts) are shifted by how far off they were when the object was last seen
        guess = dict(self.objects)
        guess.update({i: (x, y + self.offsets.get(i, 0.0)) for i, (x, y) in (object_guess or {}).items()})
        # the still art first, leaving out wherever the moving objects may be (one drawn a little off spoils the
        # match); then, for an axis that leaves open, with the objects drawn where they are thought to be
        match, score, margins, used, cover = self._match(small, look, pred, None, radius,
                                                         self._object_band(w, h, pred, guess))
        if not all(used):
            m2, s2, _, u2, _ = self._match(small, look, pred, guess, radius)
            for ax in (0, 1):
                if not used[ax] and u2[ax]:
                    match[ax], used[ax] = m2[ax], True
            score = max(score, s2)
        recovered = False
        self.bad = 0 if any(used) else self.bad + 1
        if self.bad >= RECOVER_AFTER:
            found = self._recover(small, look, pred, steve, W, H)
            if found is not None:
                match, used, recovered, self.bad = found, [True, True], True, 0
        cam = np.array([match[0] if used[0] else pred[0], match[1] if used[1] else pred[1]])
        # the moving objects, up and down their paths, around the guesses
        seen = {}
        for i, pos in guess.items():
            y, sc = self._find_object(small, look, cam, i, pos)
            if y is not None and sc > MIN_SCORE:
                seen[i] = (pos[0], y)
                self.objects[i] = (pos[0], y)
                if object_guess and i in object_guess:
                    self.offsets[i] = y - object_guess[i][1]
        self.cam, self.steve = cam, steve
        out = dict(cam=cam.tolist(), pred=pred.tolist(), match=match.tolist(), used=used, score=score,
                   margins=list(margins), cover=cover, objects={str(i): list(p) for i, p in seen.items()},
                   death_marker=round(death_marker(img), 3))
        if recovered:
            out["recovered"] = True
        if steve is not None:
            fx, fy = steve["feet"]
            out["steve"] = dict(steve, feet_world=[cam[0] + (fx - W / 2) / pxm, cam[1] + (H / 2 - fy) / pxm])
        return out

    def _recover(self, small, look, pred, steve, W, H):
        """Lost (e.g. Steve died and came back at a checkpoint): look around the prediction and around every place
        he can reappear. The camera found, or None."""
        pxm = self.px_per_m
        if steve is not None:  # where the camera would be if his feet were at the anchor
            fx, fy = steve["feet"]
            to_cam = np.array([-(fx - W / 2) / pxm, (fy - H / 2) / pxm])
        else:
            to_cam = np.array([0.0, self.cam_above])
        best = None
        for c in [pred] + [np.array(a) + to_cam for a in self.anchors]:
            cam, score, margins, _, _ = self._match(small, look, c, None, RECOVER_RADIUS)
            if score > RECOVER_SCORE and min(margins) > RECOVER_MARGIN and (best is None or score > best[1]):
                best = (cam, score)
        return None if best is None else best[0]

    def _find_object(self, small, look, cam, i, pos, radius=1.5):
        o = self.level.objects[i]
        h, w = small.shape
        oh, ow = o["grey"].shape
        bx0, _, _, by1 = o["bounds"]
        sx, sy = o["scene_pos"]
        left = int(round(w / 2 + (bx0 + pos[0] - sx - cam[0]) * MATCH_PPM))
        top = h / 2 - (by1 + pos[1] - sy - cam[1]) * MATCH_PPM  # top edge at the guess
        y0, y1 = int(math.floor(top - radius * MATCH_PPM)), int(math.ceil(top + radius * MATCH_PPM)) + oh
        y0, y1, x0, x1 = max(y0, 0), min(y1, h), max(left - 2, 0), min(left + ow + 2, w)
        if y1 - y0 < oh or x1 - x0 < ow:
            return None, -1.0  # not wholly in view
        ty, tx = int(min(max(round(top), 0), h - oh)), int(min(max(left, 0), w - ow))
        mask = (o["alpha"] > 0.8).astype(np.float32) * look[ty:ty + oh, tx:tx + ow]
        if mask.mean() < 0.2:
            return None, -1.0
        res = _match(np.ascontiguousarray(small[y0:y1, x0:x1]), o["grey"], mask)
        _, fy, best, _ = _peak(res)
        return float(sy + (h / 2 - (y0 + fy)) / MATCH_PPM + cam[1] - by1), best


def respawn_points(game) -> list[tuple[float, float]]:
    """Where Steve reappears after dying (the map's spawnable checkpoints): his feet."""
    return [tuple(w) for w, ok in zip(game.waypoints, game.wp_spawnable) if ok]


def lift_schedule(map_name: str = "Tower", seconds: float = 90.0):
    """The sim's lift and helicopter heights from the start of a run: (dt, [(path, x)], heights[tick][object])."""
    from sim.world import DT, Game
    g = Game(map_name)
    g.reset()
    moving = g.elevators + g.hovers
    ys = []
    for _ in range(int(seconds / DT)):
        ys.append([e["body"].position.y for e in moving])
        g.step(0)
    return DT, [(e["path"], e["body"].position.x) for e in moving], np.array(ys)


def sim_object_guess(level: Level, schedule, t: float) -> dict:
    """{object index: (x, y)} for the lifts and the helicopter where the sim has them t s into a run."""
    dt, lifts, ys = schedule
    k = min(max(int(round(t / dt)), 0), len(ys) - 1)
    by_path = {p: (x, ys[k][j]) for j, (p, x) in enumerate(lifts)}
    return {i: by_path[o["path"]] for i, o in enumerate(level.objects) if o["path"] in by_path}


def track_video(path: str, first: int, last: int | None, px_per_m: float, map_name: str = "Tower") -> dict:
    from sim.world import Game
    level = Level(map_name)
    g = Game(map_name)
    g.reset()
    start = (g.player.position.x, g.player.position.y + CAM_ABOVE_FEET)
    schedule = lift_schedule(map_name)
    v = cv2.VideoCapture(path)
    fps = v.get(cv2.CAP_PROP_FPS) or 60.0
    v.set(cv2.CAP_PROP_POS_FRAMES, first)
    tracker = Tracker(level, start, px_per_m, anchors=respawn_points(g))
    rows = []
    f = first
    while last is None or f <= last:
        ok, img = v.read()
        if not ok:
            break
        t = (f - first) / fps
        row = tracker.update(img, 1 / fps, sim_object_guess(level, schedule, t))
        rows.append(dict(frame=f, t=t, **row))
        f += 1
        if f % 300 == 0:
            print(f"frame {f}  camera ({row['cam'][0]:.2f}, {row['cam'][1]:.2f})  match {row['score']:.2f}  "
                  f"from the art: {'x' if row['used'][0] else '-'}{'y' if row['used'][1] else '-'}", flush=True)
    return dict(video=path, first=first, fps=fps, px_per_m=px_per_m, rows=rows)


def draw_track(track: dict, out: Path, map_name: str = "Tower"):
    """Steve's feet on the level art: green where the camera came from the art on both axes, red otherwise."""
    info = json.loads((MAPS / f"{map_name}_art.json").read_text())
    art = cv2.imread(str(MAPS / f"{map_name}_art.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    a = art[..., 3:4] / 255
    img = (art[..., :3] * a + np.array([70, 50, 50]) * (1 - a)).astype(np.uint8)
    x0, _, _, y1 = info["bounds"]
    ppm = info["ppm"]

    def px(p):
        return int(round((p[0] - x0) * ppm)), int(round((y1 - p[1]) * ppm))
    pts = []
    for r in track["rows"]:
        if "steve" in r:
            p = px(r["steve"]["feet_world"])
            pts.append(p)
            cv2.circle(img, p, 3, (60, 220, 60) if all(r["used"]) else (40, 40, 230), -1)
    if pts:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 6 * ppm
        img = img[max(min(ys) - pad, 0): max(ys) + pad, max(min(xs) - pad, 0): max(xs) + pad]
    cv2.imwrite(str(out), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--first", type=int, required=True, help="frame the run's clock starts")
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--px-per-m", type=float, default=SCREEN_PX_PER_M)
    args = ap.parse_args()
    TRACKS.mkdir(exist_ok=True)
    track = track_video(args.video, args.first, args.last, args.px_per_m)
    stem = Path(args.video).stem
    (TRACKS / f"{stem}.json").write_text(json.dumps(track))
    draw_track(track, TRACKS / f"{stem}.png")
    rows = track["rows"]
    print(f"{len(rows)} frames: camera from the art on both axes in {sum(all(r['used']) for r in rows)}, "
          f"on neither in {sum(not any(r['used']) for r in rows)} -> {TRACKS / (stem + '.json')}")


if __name__ == "__main__":
    main()
