"""Render a map's level art from the original WebGL build into one picture in world coordinates, the way the
game's camera sees it, minus the parallax city behind and anything that moves (lifts, loose props, power-ups).
The lift cars and the helicopter are drawn separately, to be put wherever they are.

    py tools/render_level.py [--map Tower] [--ppm 32]   -> sim/maps/<Map>_art.png, <Map>_art_<i>.png, <Map>_art.json

The picture is what a live frame of the real game is matched against to find where its camera is.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from unitydump import Build

OUT = Path(__file__).resolve().parents[1] / "sim" / "maps"
SKIP_UNDER = ("Backgrounds", "PowerUps")  # parallax layers (Paralax script) and pick-ups
MOVES = ("Rigidbody2D", "Animator")  # anything with these, or under them, isn't where the scene puts it


def sprite_pixels(sprite):
    """The sprite's RGBA pixels (top row first), its pixels per unit, pivot (0-1, from bottom-left) and border
    (left, bottom, right, top, in pixels)."""
    img = np.asarray(sprite.image.convert("RGBA")).astype(np.float32) / 255
    b = sprite.m_Border
    return img, float(sprite.m_PixelsToUnits), (sprite.m_Pivot.x, sprite.m_Pivot.y), (b.x, b.y, b.z, b.w)


def nine_slice(img, border, w, h, tiled):
    """Unity's sliced / tiled draw: fixed corners, the edges and middle stretched (sliced) or repeated (tiled)
    to fill w x h pixels. Tiles start at the bottom-left of each part, cut off at its top and right."""
    H, W = img.shape[:2]
    l, b, r, t = (int(round(v)) for v in border)
    w, h = max(int(round(w)), 1), max(int(round(h)), 1)
    l, r = (l, r) if l + r <= w else (w * l // max(l + r, 1), w - w * l // max(l + r, 1))
    b, t = (b, t) if b + t <= h else (h * b // max(b + t, 1), h - h * b // max(b + t, 1))
    xs_src = [(0, l), (l, W - r), (W - r, W)]
    ys_src = [(0, t), (t, H - b), (H - b, H)]  # rows: top, middle, bottom
    xs_dst = [(0, l), (l, w - r), (w - r, w)]
    ys_dst = [(0, t), (t, h - b), (h - b, h)]
    out = np.zeros((h, w, 4), np.float32)
    for (sy0, sy1), (dy0, dy1), row in zip(ys_src, ys_dst, range(3)):
        for (sx0, sx1), (dx0, dx1), col in zip(xs_src, xs_dst, range(3)):
            src = img[sy0:sy1, sx0:sx1]
            dh, dw = dy1 - dy0, dx1 - dx0
            if dh <= 0 or dw <= 0 or src.size == 0:
                continue
            if not tiled or (row != 1 and col != 1):
                part = cv2.resize(src, (dw, dh), interpolation=cv2.INTER_NEAREST) if src.shape[:2] != (dh, dw) else src
            else:
                sh, sw = src.shape[:2]
                ry, rx = (-(-dh // sh) if row == 1 else 1), (-(-dw // sw) if col == 1 else 1)
                big = np.tile(src, (ry, rx, 1))
                big = big if row == 1 else cv2.resize(src, (big.shape[1], dh), interpolation=cv2.INTER_NEAREST)
                big = big if col == 1 else cv2.resize(big, (dw, big.shape[0]), interpolation=cv2.INTER_NEAREST)
                part = big[big.shape[0] - dh:, :dw]  # anchored bottom-left
            out[dy0:dy1, dx0:dx1] = part
    return out


def pieces(b: Build, map_name: str):
    """Every sprite drawn in the map: (sort key, image, local->world affine in metres, moves?, name, world xy)."""
    tm = next(o for o in b.env.objects if o.type.name == "TagManager").read()
    layer_rank = {l.uniqueID: i for i, l in enumerate(tm.m_SortingLayers)}
    root = b.roots("Map_" + map_name, "sharedassets1.assets")[0]
    path_of, moving_of = {}, {}
    out = []
    for n, (k, depth, W) in enumerate(b.walk(root)):
        d = b.tr[k]
        gk = (k[0], d.m_GameObject.m_PathID)
        g = b.go[gk]
        fk = (k[0], d.m_Father.m_PathID) if d.m_Father.m_PathID else None
        path_of[k] = (path_of[fk] + "/" if fk in path_of else "") + g.m_Name
        comps = list(b.components(gk))
        moving_of[k] = moving_of.get(fk, False) or any(t in MOVES for _, t in comps) or any(
            t == "MonoBehaviour" and b.script_name(ck) == "Elevator" for ck, t in comps)
        active = g.m_IsActive and path_of.get(("off",) + fk, True) if fk else g.m_IsActive
        path_of[("off",) + k] = active
        if not active or any(f"/{s}" in "/" + path_of[k] for s in SKIP_UNDER):
            continue
        for ck, t in comps:
            if t != "SpriteRenderer":
                continue
            r = b.obj[ck].read()
            if not r.m_Enabled or not r.m_Sprite.m_PathID:
                continue
            img, ppu, (px, py), border = sprite_pixels(r.m_Sprite.read())
            if r.m_DrawMode == 0:
                w_m, h_m = img.shape[1] / ppu, img.shape[0] / ppu
            else:
                w_m, h_m = r.m_Size.x, r.m_Size.y
                img = nine_slice(img, border, w_m * ppu, h_m * ppu, tiled=r.m_DrawMode == 2)
            c = r.m_Color
            img = img * np.array([c.r, c.g, c.b, c.a], np.float32)  # RGBA (PIL order); BGR only when written
            # local metres of the image corners: pivot at the origin; flips mirror about it
            x0, y0 = -px * w_m, -py * h_m
            sx, sy = (-1 if r.m_FlipX else 1), (-1 if r.m_FlipY else 1)
            # image pixel (u, v) (v down) -> local (x, y): x = sx*(x0 + u*w/iw), y = sy*(y0 + h - v*h/ih)
            iw, ih = img.shape[1], img.shape[0]
            L = np.array([[sx * w_m / iw, 0, sx * x0], [0, -sy * h_m / ih, sy * (y0 + h_m)], [0, 0, 1]])
            a, bb, cc, dd, tx, ty = W
            Wm = np.array([[a, cc, tx], [bb, dd, ty], [0, 0, 1]])
            key = (layer_rank.get(r.m_SortingLayerID, 0), r.m_SortingOrder, n)
            out.append((key, img, Wm @ L, moving_of[k], path_of[k], (W[4], W[5])))
    out.sort(key=lambda p: p[0])
    return out


def render(parts, bounds, ppm):
    """Composite the pieces over a transparent canvas covering bounds (x0, y0, x1, y1) at ppm pixels per metre."""
    x0, y0, x1, y1 = bounds
    Wc, Hc = int(math.ceil((x1 - x0) * ppm)), int(math.ceil((y1 - y0) * ppm))
    canvas = np.zeros((Hc, Wc, 4), np.float32)
    world_to_px = np.array([[ppm, 0, -x0 * ppm], [0, -ppm, y1 * ppm], [0, 0, 1]])
    for _, img, M, *_ in parts:
        A = world_to_px @ M  # image pixel -> canvas pixel
        ih, iw = img.shape[:2]
        corners = A @ np.array([[0, iw, 0, iw], [0, 0, ih, ih], [1, 1, 1, 1]])
        cx0, cy0 = np.floor(corners[:2].min(1)).astype(int)
        cx1, cy1 = np.ceil(corners[:2].max(1)).astype(int)
        cx0, cy0, cx1, cy1 = max(cx0, 0), max(cy0, 0), min(cx1, Wc), min(cy1, Hc)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        T = np.array([[1, 0, -cx0], [0, 1, -cy0], [0, 0, 1]]) @ A
        patch = cv2.warpAffine(img, T[:2], (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = patch[..., 3:4]
        dst = canvas[cy0:cy1, cx0:cx1]
        dst[..., :3] = patch[..., :3] * alpha + dst[..., :3] * (1 - alpha)
        dst[..., 3:4] = alpha + dst[..., 3:4] * (1 - alpha)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="Tower")
    ap.add_argument("--ppm", type=int, default=32, help="pixels per metre (the art is 16)")
    args = ap.parse_args()
    b = Build()
    parts = pieces(b, args.map)
    bounds = json.loads((OUT / f"{args.map}.json").read_text())["bounds"]
    still = render([p for p in parts if not p[3]], bounds, args.ppm)
    rgba = (np.clip(still, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{args.map}_art.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    info = {"map": args.map, "source": "tools/render_level.py", "ppm": args.ppm, "bounds": bounds,
            "left_out": sorted({p[4] for p in parts if p[3]}), "objects": []}
    # Things that move but whose look doesn't change, each on its own, to be drawn wherever they are (offset from
    # where the scene puts them): the lift cars, and the helicopter (it hovers; its spinning blades are left out).
    roots = sorted({p[4] for p in parts if p[4].endswith("/MapObj_Elevator") or p[4].endswith("/Helicopter_Body")})
    for i, root in enumerate(roots):
        own = [p for p in parts if (p[4] == root or p[4].startswith(root + "/")) and "Blade" not in p[4]]
        corners = np.hstack([p[2] @ np.array([[0, p[1].shape[1], 0, p[1].shape[1]], [0, 0, p[1].shape[0], p[1].shape[0]],
                                                 [1, 1, 1, 1]]) for p in own])
        box = [float(corners[0].min()), float(corners[1].min()), float(corners[0].max()), float(corners[1].max())]
        img = render(own, box, args.ppm)
        name = f"{args.map}_art_{i}.png"
        cv2.imwrite(str(OUT / name), cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA))
        info["objects"].append({"path": root, "png": name, "bounds": box,
                                "scene_pos": list(next(p[5] for p in own if p[4] == root))})
    (OUT / f"{args.map}_art.json").write_text(json.dumps(info, indent=1))
    print(f"{len(parts)} sprites, {sum(p[3] for p in parts)} moving ones left out -> {OUT / (args.map + '_art.png')} "
          f"{rgba.shape[1]}x{rgba.shape[0]}")


if __name__ == "__main__":
    main()
