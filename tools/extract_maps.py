"""Extract the real maps from the original WebGL build into JSON (world-space collider geometry).

    py tools/extract_maps.py            -> sim/maps/<Map>.json
"""
import json
import math
import struct
from pathlib import Path

from unitydump import Build

OUT = Path(__file__).resolve().parents[1] / "sim" / "maps"
MAPS = ["Map_Street", "Map_Train", "Map_Town", "Map_Tower"]


def xf(W, x, y):
    a, b, c, d, tx, ty = W
    return [a * x + c * y + tx, b * x + d * y + ty]


def vec(v):
    return [float(v.x), float(v.y)]


def material(pptr):
    if not pptr.m_PathID:
        return None
    m = pptr.read()  # resolves materials stored in other asset files too
    return {"name": m.m_Name, "friction": float(m.friction), "bounciness": float(m.bounciness)}


def parse_elevator(raw: bytes, world_pos):
    n = struct.unpack_from("<i", raw, 0)[0]
    pts = [world_pos.get(struct.unpack_from("<q", raw, 4 + 12 * i + 4)[0]) for i in range(n)]
    off = 4 + 12 * n
    going_up, speed, wait, dark = struct.unpack_from("<ifff", raw, off)
    return {"points": pts, "going_up": bool(going_up), "speed": speed, "wait": wait, "start_darkness": dark}


def extract(b: Build, name: str) -> dict:
    root = b.roots(name, "sharedassets1.assets")[0]
    shapes, bodies, joints, scripts, animators, effectors = [], [], [], [], [], []
    parent_body = {}
    path_of = {}
    world_pos = {k[1]: [W[4], W[5]] for k, _, W in b.walk(root)}
    # The route order of the waypoints is their order in the scene (WaypointManager uses
    # GetComponentsInChildren), not the numbers in their names.
    wp_order = {}
    for k, _, _ in b.walk(root):
        if b.go[(k[0], b.tr[k].m_GameObject.m_PathID)].m_Name == "Waypoints":
            wp_order = {c.m_PathID: i for i, c in enumerate(b.tr[k].m_Children)}
    for k, depth, W in b.walk(root):
        d = b.tr[k]
        gk = (k[0], d.m_GameObject.m_PathID)
        g = b.go[gk]
        fk = (k[0], d.m_Father.m_PathID) if d.m_Father.m_PathID else None
        path = (path_of[fk] + "/" if fk in path_of else "") + g.m_Name
        path_of[k] = path
        active = bool(g.m_IsActive) and (fk is None or path_of.get(("active",) + fk, True))
        path_of[("active",) + k] = active
        inherited = parent_body.get(fk)
        comps = list(b.components(gk))
        ang = math.degrees(math.atan2(W[1], W[0]))
        node = {"go": g.m_Name, "path": path, "tag": b.tag_name(g.m_Tag), "layer": g.m_Layer, "active": active,
                "pos": [W[4], W[5]], "angle": ang, "scale": [math.hypot(W[0], W[1]), math.hypot(W[2], W[3])]}
        for ck, t in comps:
            if t == "Rigidbody2D":
                r = b.obj[ck].read()
                inherited = len(bodies)
                bodies.append({**node, "id": inherited, "rb_path_id": ck[1], "type": ["dynamic", "kinematic", "static"][r.m_BodyType],
                               "mass": r.m_Mass, "gravity": r.m_GravityScale, "drag": r.m_LinearDrag,
                               "angular_drag": r.m_AngularDrag, "constraints": r.m_Constraints,
                               "material": material(r.m_Material)})
        parent_body[k] = inherited
        for ck, t in comps:
            if t in ("BoxCollider2D", "PolygonCollider2D", "CircleCollider2D"):
                c = b.obj[ck].read()
                base = {**node, "body": inherited, "trigger": bool(c.m_IsTrigger), "enabled": bool(c.m_Enabled),
                        "effector": bool(c.m_UsedByEffector), "material": material(c.m_Material)}
                ox, oy = vec(c.m_Offset)
                if t == "BoxCollider2D":
                    sx, sy = vec(c.m_Size)
                    pts = [xf(W, ox + dx * sx / 2, oy + dy * sy / 2) for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                    shapes.append({**base, "kind": "box", "points": pts, "local": {"offset": [ox, oy], "size": [sx, sy]}})
                elif t == "PolygonCollider2D":
                    for p in c.m_Points.m_Paths:
                        shapes.append({**base, "kind": "poly", "points": [xf(W, ox + q.x, oy + q.y) for q in p]})
                else:
                    shapes.append({**base, "kind": "circle", "center": xf(W, ox, oy),
                                   "radius": float(c.m_Radius) * max(node["scale"])})
            elif t == "WheelJoint2D":
                j = b.obj[ck].read()
                joints.append({**node, "body": inherited, "type": "wheel", "anchor": vec(j.m_Anchor),
                               "connected_anchor": vec(j.m_ConnectedAnchor), "connected": j.m_ConnectedRigidBody.m_PathID,
                               "use_motor": bool(j.m_UseMotor), "motor_speed": j.m_Motor.m_MotorSpeed,
                               "motor_force": j.m_Motor.m_MaximumMotorForce,
                               "suspension": {"damping": j.m_Suspension.m_DampingRatio,
                                              "frequency": j.m_Suspension.m_Frequency,
                                              "angle": j.m_Suspension.m_Angle}})
            elif t in ("SurfaceEffector2D", "AreaEffector2D"):
                e = b.obj[ck].read()
                keep = {a: getattr(e, a) for a in dir(e) if a.startswith("m_") and isinstance(getattr(e, a), (int, float, bool))}
                effectors.append({**node, "type": t, **{k2[2:]: v for k2, v in keep.items()}})
            elif t == "MonoBehaviour":
                sname = b.script_name(ck)
                if sname in ("Paralax",):
                    continue
                raw = b.mb_fields_raw(ck)
                entry = {**node, "script": sname, "raw": raw.hex()}
                if sname == "Waypoint":
                    ints = struct.unpack_from("<7i", raw, 0)
                    entry["spawnable"] = bool(raw[0])
                    entry["order"] = wp_order[k[1]]
                    # horizontal side that counts as passed: +1 right of the line, -1 left, 0 either
                    entry["side_x"] = 1 if ints[2] == 256 else (-1 if ints[2] == 1 else 0)
                    entry["above"] = bool(ints[4])  # also has to be above the waypoint's height
                    entry["y_offset"] = struct.unpack_from("<f", raw, 24)[0]
                elif sname == "Elevator":
                    entry["elevator"] = parse_elevator(raw, world_pos)
                elif sname == "Zipline":
                    ends = {b.go[(c[0], b.tr[c].m_GameObject.m_PathID)].m_Name: world_pos[c[1]] for c in b.children[k]}
                    entry["zipline"] = {"top": ends["P1"], "bottom": ends["P2"],
                                        "catch_zone": struct.unpack_from("<f", raw, 0)[0]}
                scripts.append(entry)
            elif t == "Animator":
                a = b.obj[ck].read()
                ctrl = a.m_Controller.read().m_Name if a.m_Controller.m_PathID else None
                animators.append({**node, "controller": ctrl, "enabled": bool(a.m_Enabled)})
    spawn = next((s["pos"] for s in scripts if s["script"] == "StartPlayers"), None)
    xs = [p[0] for s in shapes if s["kind"] != "circle" for p in s["points"]]
    ys = [p[1] for s in shapes if s["kind"] != "circle" for p in s["points"]]
    return {"name": name.replace("Map_", ""), "source": "GetawayShootoutPoki (2019, Unity 2019.3.10f1)",
            "bounds": [min(xs), min(ys), max(xs), max(ys)], "spawn": spawn,
            "shapes": shapes, "bodies": bodies, "joints": joints, "effectors": effectors,
            "scripts": scripts, "animators": animators}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = Build()
    for name in MAPS:
        m = extract(b, name)
        (OUT / f"{m['name']}.json").write_text(json.dumps(m, indent=1, default=float))
        print(f"{m['name']:<8} shapes={len(m['shapes'])} bodies={len(m['bodies'])} joints={len(m['joints'])} "
              f"effectors={len(m['effectors'])} scripts={len(m['scripts'])} animators={len(m['animators'])} spawn={m['spawn']}")


if __name__ == "__main__":
    main()
