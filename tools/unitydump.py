"""Shared helpers for reading the original Getaway Shootout WebGL build with UnityPy."""
import collections
import struct
from pathlib import Path

import UnityPy

DATA = Path(__file__).resolve().parent / "GetawayShootoutPoki.data.unityweb"  # original 2019 WebGL build data


class Build:
    def __init__(self, path=DATA):
        self.env = UnityPy.load(str(path))
        self.obj = {}
        self.typ = {}
        for o in self.env.objects:
            k = (o.assets_file.name, o.path_id)
            self.obj[k] = o
            self.typ[k] = o.type.name
        self.go = {}
        self.tr = {}  # transform key -> data
        self.children = collections.defaultdict(list)
        for k, o in self.obj.items():
            if self.typ[k] == "GameObject":
                self.go[k] = o.read()
            elif self.typ[k] in ("Transform", "RectTransform"):
                d = o.read()
                self.tr[k] = d
                if d.m_Father.m_PathID:
                    self.children[(k[0], d.m_Father.m_PathID)].append(k)
        tm = next(o for o in self.env.objects if o.type.name == "TagManager").read()
        self.tags = ["Untagged", "Respawn", "Finish", "EditorOnly", "MainCamera", "Player", "GameController"]
        self.custom_tags = list(tm.tags)

    def tag_name(self, idx):
        if idx < 20000:
            return self.tags[idx] if idx < len(self.tags) else str(idx)
        i = idx - 20000
        return self.custom_tags[i] if i < len(self.custom_tags) else str(idx)

    def roots(self, name, file=None):
        out = []
        for k, d in self.tr.items():
            if d.m_Father.m_PathID == 0:
                g = self.go[(k[0], d.m_GameObject.m_PathID)]
                if g.m_Name == name and (file is None or k[0] == file):
                    out.append(k)
        return out

    def components(self, go_key):
        f = go_key[0]
        for c in self.go[go_key].m_Component:
            if c.component.m_FileID == 0:
                k = (f, c.component.m_PathID)
                yield k, self.typ.get(k)

    def script_name(self, mb_key):
        o = self.obj[mb_key]
        raw = o.get_raw_data()
        # MonoBehaviour header: m_GameObject PPtr(12) m_Enabled(1+3 align) m_Script PPtr(12) m_Name(str)
        fid, pid = struct.unpack_from("<iq", raw, 16)
        f = mb_key[0]
        if fid:
            f = o.assets_file.externals[fid - 1].path.split("/")[-1]
        sk = (f, pid)
        if sk in self.obj:
            return self.obj[sk].read().m_ClassName
        # fall back: search by path id in any file
        for (ff, pp), t in self.typ.items():
            if pp == pid and t == "MonoScript":
                return self.obj[(ff, pp)].read().m_ClassName
        return "?"

    def mb_fields_raw(self, mb_key):
        """Raw serialized bytes after the MonoBehaviour header (fields of the script)."""
        raw = self.obj[mb_key].get_raw_data()
        n = struct.unpack_from("<i", raw, 28)[0]
        off = 32 + n
        off += (-off) % 4
        return raw[off:]

    def walk(self, root_tr):
        """Yield (transform_key, depth, world_matrix) for every node under root (2D affine)."""
        import math
        stack = [(root_tr, 0, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))]
        while stack:
            k, depth, M = stack.pop()
            d = self.tr[k]
            p, q, s = d.m_LocalPosition, d.m_LocalRotation, d.m_LocalScale
            ang = 2 * math.atan2(q.z, q.w)
            c, sn = math.cos(ang), math.sin(ang)
            # local = T * R * S
            L = (c * s.x, sn * s.x, -sn * s.y, c * s.y, p.x, p.y)
            a, b, cc, dd, tx, ty = M
            W = (a * L[0] + cc * L[1], b * L[0] + dd * L[1], a * L[2] + cc * L[3], b * L[2] + dd * L[3],
                 a * L[4] + cc * L[5] + tx, b * L[4] + dd * L[5] + ty)
            yield k, depth, W
            for ch in reversed(self.children[k]):
                stack.append((ch, depth + 1, W))
