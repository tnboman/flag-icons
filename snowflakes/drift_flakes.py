"""Four branching variations of snowflake 09 ("twelve point"), with a drift-and-lock animation.

Structure: the 09 rows, plus micro brand shapes that branch off the larger ones (forks at
their tips, spurs off their sides, and a second generation of forks on the forks). Every
shape stays distinct with a small gap, and every shape keeps the flake's mirror symmetry.

Animation (240 frames at 24 fps, held from frame 204):
    1-100    drift: shapes float like motes in low gravity, tumbling slowly, spread in depth
    56-200   kaleidoscope: each ring of shapes twists in alternating directions while it
             draws inward; rings sit on separate depth layers
    172-204  lock: the layers flatten onto one plane at the last moment
    204-240  hold on the finished flake

Usage:
    python3 snowflakes/drift_flakes.py [--stills] [--preview N] [--samples N] [--res PX]
        --stills       render the final locked frame of all four variations
        --preview N    render variation N's animation as a low-res MP4 preview

Outputs in snowflakes/drift/: drift_flakes.blend, stills, previews.
"""
import argparse
import math
import os
import random
import sys

import bpy
import shapely
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_snowflakes as base  # noqa: E402

OUT = os.path.join(base.HERE, "drift")
FPS = 24
FRAMES = 240
HOLD = 204


# ---------------------------------------------------------------- structure

BASE_09 = [base.row(0.8, 10, 0.26), base.row(0.56, 45, 0.24), base.row(0.34, 45, 0.18),
           base.row(0.62, 30, 0.2, 30)]


def direction(rot):
    return math.cos(math.radians(rot)), math.sin(math.radians(rot))


def clears(spec, accepted):
    """True if the spec's 12 copies clear each other and every accepted spec's copies."""
    prints = [base.footprint(ga, gt, base.BRAND_FIT) for ga, gt, _ in base.d6_orbit(spec)]
    if float(shapely.distance(prints[0], prints[1:]).min()) < base.GAP:
        return False
    for _, other in accepted:
        if float(shapely.distance(prints[0], other).min()) < base.GAP:
            return False
    return True


def orbit_prints(spec):
    return [base.footprint(ga, gt, base.BRAND_FIT) for ga, gt, _ in base.d6_orbit(spec)]


def place(spec, accepted, group, tries=8):
    """Add a micro shape if it fits, shrinking it a little at a time; skip it if it can't."""
    x, y, s, rot, m = spec
    for _ in range(tries):
        cand = (x, y, s, rot, m)
        if clears(cand, accepted):
            accepted.append((cand, orbit_prints(cand)))
            groups.append(group)
            return cand
        # step outward from the parent a little and shrink, keeping it close to where it grows
        ux, uy = direction(rot)
        x, y, s = x + 0.006 * ux, y + 0.006 * uy, s * 0.9
    return None


def fork(parent, ratio, angle):
    """Two micro shapes rooted just beyond the parent's tip, opening at +-angle."""
    x, y, s, rot, _ = parent
    ux, uy = direction(rot)
    # root sits right at the tip so the fork reads as growing out of its parent
    rx, ry = x + (0.5 * s - 0.01 * s) * ux, y + (0.5 * s - 0.01 * s) * uy
    sm = s * ratio
    out = []
    for a in (angle, -angle):
        dx, dy = direction(rot + a)
        out.append((rx + 0.5 * sm * dx, ry + 0.5 * sm * dy, sm, rot + a, False))
    return out


def spurs(parent, frac, ratio, angle):
    """Two micro shapes sprouting from the parent's sides at frac along its length."""
    x, y, s, rot, _ = parent
    ux, uy = direction(rot)
    d = (frac - 0.5) * s
    half = 0.35 * s * (1 - 2 * abs(d) / s) + base.GAP * 0.5
    sm = s * ratio
    out = []
    for side in (1, -1):
        nx, ny = -uy * side, ux * side
        rx, ry = x + d * ux + half * nx, y + d * uy + half * ny
        dx, dy = direction(rot + side * angle)
        out.append((rx + 0.55 * sm * dx, ry + 0.55 * sm * dy, sm, rot + side * angle, False))
    return out


groups = []


def build(name, tips=None, sides=None, second=None, dust=()):
    """tips/sides: micro-branch settings applied to every base shape. second: forks on the
    forks. dust: free micro shapes as (t, a, s, axis) V rows."""
    global groups
    groups = []
    accepted = []
    for g, spec in enumerate(BASE_09):
        accepted.append((spec, orbit_prints(spec)))
        groups.append(g)
    n_base = len(BASE_09)
    firsts = []
    for g, spec in enumerate(BASE_09):
        if tips:
            firsts += [(c, g) for c in [place(m, accepted, n_base + g) for m in fork(spec, *tips)] if c]
        if sides:
            for m in spurs(spec, *sides):
                place(m, accepted, n_base + g)
    if second:
        for spec, g in firsts:
            for m in fork(spec, *second):
                place(m, accepted, g + n_base)
    for r in dust:
        place(base.row(*r), accepted, 2 * n_base)
    return dict(name=name, rows=[spec for spec, _ in accepted], groups=list(groups))


VARIANTS = [
    build("A frost tips", tips=(0.42, 34)),
    build("B side spurs", tips=(0.34, 30), sides=(0.62, 0.32, 52)),
    build("C dendrite", tips=(0.46, 36), second=(0.5, 36)),
    build("D halo dust", tips=(0.4, 34), sides=(0.6, 0.26, 55),
          dust=[(0.16, 30, 0.07), (0.2, 30, 0.06, 30), (0.98, 0, 0.06)]),
]


def layout_groups(p):
    """Exact layout with each shape's group (which ring it twists with)."""
    shapes = base.layout(p)
    per = 12
    return [(ga, gt, s, p["groups"][i // per]) for i, (ga, gt, s) in enumerate(shapes)]


if __name__ == "__main__" and "--2d" in sys.argv:
    from PIL import Image, ImageDraw
    W = 520
    sheet = Image.new("RGB", (W * 4, W), "white")
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(VARIANTS):
        shapes = layout_groups(p)
        print(p["name"], "shapes", len(shapes))
        for ga, gt, s, g in shapes:
            poly = base.footprint(ga, gt)
            col = (0, 140, 220) if g < 4 else (90, 190, 240)
            d.polygon([(i * W + W / 2 + x * W * 0.42, W / 2 - y * W * 0.42) for x, y in poly.exterior.coords],
                      fill=col)
        d.text((i * W + 6, 6), p["name"], fill="black")
    sheet.save(sys.argv[-1])


# ---------------------------------------------------------------- motion

def smooth(f, a, b):
    """Smootherstep from 0 at frame a to 1 at frame b."""
    x = min(max((f - a) / (b - a), 0.0), 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def rot_z(deg):
    return Quaternion((0, 0, 1), math.radians(deg))


class Motion:
    """Drift, kaleidoscope twist and lock for one shape, all in the flake's own plane
    (x, y) with z as depth toward the camera."""

    def __init__(self, rng, final_xy, final_rot, group):
        self.fx, self.fy, self.frot, g = final_xy[0], final_xy[1], final_rot, group
        # start somewhere in a loose cloud; local x points down, so cap it to stay above the floor
        self.p0 = Vector((rng.uniform(-1.7, 1.0), rng.uniform(-1.8, 1.8), rng.uniform(-1.6, 1.3)))
        d = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
        self.v = d * rng.uniform(0.02, 0.06)
        self.wob = [(rng.uniform(0.02, 0.07), rng.uniform(0.15, 0.45), rng.uniform(0, 6.28)) for _ in range(3)]
        self.axis = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
        self.spin = rng.uniform(0.25, 0.7) * rng.choice((-1, 1))
        self.q0 = Quaternion((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
        # each ring twists the opposite way to its neighbour and sits on its own depth layer
        self.twist = (1 if g % 2 == 0 else -1) * (95 + 30 * (g % 4)) * (1.4 if g >= 4 else 1.0)
        self.layer = (((g * 0.618) % 1.0) - 0.5) * 1.2
        k = g % 5
        self.gather = (56 + 7 * k, 150 + 7 * k)

    def at(self, f):
        t = f / FPS
        drift = self.p0 + self.v * t + Vector(tuple(a * math.sin(w * 6.283 * t + ph) for a, w, ph in self.wob))
        tumble = Quaternion(self.axis, self.spin * t) @ self.q0

        w = smooth(f, *self.gather)        # cloud -> kaleidoscope
        e = smooth(f, 96, HOLD - 4)        # twist and spread unwind
        ez = smooth(f, 172, HOLD)          # depth layers flatten at the last moment
        wq = smooth(f, 80, 182)            # tumbling settles into the plane

        phi = self.twist * (1 - e) ** 1.2
        r = 1 + 0.45 * (1 - e)
        kal = rot_z(phi) @ Vector((self.fx * r, self.fy * r, 0))
        kal.z = self.layer * (1 - ez)
        pos = drift.lerp(kal, w)

        settle = rot_z(phi + self.frot)
        q = tumble.slerp(settle, wq) if wq < 1 else settle
        return pos, q


def add_curves(ob, frames, locs, quats, scale):
    from bpy_extras import anim_utils
    ob.rotation_mode = "QUATERNION"
    ob.keyframe_insert("location", frame=1)
    ad = ob.animation_data
    bag = anim_utils.action_ensure_channelbag_for_slot(ad.action, ad.action_slot)
    for fc in list(bag.fcurves):
        bag.fcurves.remove(fc)
    chans = [("location", i, [p[i] for p in locs]) for i in range(3)]
    chans += [("rotation_quaternion", i, [q[i] for q in quats]) for i in range(4)]
    for path, i, vals in chans:
        fc = bag.fcurves.new(path, index=i)
        fc.keyframe_points.add(len(frames))
        co = [c for pair in zip(frames, vals) for c in pair]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [0] * len(frames))  # constant: one key per frame
        fc.update()
    ob.scale = (scale, scale, scale)


def build_animated(idx, p, meshes, location, seed):
    rng = random.Random(seed)
    root = bpy.data.objects.new(f"Drift_{p['name'][0]}", None)
    bpy.context.scene.collection.objects.link(root)
    shapes = layout_groups(p)
    extent = max(max(abs(c) for c in base.footprint(ga, gt).bounds) for ga, gt, _, _ in shapes)
    frames = list(range(1, FRAMES + 1))
    for k, (ga, gt, s, g) in enumerate(shapes):
        det = ga[0][0] * ga[1][1] - ga[0][1] * ga[1][0]
        mirrored = det < 0
        # mirrored shapes use the mirrored mesh, so every transform is a plain rotation
        a00, a10 = (ga[0][0], ga[1][0])
        rot = math.degrees(math.atan2(a10, a00))
        ob = bpy.data.objects.new(f"Drift_{p['name'][0]}_shape_{k:03d}", meshes[mirrored])
        bpy.context.scene.collection.objects.link(ob)
        ob.parent = root
        mo = Motion(rng, gt, rot, g)
        locs, quats = [], []
        prev = None
        for f in frames:
            pos, q = mo.at(f)
            if prev is not None and prev.dot(q) < 0:
                q = -q
            prev = q
            locs.append(pos)
            quats.append(q)
        add_curves(ob, frames, locs, quats, s)
    root.location = location
    root.rotation_euler = (math.radians(90), math.radians(90), 0)
    sc = min(1.0, 1.02 / extent)
    root.scale = (sc, sc, sc)
    return root


def mirrored_copy(me):
    m2 = me.copy()
    m2.name = me.name + "_mirror"
    m2.transform(Matrix.Diagonal((1, -1, 1, 1)))
    m2.flip_normals()
    return m2


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--frames", default="")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--res", type=int, default=1200)
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    rig = base.setup_scene(args.samples, args.res)
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAMES
    mesh = base.brand_mesh(0.1, base.glass_material())
    meshes = (mesh, mirrored_copy(mesh))
    flakes = [build_animated(i, p, meshes, (i * 4.0, 0, 1.12), seed=900 + i)
              for i, p in enumerate(VARIANTS)]
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "drift_flakes.blend"))

    def solo(ob):
        for other in flakes:
            for child in other.children:
                child.hide_render = other is not ob
        rig.location = (ob.location.x, 0, 0)

    if args.stills:
        sc.frame_set(FRAMES)
        for i, ob in enumerate(flakes):
            solo(ob)
            sc.render.filepath = os.path.join(OUT, f"drift_{VARIANTS[i]['name'][0]}_final.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", sc.render.filepath, flush=True)

    if args.frames:
        # a few key moments of one variation: --preview N --frames 1,90,150,190,240
        ob = flakes[args.preview - 1]
        solo(ob)
        for f in [int(x) for x in args.frames.split(",")]:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(OUT, f"drift_{VARIANTS[args.preview - 1]['name'][0]}_f{f:03d}.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", sc.render.filepath, flush=True)
    elif args.preview:
        ob = flakes[args.preview - 1]
        solo(ob)
        tag = VARIANTS[args.preview - 1]["name"][0]
        sc.render.filepath = os.path.join(OUT, f"frames_{tag}", "f_")
        sc.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=True)
        print("rendered animation", tag, flush=True)


if __name__ == "__main__" and "--2d" not in sys.argv:
    main()
