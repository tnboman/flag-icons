"""Chandelier snowflakes from the Access brand shape, with a dancing assembly animation.

Structure: a scale hierarchy. Large facets (an arrowhead at each tip, wings mid-arm) carry
the snowflake silhouette, medium chevrons join them and form the core, and fine prongs
appear only at the wing tips and arrowheads. Every shape is distinct, and the flake keeps
its mirror symmetry.

Motion (240 frames at 24 fps): shapes begin as a slow swirl of glints, each orbiting the
centre on its own spiral while pirouetting and tilting, with depth that breathes like a
wave. Rings by radius counter-rotate like a kaleidoscope, then lock from the centre
outward, each gliding to a stop. The finished flake holds from frame 212.

Usage:
    python3 snowflakes/dance_flakes.py --2d OUT.png
    python3 snowflakes/dance_flakes.py [--stills] [--preview N] [--frames 1,90,...]
                                       [--samples N] [--res PX] [--genomes S07,S12]
Outputs in snowflakes/dance/.
"""
import argparse
import math
import os
import random
import sys

import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_snowflakes as base  # noqa: E402

OUT = os.path.join(base.HERE, "dance")
FPS = 24
FRAMES = 240
HOLD = 212
CHAIN_GAP = base.GAP * 1.4


# ---------------------------------------------------------------- structure

def direction(deg):
    return math.cos(math.radians(deg)), math.sin(math.radians(deg))


def orbit_prints(spec):
    return [base.footprint(ga, gt, base.BRAND_FIT) for ga, gt, _ in base.d6_orbit(spec)]


class Flake:
    """Accumulates shape specs, keeping every shape (and its 12 copies) clear of the rest."""

    def __init__(self, name):
        self.name = name
        self.specs = []
        self.prints = []

    def fits(self, spec):
        prints = orbit_prints(spec)
        if float(shapely.distance(prints[0], prints[1:]).min()) < base.GAP:
            return None
        for other in self.prints:
            if float(shapely.distance(prints[0], other).min()) < base.GAP:
                return None
        return prints

    def add(self, spec, shrink=True):
        """Add a shape; if it touches something, shrink it toward its root a few times."""
        x, y, s, rot, m = spec
        ux, uy = direction(rot)
        for _ in range(6 if shrink else 1):
            cand = (x, y, s, rot, m)
            prints = self.fits(cand)
            if prints is not None:
                self.specs.append(cand)
                self.prints.append(prints)
                return cand
            x, y, s = x - 0.05 * s * ux, y - 0.05 * s * uy, s * 0.9
        return None

    def chain(self, root, angle, sizes):
        """A strand of shapes laid end to end along angle from root, graduated by sizes."""
        ux, uy = direction(angle)
        acc, out = 0.0, []
        # step the root away from the spine until the first crystal clears it
        s0 = sizes[0]
        for _ in range(20):
            if self.fits((root[0] + s0 / 2 * ux, root[1] + s0 / 2 * uy, s0, angle, False)):
                break
            root = (root[0], root[1] + 0.006)
        for s in sizes:
            spec = self.add((root[0] + (acc + s / 2) * ux, root[1] + (acc + s / 2) * uy, s, angle, False))
            if spec is None:
                break
            out.append(spec)
            acc += s + CHAIN_GAP
        return out

    def chevrons(self, positions, sizes, splay):
        """The arm's spine: spaced chevrons (a shape and its mirror) along the arm axis.
        Returns the outermost shape so a crown can grow from it."""
        out = None
        for t, sz, a in zip(positions, sizes, splay):
            spec = base.row(t, a, sz)
            if self.add(spec, shrink=False):
                out = spec
        return out

    def crown(self, parent, ratio, angle, centre=False):
        """Accent at a strand's tip: two prongs at +-angle, optionally a third straight on."""
        x, y, s, rot, _ = parent
        ux, uy = direction(rot)
        rx, ry = x + (0.5 * s + 2.2 * base.GAP) * ux, y + (0.5 * s + 2.2 * base.GAP) * uy
        sm = s * ratio
        prongs = [angle, -angle] + ([0] if centre else [])
        for a in prongs:
            dx, dy = direction(rot + a)
            self.add((rx + 0.5 * sm * dx, ry + 0.5 * sm * dy, sm, rot + a, False))

    def done(self):
        return dict(name=self.name, rows=self.specs)


def graduated(n, a, b):
    return [a + (b - a) * k / max(n - 1, 1) for k in range(n)]


def crystal():
    """Big arrowhead at each tip and large wings mid-arm carry the silhouette; a medium
    chevron and core ring fill it; fine prongs only at the wing tips and the arrowhead."""
    f = Flake("G crystal")
    arrow = f.add(base.row(0.76, 12, 0.28), shrink=False)
    wing = f.add(base.row(0.44, 50, 0.24), shrink=False)
    f.add(base.row(0.63, 26, 0.13), shrink=False)
    f.add(base.row(0.27, 34, 0.13), shrink=False)
    f.add(base.row(0.13, 30, 0.1, 30), shrink=False)
    f.add(base.row(0.45, 16, 0.11, 30), shrink=False)
    f.crown(wing, 0.36, 34)
    f.crown(arrow, 0.3, 30)
    return f.done()


def frond():
    """A feather of facets growing outward to a bold tip, a medium core ring, and fine
    sparks on the two largest wings only."""
    f = Flake("H frond")
    f.add(base.row(0.13, 30, 0.11, 30), shrink=False)
    f.add(base.row(0.22, 40, 0.1), shrink=False)
    f.add(base.row(0.35, 46, 0.15), shrink=False)
    big = f.add(base.row(0.5, 50, 0.21), shrink=False)
    f.add(base.row(0.66, 24, 0.12), shrink=False)
    tip = f.add(base.row(0.79, 14, 0.25), shrink=False)
    f.crown(big, 0.32, 36)
    f.crown(tip, 0.28, 34)
    f.add(base.row(0.5, 14, 0.09, 30), shrink=False)
    return f.done()


VARIANTS = [crystal(), frond()]


def preview_2d(path):
    from PIL import Image, ImageDraw
    W = 640
    sheet = Image.new("RGB", (W * len(VARIANTS), W), "white")
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(VARIANTS):
        shapes = base.layout(p)
        print(p["name"], "shapes", len(shapes))
        for ga, gt, s in shapes:
            poly = base.footprint(ga, gt)
            d.polygon([(i * W + W / 2 + x * W * 0.45, W / 2 - y * W * 0.45) for x, y in poly.exterior.coords],
                      fill=(0, 140, 220))
        d.text((i * W + 6, 6), p["name"], fill="black")
    sheet.save(path)




# ---------------------------------------------------------------- motion

def smooth(f, a, b):
    """Smootherstep from 0 at frame a to 1 at frame b."""
    x = min(max((f - a) / (b - a), 0.0), 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


class Dance:
    """One shape's choreography in the flake's plane (x, y), with z as depth toward the camera.

    Free: it orbits the centre on its own radius, pirouetting and tilting, bobbing in depth.
    Formation: it joins its ring, which counter-rotates against its neighbours and sits on
    its own depth layer. Lock: rings glide to a stop from the centre outward."""

    def __init__(self, rng, final_xy, final_rot):
        from mathutils import Vector
        fx, fy = final_xy
        self.rf, self.thf, self.rot = math.hypot(fx, fy), math.atan2(fy, fx), final_rot
        b = min(int(self.rf / 0.18), 5)
        self.sign = 1 if b % 2 == 0 else -1
        self.lock = 150 + 11 * b                       # inner rings lock first
        self.gather = (24 + 10 * b, self.lock - 26)
        self.layer = self.sign * (0.16 + 0.07 * b)
        self.r0 = rng.uniform(0.45, 2.1)
        d0 = rng.uniform(-math.pi, math.pi) - self.thf
        self.d0 = math.atan2(math.sin(d0), math.cos(d0))
        self.omega = self.sign * rng.uniform(0.25, 0.45) / math.sqrt(self.r0)
        self.z0 = rng.uniform(-1.0, 0.7)
        self.ph = rng.uniform(0, 6.283)
        self.bob = rng.uniform(0.04, 0.09)
        self.twist = self.sign * math.radians(rng.uniform(120, 170))
        self.spin = rng.uniform(0.5, 1.1) * rng.choice((-1, 1))
        self.tilt_axis = Vector((rng.gauss(0, 1), rng.gauss(0, 1), 0)).normalized()
        self.tilt = math.radians(rng.uniform(35, 75))

    def at(self, f):
        from mathutils import Quaternion, Vector
        t = f / FPS
        w = smooth(f, *self.gather)                      # free orbit -> ring
        e = smooth(f, 60, self.lock)                     # ring twist unwinds
        ez = smooth(f, self.lock - 24, self.lock)        # depth layer flattens
        wq = smooth(f, self.lock - 80, self.lock - 4)    # pirouette settles

        free_th = self.d0 + self.omega * t
        # long ease-out: the ring glides in and stops, no overshoot
        ring_th = self.twist * (1 - e) ** 2
        th = self.thf + free_th + (ring_th - free_th) * w

        free_r = self.r0 * (1 + 0.06 * math.sin(0.9 * t + self.ph))
        ring_r = self.rf * (1 + 0.28 * (1 - e) ** 1.5)
        r = free_r + (ring_r - free_r) * w

        wave = 0.14 * math.sin(2 * math.pi * 0.22 * t - 4 * self.rf + self.ph * 0.2)
        free_z = self.z0 + self.bob * math.sin(1.3 * t + self.ph) + wave
        ring_z = (self.layer + wave) * (1 - ez)
        z = free_z + (ring_z - free_z) * w

        pos = Vector((r * math.cos(th), r * math.sin(th), z))
        # local +x points down: ease free-floating shapes to a stop above the floor
        if pos.x > 0.75:
            pos.x = 0.75 + 0.22 * math.tanh((pos.x - 0.75) / 0.22)

        settle = Quaternion((0, 0, 1), math.radians(self.rot) + ring_th)
        twirl = Quaternion((0, 0, 1), math.radians(self.rot) + self.spin * t) @ \
            Quaternion(self.tilt_axis, self.tilt * math.sin(0.7 * t + self.ph))
        q = twirl.slerp(settle, wq) if wq < 1 else settle
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
        fc.keyframe_points.foreach_set("co", [c for pair in zip(frames, vals) for c in pair])
        fc.keyframe_points.foreach_set("interpolation", [0] * len(frames))  # one key per frame
        fc.update()
    ob.scale = (scale, scale, scale)


def mirrored_copy(me):
    from mathutils import Matrix
    m2 = me.copy()
    m2.name = me.name + "_mirror"
    m2.transform(Matrix.Diagonal((1, -1, 1, 1)))
    m2.flip_normals()
    return m2


def build_animated(p, meshes, location, seed):
    import bpy
    rng = random.Random(seed)
    tag = p["name"].split()[0]
    root = bpy.data.objects.new(f"Dance_{tag}", None)
    bpy.context.scene.collection.objects.link(root)
    shapes = base.layout(p)
    extent = max(max(abs(c) for c in base.footprint(ga, gt).bounds) for ga, gt, _ in shapes)
    frames = list(range(1, FRAMES + 1))
    for k, (ga, gt, s) in enumerate(shapes):
        mirrored = ga[0][0] * ga[1][1] - ga[0][1] * ga[1][0] < 0
        # mirrored shapes use the mirrored mesh, so every transform is a plain rotation
        rot = math.degrees(math.atan2(ga[1][0], ga[0][0]))
        ob = bpy.data.objects.new(f"Dance_{tag}_shape_{k:03d}", meshes[mirrored])
        bpy.context.scene.collection.objects.link(ob)
        ob.parent = root
        d = Dance(rng, gt, rot)
        locs, quats, prev = [], [], None
        for f in frames:
            pos, q = d.at(f)
            if prev is not None and prev.dot(q) < 0:
                q = -q
            prev = q
            locs.append(pos)
            quats.append(q)
        add_curves(ob, frames, locs, quats, s)
    root.location = location
    root.rotation_euler = (math.radians(90), math.radians(90), 0)
    sc = 1.02 / extent
    root.scale = (sc, sc, sc)
    return root


def main():
    import bpy
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--frames", default="")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--res", type=int, default=1200)
    ap.add_argument("--genomes", default="", help="evolved designs to use instead, e.g. S07,S12")
    args = ap.parse_args(argv)
    global VARIANTS
    if args.genomes:
        import json
        from evolve_flakes import express
        with open(os.path.join(base.HERE, "evolve", "genomes.json")) as fh:
            pool = json.load(fh)
        VARIANTS = [express(pool[k]["tiers"], f"{k} evolved")[0] for k in args.genomes.split(",")]

    os.makedirs(OUT, exist_ok=True)
    rig = base.setup_scene(args.samples, args.res)
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAMES
    mesh = base.brand_mesh(0.1, base.glass_material())
    meshes = (mesh, mirrored_copy(mesh))
    flakes = [build_animated(p, meshes, (i * 4.0, 0, 1.12), seed=2024 + i) for i, p in enumerate(VARIANTS)]
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "dance_flakes.blend"))

    def solo(ob):
        for other in flakes:
            for child in other.children:
                child.hide_render = other is not ob
        rig.location = (ob.location.x, 0, 0)

    if args.stills:
        sc.frame_set(FRAMES)
        for p, ob in zip(VARIANTS, flakes):
            solo(ob)
            sc.render.filepath = os.path.join(OUT, f"dance_{p['name'].split()[0]}_final.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", sc.render.filepath, flush=True)

    if args.preview:
        p, ob = VARIANTS[args.preview - 1], flakes[args.preview - 1]
        solo(ob)
        tag = p["name"].split()[0]
        if args.frames:
            for f in [int(x) for x in args.frames.split(",")]:
                sc.frame_set(f)
                sc.render.filepath = os.path.join(OUT, f"dance_{tag}_f{f:03d}.png")
                bpy.ops.render.render(write_still=True)
                print("rendered", sc.render.filepath, flush=True)
        else:
            sc.render.filepath = os.path.join(OUT, f"frames_{tag}", "f_")
            bpy.ops.render.render(animation=True)
            print("rendered animation", tag, flush=True)


if __name__ == "__main__":
    if "--2d" in sys.argv:
        preview_2d(sys.argv[-1])
    else:
        main()
