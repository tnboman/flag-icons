"""Build 12 glass snowflake variations in Blender from the Access brand shape.

Every brand shape stays a distinct glass object (linked copies of one mesh), never merged.
Shapes vary in scale and angle and always appear with their mirror image across each arm,
so the flakes have a crystal's mirror symmetry instead of a rotating pinwheel.

Usage:
    python3 snowflakes/build_snowflakes.py [--samples N] [--res PX] [--only i,j]

Outputs (next to this script):
    snowflakes.blend            all 12 variations laid out in a 4x3 grid
    renders/snowflake_XX.png    one render per variation
    renders/contact_sheet.png   all renders in one image
"""
import argparse
import math
import os
import re
import sys

import bpy
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(HERE, "renders")
ICE_IOR = 1.31


# ---------------------------------------------------------------- 2D shape

# Access brand shape (AccessShape_AccessBlue), path copied verbatim from the brand SVG.
BRAND_PATH = (
    "M1264.07,0H341.4c-24.054,0-40.6,10.662-49.314,33.647L3.82,793.821"
    "c-11.056,29.161,2.402,48.069,33.672,48.069h922.712c24.054,0,40.579-10.604,49.318-33.65"
    "L1297.743,48.073c11.192-29.289-2.402-48.073-33.673-48.073"
)
ACCESS_BLUE = (0.0, 0.51, 0.89)  # #00bdf2 in linear RGB


def parse_svg_path(d, steps=12):
    """Supports the commands used by the brand path: M H h L l C c (absolute/relative)."""
    tokens = re.findall(r"[MHhLlCcVvZz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    pts, i, cmd = [], 0, None
    x = y = 0.0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                continue
        n = {"M": 2, "L": 2, "l": 2, "H": 1, "h": 1, "V": 1, "v": 1, "C": 6, "c": 6}[cmd]
        v = [float(t) for t in tokens[i:i + n]]
        i += n
        if cmd in "ML":
            x, y = v
        elif cmd == "l":
            x, y = x + v[0], y + v[1]
        elif cmd == "H":
            x = v[0]
        elif cmd == "h":
            x += v[0]
        elif cmd == "V":
            y = v[0]
        elif cmd == "v":
            y += v[0]
        else:
            if cmd == "c":
                v = [v[k] + (x if k % 2 == 0 else y) for k in range(6)]
            x0, y0 = x, y
            for k in range(1, steps + 1):
                t = k / steps
                mt = 1 - t
                pts.append((mt ** 3 * x0 + 3 * mt * mt * t * v[0] + 3 * mt * t * t * v[2] + t ** 3 * v[4],
                            mt ** 3 * y0 + 3 * mt * mt * t * v[1] + 3 * mt * t * t * v[3] + t ** 3 * v[5]))
            x, y = v[4], v[5]
            continue
        pts.append((x, y))
    return pts


def brand_unit():
    """Brand shape centred on the origin, 1 unit wide, long axis along +x, y flipped to y-up."""
    poly = Polygon([(px, -py) for px, py in parse_svg_path(BRAND_PATH)]).buffer(0)
    minx, miny, maxx, maxy = poly.bounds
    poly = affinity.translate(poly, -(minx + maxx) / 2, -(miny + maxy) / 2)
    return affinity.scale(poly, 1 / (maxx - minx), 1 / (maxx - minx), origin=(0, 0))


BRAND = brand_unit()
BRAND_FIT = BRAND.simplify(0.004).buffer(0.002)


GAP = 0.012  # minimum clear space between any two shapes


def piece_matrix(x, y, s, rot=0.0, mirror=False):
    """2x2 linear part and offset for one brand shape: scale s, optional mirror, rotate, place."""
    c, n = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    m = -1 if mirror else 1
    return [[c * s, -n * s * m], [n * s, c * s * m]], (x, y)


def d6_orbit(spec):
    """Place one shape spec in all 12 positions of the snowflake's mirror symmetry.

    The 6 rotations alone would make a pinwheel (the brand shape is slanted, so it spins).
    Adding the mirror image across every arm axis cancels the spin: each shape always
    appears together with its reflection, giving the crystal's mirrored architecture.
    """
    (a, t) = piece_matrix(*spec)
    out = []
    for k in range(6):
        c, n = math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))
        for f in (1, -1):
            g = [[c, -n * f], [n, c * f]]
            ga = [[g[i][0] * a[0][j] + g[i][1] * a[1][j] for j in range(2)] for i in range(2)]
            gt = (g[0][0] * t[0] + g[0][1] * t[1], g[1][0] * t[0] + g[1][1] * t[1])
            out.append((ga, gt, spec[2]))
    return out


def footprint(ga, gt, shape=None):
    shape = BRAND if shape is None else shape
    return affinity.affine_transform(shape, [ga[0][0], ga[0][1], ga[1][0], ga[1][1], gt[0], gt[1]])


def orbit_prints(spec):
    # fitting only: a slightly coarser outline keeps the search fast, layout() checks the exact one
    return np.array([footprint(ga, gt, BRAND_FIT) for ga, gt, _ in d6_orbit(spec)])


def clearance(a, b):
    return float(shapely.distance(a[:, None], b[None, :]).min())


def fit_specs(specs):
    """Keep every shape distinct: nudge each spec off the arm axis until it clears its own
    mirror images, then shrink a spec that still touches an earlier one."""
    fitted = []
    for spec in specs:
        x, y, s, rot, *mirror = spec
        mirror = bool(mirror and mirror[0])
        s0 = s
        while True:
            if s < 0.3 * s0:
                raise ValueError(f"spec {spec} cannot be fitted without touching another shape")
            cand = (x, y, s, rot, mirror)
            prints = orbit_prints(cand)
            if float(shapely.distance(prints[0], prints[1:]).min()) < GAP:
                # move off the arm axis while that helps; near the 30 degree line, shrink instead
                if math.degrees(math.atan2(y + 0.004, x)) < 24:
                    y += 0.004
                else:
                    s *= 0.97
                continue
            if any(clearance(prints, orbit_prints(f)) < GAP for f in fitted):
                s *= 0.97
                continue
            break
        fitted.append(cand)
    return fitted


def layout(p):
    """All shapes of one snowflake as (matrix, offset, scale). Fails if any two touch."""
    shapes = [o for spec in fit_specs(p["specs"]) for o in d6_orbit(spec)]
    prints = [footprint(ga, gt) for ga, gt, _ in shapes]
    tree = shapely.STRtree(prints)
    for i, g in enumerate(prints):
        for j in tree.query(g.buffer(GAP)):
            if j > i and g.distance(prints[j]) < GAP:
                raise ValueError(f"shapes {i} and {j} touch in {p['name']}")
    return shapes, prints


# ---------------------------------------------------------------- mesh

def prism_mesh(name, poly, z0, z1):
    verts, faces, index = [], [], {}

    def vid(x, y, z):
        key = (round(x, 6), round(y, 6), z)
        if key not in index:
            index[key] = len(verts)
            verts.append((x, y, z))
        return index[key]

    poly = orient(poly.simplify(0.0005), 1.0)
    for tri in shapely.constrained_delaunay_triangles(poly).geoms:
        (ax, ay), (bx, by), (cx, cy) = list(tri.exterior.coords)[:3]
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
            (bx, by), (cx, cy) = (cx, cy), (bx, by)
        faces.append((vid(ax, ay, z1), vid(bx, by, z1), vid(cx, cy, z1)))
        faces.append((vid(ax, ay, z0), vid(cx, cy, z0), vid(bx, by, z0)))
    c = list(poly.exterior.coords)[:-1]
    for i in range(len(c)):
        (ax, ay), (bx, by) = c[i], c[(i + 1) % len(c)]
        faces.append((vid(ax, ay, z0), vid(bx, by, z0), vid(bx, by, z1), vid(ax, ay, z1)))

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    for f in me.polygons:
        f.use_smooth = True
    me.update()
    return me


def brand_mesh(thick):
    """One unit brand-shape glass slab; every shape in every snowflake is a linked copy of it."""
    me = prism_mesh(f"AccessShape_t{thick}", BRAND, -thick / 2, thick / 2)
    return me


def build_snowflake(idx, p, material, location):
    from mathutils import Matrix

    shapes, _ = layout(p)
    root = bpy.data.objects.new(f"Snowflake_{idx:02d}", None)
    root.empty_display_size = 0.3
    bpy.context.scene.collection.objects.link(root)
    me = brand_mesh(p.get("thick", 0.16))
    me.materials.append(material)
    for k, (ga, gt, s) in enumerate(shapes):
        ob = bpy.data.objects.new(f"Snowflake_{idx:02d}_shape_{k:02d}", me)
        bpy.context.scene.collection.objects.link(ob)
        # thickness scales with the shape so small shapes stay in proportion
        ob.matrix_basis = Matrix(((ga[0][0], ga[0][1], 0, gt[0]),
                                  (ga[1][0], ga[1][1], 0, gt[1]),
                                  (0, 0, s, 0),
                                  (0, 0, 0, 1)))
        ob.parent = root
        bev = ob.modifiers.new("bevel", "BEVEL")
        bev.width = p.get("bevel", 0.03)
        bev.segments = 4
        bev.limit_method = "ANGLE"
        bev.angle_limit = math.radians(30)
        bev.harden_normals = True
    root.location = location
    root.rotation_euler.z = math.radians(90)
    return root


# ---------------------------------------------------------------- look

def glass_material(name, tint, film):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*tint, 1)
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.015
    bsdf.inputs["IOR"].default_value = ICE_IOR
    bsdf.inputs["Thin Film Thickness"].default_value = film
    bsdf.inputs["Thin Film IOR"].default_value = 1.45
    return m


def setup_scene(samples, res):
    sc = bpy.context.scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.max_bounces = 24
    sc.cycles.transmission_bounces = 24
    sc.cycles.glossy_bounces = 12
    sc.cycles.caustics_refractive = False
    sc.cycles.caustics_reflective = False
    sc.cycles.blur_glossy = 1.0
    sc.cycles.use_denoising = False
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    sc.world = world
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.3, 0.55, 0.8, 1)
    # bright ring high in the sky: the top faces mirror it, which is what reads as glass
    wt = world.node_tree
    wtc = wt.nodes.new("ShaderNodeTexCoord")
    sep = wt.nodes.new("ShaderNodeSeparateXYZ")
    band = wt.nodes.new("ShaderNodeValToRGB")
    els = band.color_ramp.elements
    els[0].position, els[0].color = 0.80, (0, 0, 0, 1)
    els[1].position, els[1].color = 0.86, (1, 1, 1, 1)
    e = els.new(0.93)
    e.color = (1, 1, 1, 1)
    e = els.new(0.98)
    e.color = (0, 0, 0, 1)
    strength = wt.nodes.new("ShaderNodeMath")
    strength.operation = "MULTIPLY_ADD"
    strength.inputs[1].default_value = 5.0
    strength.inputs[2].default_value = 0.8
    wt.links.new(wtc.outputs["Generated"], sep.inputs[0])
    wt.links.new(sep.outputs["Z"], band.inputs["Fac"])
    wt.links.new(band.outputs["Color"], strength.inputs[0])
    wt.links.new(strength.outputs["Value"], bg.inputs["Strength"])

    bpy.ops.object.empty_add(location=(0, 0, 0))
    rig = bpy.context.object
    rig.name = "CameraRig"

    # backdrop: pale ice centre fading to deep Access Blue, gives the glass something to refract
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, -0.9))
    back = bpy.context.object
    back.name = "Backdrop"
    m = bpy.data.materials.new("Backdrop")
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    tex = nt.nodes.new("ShaderNodeTexGradient")
    tex.gradient_type = "SPHERICAL"
    mapn = nt.nodes.new("ShaderNodeMapping")
    mapn.inputs["Scale"].default_value = (0.28, 0.28, 0.28)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.0, 0.10, 0.30, 1)
    ramp.color_ramp.elements[1].color = (0.55, 0.80, 0.95, 1)
    nt.links.new(tc.outputs["Object"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
    # soft diagonal bands: refraction bends them, which is how the eye reads clear glass
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 0.9
    wave.inputs["Distortion"].default_value = 2.0
    wave.inputs["Detail"].default_value = 1.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    shade = nt.nodes.new("ShaderNodeMath")
    shade.operation = "MULTIPLY_ADD"
    shade.inputs[1].default_value = 0.18
    shade.inputs[2].default_value = 0.82
    nt.links.new(wave.outputs["Fac"], shade.inputs[0])
    lum = nt.nodes.new("ShaderNodeMath")
    lum.operation = "MULTIPLY"
    nt.links.new(tex.outputs["Color"], lum.inputs[0])
    nt.links.new(shade.outputs["Value"], lum.inputs[1])
    nt.links.new(lum.outputs["Value"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    back.data.materials.append(m)

    back.parent = rig
    bpy.ops.object.camera_add(location=(0, -2.6, 5.2))
    cam = bpy.context.object
    cam.parent = rig
    cam.constraints.new("TRACK_TO").target = rig
    cam.data.lens = 85
    cam.data.dof.use_dof = True
    cam.data.dof.focus_object = rig
    cam.data.dof.aperture_fstop = 2.8
    sc.camera = cam

    # long strip lights read as crisp highlights along the bevelled glass edges
    for loc, energy, color in [((-2.2, -1.2, 2.6), 350, (1.0, 0.95, 0.9)),
                               ((2.4, 0.8, 2.2), 250, (0.8, 0.92, 1.0)),
                               ((0.0, 2.6, 1.0), 300, (0.7, 0.9, 1.0))]:
        bpy.ops.object.light_add(type="AREA", location=loc)
        lt = bpy.context.object
        lt.data.energy = energy
        lt.data.shape = "RECTANGLE"
        lt.data.size, lt.data.size_y = 3.0, 0.25
        lt.data.color = color
        lt.parent = rig
        lt.rotation_euler = (0, 0, 0)
        con = lt.constraints.new("TRACK_TO")
        con.target = rig

    return rig


# ---------------------------------------------------------------- variations
# A spec is one brand shape (x, y, scale, rotation_deg, mirror) in the sector above the
# first arm (arm along +x). d6_orbit repeats it 12 times: 6 arms, each with its mirror image.
# The brand shape keeps its aspect ratio throughout. Units: centre to tip is about 1.
# Tints run from clear ice to Access Blue.

def mix(t):
    return tuple(1 - t + t * c for c in ACCESS_BLUE)


VARIANTS = [
    dict(name="01 arrow arms", tint=mix(0.35), film=0,
         specs=[(0.76, 0.17, 0.46, 0), (0.38, 0.08, 0.2, 0), (0.17, 0.04, 0.07, 0)]),
    dict(name="02 swallowtail", tint=mix(0.2), film=380,
         specs=[(0.78, 0.15, 0.4, 0, True), (0.42, 0.07, 0.19, 0, True), (0.2, 0.04, 0.07, 0, True)]),
    dict(name="03 feather", tint=mix(0.45), film=0,
         specs=[(0.86, 0.23, 0.24, 30), (0.62, 0.17, 0.19, 30), (0.4, 0.11, 0.12, 30),
                (0.22, 0.06, 0.06, 30)]),
    dict(name="04 dendrite", tint=mix(0.25), film=0,
         specs=[(0.88, 0.05, 0.16, 0), (0.66, 0.05, 0.14, 0), (0.44, 0.04, 0.12, 0),
                (0.24, 0.03, 0.08, 0), (0.7, 0.2, 0.14, 60), (0.48, 0.16, 0.1, 60)]),
    dict(name="05 halo", tint=mix(0.55), film=0,
         specs=[(0.8, 0.12, 0.3, 0), (0.46, 0.14, 0.14, 90), (0.22, 0.05, 0.06, 90)]),
    dict(name="06 crown", tint=mix(0.3), film=450,
         specs=[(0.42, 0.088, 0.24, 0), (0.78, 0.13, 0.22, 30), (0.537, 0.25, 0.12, 30)]),
    dict(name="07 star burst", tint=mix(0.4), film=0,
         specs=[(0.78, 0.22, 0.46, 15), (0.34, 0.07, 0.13, 15)]),
    dict(name="08 sectored plate", tint=mix(0.15), film=320,
         specs=[(0.86, 0.12, 0.3, 0), (0.56, 0.12, 0.26, 0), (0.3, 0.06, 0.15, 0)]),
    dict(name="09 fern", tint=mix(0.5), film=0,
         specs=[(0.22 + 0.15 * k, 0.038, 0.09, 0) for k in range(5)]
               + [(0.34 + 0.15 * k, 0.11, 0.054, 60) for k in range(4)]),
    dict(name="10 inward arrows", tint=mix(0.25), film=400,
         specs=[(0.4, 0.086, 0.24, 0), (0.78, 0.112, 0.26, -25)]),
    dict(name="11 fork", tint=mix(0.6), film=0,
         specs=[(0.8, 0.25, 0.4, 30, True), (0.42, 0.08, 0.2, 0)]),
    dict(name="12 tiered", tint=mix(0.35), film=300, thick=0.2,
         specs=[(0.86, 0.12, 0.34, 0), (0.55, 0.1, 0.25, 0, True), (0.3, 0.05, 0.13, 0)]),
]


def contact_sheet(paths, out, cols=4):
    from PIL import Image
    ims = [Image.open(p) for p in paths]
    w, h = ims[0].size
    rows = math.ceil(len(ims) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(ims):
        sheet.paste(im.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    sheet.thumbnail((cols * 600, rows * 600))
    sheet.save(out)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--res", type=int, default=900)
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    only = {int(i) for i in args.only.split(",") if i}

    os.makedirs(RENDER_DIR, exist_ok=True)
    rig = setup_scene(args.samples, args.res)
    spacing = 3.0
    flakes = []
    for i, p in enumerate(VARIANTS, 1):
        loc = (((i - 1) % 4) * spacing, -((i - 1) // 4) * spacing, 0)
        mat = glass_material(f"Ice_{i:02d}", p["tint"], p["film"])
        flakes.append(build_snowflake(i, p, mat, loc))

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "snowflakes.blend"))

    sc = bpy.context.scene
    paths = []
    for i, ob in enumerate(flakes, 1):
        path = os.path.join(RENDER_DIR, f"snowflake_{i:02d}.png")
        paths.append(path)
        if only and i not in only:
            continue
        rig.location = ob.location
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"rendered {path}", flush=True)

    if all(os.path.exists(p) for p in paths):
        contact_sheet(paths, os.path.join(RENDER_DIR, "contact_sheet.png"))


if __name__ == "__main__":
    main()
