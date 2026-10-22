"""Build 12 glass snowflake variations in Blender from the Access brand shape.

Every brand shape stays a distinct glass object (linked copies of one mesh), never merged.
The brand shape is close to a rhombus, so turned along its long diagonal it reads as a
crystal facet. Each arm is a stack of V rows: a shape and its mirror image opening outward
from the arm axis. Repeating that around 6 arms gives a crystal's mirror symmetry.

Usage:
    python3 snowflakes/build_snowflakes.py [--samples N] [--res PX] [--only i,j]

Outputs (next to this script):
    snowflakes.blend            all 12 variations side by side
    renders/snowflake_XX.png    one render per variation
    renders/contact_sheet.png   all renders in one image
"""
import argparse
import itertools
import math
import os
import re
import sys

import bpy
import shapely
from mathutils import Matrix
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(HERE, "renders")
GLASS_IOR = 1.5
GAP = 0.012  # minimum clear space between any two shapes


# ---------------------------------------------------------------- brand shape

# Access brand shape (AccessShape_AccessBlue), path copied verbatim from the brand SVG.
BRAND_PATH = (
    "M1264.07,0H341.4c-24.054,0-40.6,10.662-49.314,33.647L3.82,793.821"
    "c-11.056,29.161,2.402,48.069,33.672,48.069h922.712c24.054,0,40.579-10.604,49.318-33.65"
    "L1297.743,48.073c11.192-29.289-2.402-48.073-33.673-48.073"
)
ACCESS_BLUE = (0.0, 0.51, 0.89)  # #00bdf2 in linear RGB


def parse_svg_path(d, steps=12):
    """Supports the commands used by the brand path: M H h L l C c V v (absolute/relative)."""
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
    """Brand shape centred on the origin, y-up, turned so its long diagonal lies on +x with length 1."""
    poly = Polygon([(px, -py) for px, py in parse_svg_path(BRAND_PATH)]).buffer(0)
    poly = affinity.translate(poly, -poly.centroid.x, -poly.centroid.y)
    p, q = max(itertools.combinations(poly.exterior.coords, 2), key=lambda pq: math.dist(*pq))
    poly = affinity.rotate(poly, -math.atan2(q[1] - p[1], q[0] - p[0]), origin=(0, 0), use_radians=True)
    return affinity.scale(poly, 1 / math.dist(p, q), 1 / math.dist(p, q), origin=(0, 0))


BRAND = brand_unit()
BRAND_FIT = BRAND.simplify(0.004).buffer(0.002)


# ---------------------------------------------------------------- symmetry and layout

def piece_matrix(x, y, s, rot=0.0, mirror=False):
    """2x2 linear part and offset for one brand shape: scale s, optional mirror, rotate, place."""
    c, n = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    m = -1 if mirror else 1
    return [[c * s, -n * s * m], [n * s, c * s * m]], (x, y)


def d6_orbit(spec):
    """Place one shape spec in all 12 positions of the snowflake's mirror symmetry:
    6 arms, each with the shape's mirror image across the arm axis."""
    a, t = piece_matrix(*spec)
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


def row(t, a, s, axis=0):
    """A V row: brand shape of length s rooted on an axis (0 = arm, 30 = between arms) at
    distance t from the centre, opening outward at a degrees. Its mirror image comes from
    d6_orbit. The root is pushed off the axis just far enough that the two never touch."""
    c, n = math.cos(math.radians(axis)), math.sin(math.radians(axis))
    off = 0.0
    while True:
        lx = t + 0.5 * s * math.cos(math.radians(a))
        ly = off + 0.5 * s * math.sin(math.radians(a))
        spec = (c * lx - n * ly, n * lx + c * ly, s, a + axis, False)
        prints = [footprint(ga, gt, BRAND_FIT) for ga, gt, _ in d6_orbit(spec)]
        if float(shapely.distance(prints[0], prints[1:]).min()) >= GAP:
            return spec
        off += 0.003


def layout(p):
    """All shapes of one snowflake as (matrix, offset, scale). Fails if any two touch."""
    shapes = [o for spec in p["rows"] for o in d6_orbit(spec)]
    prints = [footprint(ga, gt) for ga, gt, _ in shapes]
    tree = shapely.STRtree(prints)
    for i, g in enumerate(prints):
        for j in tree.query(g.buffer(GAP)):
            if j > i and g.distance(prints[j]) < GAP:
                raise ValueError(f"shapes {i} and {j} touch in {p['name']}")
    return shapes


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


def brand_mesh(thick, material):
    """One unit brand-shape glass slab with a fully rounded rim. Every shape in every
    snowflake is a linked copy of it, so the rounding is applied once and shared."""
    me = prism_mesh("AccessShape", BRAND, -thick / 2, thick / 2)
    tmp = bpy.data.objects.new("tmp", me)
    bpy.context.scene.collection.objects.link(tmp)
    bev = tmp.modifiers.new("rim", "BEVEL")
    bev.width = thick * 0.48
    bev.segments = 10
    bev.profile = 0.5
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(40)
    bev.harden_normals = False
    # the brand shape is convex, so the inset can't fold over; without this the short
    # corner edges clamp the rounding to almost nothing
    bev.use_clamp_overlap = False
    dg = bpy.context.evaluated_depsgraph_get()
    rounded = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    rounded.name = "AccessShape_glass"
    bpy.data.objects.remove(tmp)
    bpy.data.meshes.remove(me)
    # only the rim is curved: the flat faces stay flat so the glass reads clear in the middle
    for f in rounded.polygons:
        f.use_smooth = abs(f.normal.z) < 0.999
    rounded.materials.append(material)
    return rounded


def build_snowflake(idx, p, mesh, location):
    root = bpy.data.objects.new(f"Snowflake_{idx:02d}", None)
    root.empty_display_size = 0.3
    bpy.context.scene.collection.objects.link(root)
    for k, (ga, gt, s) in enumerate(layout(p)):
        ob = bpy.data.objects.new(f"Snowflake_{idx:02d}_shape_{k:03d}", mesh)
        bpy.context.scene.collection.objects.link(ob)
        # thickness scales with the shape so small shapes stay in proportion
        ob.matrix_basis = Matrix(((ga[0][0], ga[0][1], 0, gt[0]),
                                  (ga[1][0], ga[1][1], 0, gt[1]),
                                  (0, 0, s, 0),
                                  (0, 0, 0, 1)))
        ob.parent = root
    # stand the flake upright, facing the camera, one arm pointing up
    root.location = location
    root.rotation_euler = (math.radians(90), math.radians(90), 0)
    return root


# ---------------------------------------------------------------- look

def glass_material():
    m = bpy.data.materials.new("AccessGlass")
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.86, 0.93, 1.0, 1)
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = GLASS_IOR
    # thin film tints the rim reflections orange and blue, standing in for dispersion
    bsdf.inputs["Thin Film Thickness"].default_value = 520
    bsdf.inputs["Thin Film IOR"].default_value = 1.33

    # where the rounded rim turns away from the camera, fade to a dark glossy navy: this is
    # the inked outline that makes each shape read as a solid piece of glass
    nt = m.node_tree
    rim = nt.nodes.new("ShaderNodeBsdfPrincipled")
    rim.inputs["Base Color"].default_value = (0.03, 0.06, 0.2, 1)
    rim.inputs["Metallic"].default_value = 1.0
    rim.inputs["Roughness"].default_value = 0.07
    rim.inputs["Thin Film Thickness"].default_value = 620
    rim.inputs["Thin Film IOR"].default_value = 1.4
    weight = nt.nodes.new("ShaderNodeLayerWeight")
    weight.inputs["Blend"].default_value = 0.5
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.38
    ramp.color_ramp.elements[1].position = 0.75
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(weight.outputs["Facing"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(rim.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], nt.nodes["Material Output"].inputs["Surface"])
    return m


def emission_material(name, color, strength):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    return m, nt, em


def setup_scene(samples, res):
    sc = bpy.context.scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.max_bounces = 32
    sc.cycles.transmission_bounces = 32
    sc.cycles.glossy_bounces = 16
    sc.cycles.caustics_refractive = False
    sc.cycles.caustics_reflective = False
    sc.cycles.blur_glossy = 1.0
    sc.cycles.use_denoising = False
    sc.render.resolution_x = res
    sc.render.resolution_y = round(res * 2 / 3)
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    sc.world = world
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.45, 0.62, 0.95, 1)
    bg.inputs["Strength"].default_value = 0.5

    bpy.ops.object.empty_add(location=(0, 0, 0))
    rig = bpy.context.object
    rig.name = "CameraRig"

    # glossy pale floor: carries the flake's reflection and soft light pools
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"
    fm = bpy.data.materials.new("Floor")
    fb = fm.node_tree.nodes["Principled BSDF"]
    fb.inputs["Base Color"].default_value = (0.22, 0.36, 0.72, 1)
    fb.inputs["Roughness"].default_value = 0.18
    fb.inputs["Coat Weight"].default_value = 1.0
    fb.inputs["Coat Roughness"].default_value = 0.03
    floor.data.materials.append(fm)
    floor.parent = rig

    # backdrop: bright centre glow falling off to soft blue, behind the flake
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 4.0, 3.0), rotation=(math.radians(90), 0, 0))
    back = bpy.context.object
    back.name = "Backdrop"
    back.scale = (30, 12, 1)
    m, nt, em = emission_material("Backdrop", (1, 1, 1), 0.9)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    mapn = nt.nodes.new("ShaderNodeMapping")
    # map the gradient onto the part of the backdrop the camera actually sees
    mapn.inputs["Location"].default_value = (0, 0.62, 0)
    mapn.inputs["Scale"].default_value = (6.5, 4.6, 1)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.22, 0.42, 0.85, 1)
    ramp.color_ramp.elements[1].color = (0.82, 0.90, 1.0, 1)
    nt.links.new(tc.outputs["Object"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], grad.inputs["Vector"])
    nt.links.new(grad.outputs["Color"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    back.data.materials.append(m)
    back.parent = rig

    # light shafts from above: long soft emissive strips fading downward
    shaft_mat, nt, em = emission_material("Shaft", (0.9, 0.95, 1.0), 2.5)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    fade = nt.nodes.new("ShaderNodeValToRGB")
    fade.color_ramp.elements[0].color = (0, 0, 0, 1)
    fade.color_ramp.elements[1].position = 1.0
    fade.color_ramp.elements[1].color = (1, 1, 1, 1)
    side = nt.nodes.new("ShaderNodeMath")
    side.operation = "PINGPONG"
    side.inputs[1].default_value = 0.5
    soft = nt.nodes.new("ShaderNodeMath")
    soft.operation = "MULTIPLY"
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    nt.links.new(sep.outputs["Y"], fade.inputs["Fac"])
    nt.links.new(sep.outputs["X"], side.inputs[0])
    nt.links.new(fade.outputs["Color"], soft.inputs[0])
    nt.links.new(side.outputs["Value"], soft.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(soft.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], nt.nodes["Material Output"].inputs["Surface"])
    for x, tilt, width in [(-2.6, -18, 0.9), (-1.2, -10, 0.6), (1.6, 12, 0.8), (3.0, 20, 0.5)]:
        bpy.ops.mesh.primitive_plane_add(size=1, location=(x, 3.6, 4.2),
                                         rotation=(math.radians(90), math.radians(tilt), 0))
        sh = bpy.context.object
        sh.name = "LightShaft"
        sh.scale = (width, 8, 1)
        sh.data.materials.append(shaft_mat)
        sh.visible_shadow = False
        sh.parent = rig

    bpy.ops.object.camera_add(location=(0, -7.4, 1.55))
    cam = bpy.context.object
    cam.parent = rig
    target = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(target)
    target.location = (0, 0, 1.12)
    target.parent = rig
    cam.constraints.new("TRACK_TO").target = target
    cam.data.lens = 70
    cam.data.dof.use_dof = True
    cam.data.dof.focus_object = target
    cam.data.dof.aperture_fstop = 4.0
    sc.camera = cam

    # key from above-front, two rim lights behind for edge glints, a top light for floor pools
    for loc, energy, size, color in [((0.0, -4.0, 5.0), 350, 4.0, (1.0, 1.0, 1.0)),
                                     ((-3.5, 2.5, 3.0), 600, 0.6, (0.85, 0.92, 1.0)),
                                     ((3.5, 2.5, 3.0), 600, 0.6, (1.0, 0.92, 0.85)),
                                     ((0.0, 0.5, 6.0), 120, 1.5, (1.0, 1.0, 1.0))]:
        bpy.ops.object.light_add(type="AREA", location=loc)
        lt = bpy.context.object
        lt.data.energy = energy
        lt.data.size = size
        lt.data.color = color
        lt.parent = rig
        lt.constraints.new("TRACK_TO").target = target
    return rig


# ---------------------------------------------------------------- variations
# rows: (t, a, s[, axis]) -> a V row rooted t from the centre, opening at a degrees, shape
# length s along its long diagonal. axis=30 roots it on the line between two arms.
# Units: centre to arm tip is about 1.

def flake(name, *rows):
    return dict(name=name, rows=[row(*r) for r in rows])


VARIANTS = [
    flake("01 frond", (0.78, 10, 0.3), (0.56, 50, 0.3), (0.34, 50, 0.22), (0.16, 45, 0.12)),
    flake("02 fern", (0.84, 8, 0.24), (0.66, 38, 0.2), (0.5, 38, 0.18), (0.35, 38, 0.15),
          (0.21, 38, 0.11)),
    flake("03 open ring", (0.8, 12, 0.28), (0.56, 55, 0.3), (0.34, 60, 0.22)),
    flake("04 dense frond", (0.8, 10, 0.28), (0.6, 45, 0.25), (0.41, 45, 0.2), (0.24, 45, 0.14),
          (0.1, 45, 0.08)),
    flake("05 wide wings", (0.78, 15, 0.3), (0.5, 65, 0.3), (0.26, 65, 0.18)),
    flake("06 spear", (0.9, 6, 0.24), (0.66, 25, 0.24), (0.42, 25, 0.22), (0.2, 25, 0.13)),
    flake("07 heavy core", (0.84, 10, 0.18), (0.64, 45, 0.2), (0.4, 45, 0.3), (0.16, 40, 0.12)),
    flake("08 crown", (0.8, 20, 0.3), (0.52, 50, 0.28), (0.3, 40, 0.18), (0.5, 40, 0.16, 30)),
    flake("09 twelve point", (0.8, 10, 0.26), (0.56, 45, 0.24), (0.34, 45, 0.18), (0.62, 30, 0.2, 30)),
    flake("10 halo", (0.8, 12, 0.26), (0.56, 75, 0.26), (0.32, 70, 0.2)),
    flake("11 feather", (0.86, 8, 0.18), *[(0.72 - 0.13 * k, 42, 0.15 - 0.012 * k) for k in range(5)]),
    flake("12 bold", (0.76, 14, 0.36), (0.46, 50, 0.36), (0.2, 40, 0.16)),
]


def contact_sheet(paths, out, cols=4):
    from PIL import Image
    ims = [Image.open(p) for p in paths]
    w, h = ims[0].size
    rows = math.ceil(len(ims) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(ims):
        sheet.paste(im.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    sheet.thumbnail((cols * 640, rows * 640))
    sheet.save(out)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--res", type=int, default=1200)
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    only = {int(i) for i in args.only.split(",") if i}

    os.makedirs(RENDER_DIR, exist_ok=True)
    rig = setup_scene(args.samples, args.res)
    mesh = brand_mesh(0.1, glass_material())
    spacing = 4.0
    flakes = []
    for i, p in enumerate(VARIANTS, 1):
        # bottom arm tip hovers just above the floor
        flakes.append(build_snowflake(i, p, mesh, ((i - 1) * spacing, 0, 1.12)))

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "snowflakes.blend"))

    sc = bpy.context.scene
    paths = []
    for i, ob in enumerate(flakes, 1):
        path = os.path.join(RENDER_DIR, f"snowflake_{i:02d}.png")
        paths.append(path)
        if only and i not in only:
            continue
        # show only this flake so neighbours never appear in reflections or refraction
        for other in flakes:
            for child in other.children:
                child.hide_render = other is not ob
        rig.location = (ob.location.x, 0, 0)
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"rendered {path}", flush=True)

    if all(os.path.exists(p) for p in paths):
        contact_sheet(paths, os.path.join(RENDER_DIR, "contact_sheet.png"))


if __name__ == "__main__":
    main()
