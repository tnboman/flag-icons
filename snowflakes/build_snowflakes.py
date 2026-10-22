"""Build 12 glass snowflake variations in Blender from the sectored-plate reference shape.

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
import sys

import bpy
import shapely
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(HERE, "renders")
ICE_IOR = 1.31


# ---------------------------------------------------------------- 2D shape

def hexagon(r, cx=0.0, cy=0.0, rot=0.0):
    return Polygon([(cx + r * math.cos(rot + i * math.pi / 3),
                     cy + r * math.sin(rot + i * math.pi / 3)) for i in range(6)])


def bar(x0, x1, w):
    return box(x0, -w / 2, x1, w / 2)


def branch(pos, length, width, angle_deg):
    """Side branch rooted on the arm axis at x=pos, leaning outward by angle."""
    b = bar(-width, length, width)
    b = affinity.rotate(b, angle_deg, origin=(0, 0))
    return affinity.translate(b, pos, 0)


def make_arm(p):
    """One arm along +x. Every arm is the same shape, rotated 6 times."""
    L, w = p["arm_len"], p["arm_w"]
    parts = []
    if p.get("taper"):
        parts.append(Polygon([(0, w / 2), (L * 0.9, w * p["taper"] / 2),
                              (L, 0), (L * 0.9, -w * p["taper"] / 2), (0, -w / 2)]))
    else:
        parts.append(bar(0, L, w))

    for pos, length, width in p.get("branches", []):
        for s in (1, -1):
            parts.append(branch(pos * L, length, width, 60 * s))

    tip = p.get("tip")
    cut = None
    if tip == "notch":
        tw, td = p["tip_w"], p["tip_d"]
        parts.append(bar(L - td, L, tw))
        for s in (1, -1):
            parts.append(branch(L - td * 0.5, p["hook"], td * 0.55, 60 * s))
        cut = bar(L - p["notch_d"], L + 1, p["notch_w"])
    elif tip == "fork":
        for s in (1, -1):
            parts.append(branch(L - w * 0.4, p["prong"], w * 0.55, 30 * s))
    elif tip == "plate":
        parts.append(hexagon(p["plate_r"], L, 0))
    elif tip == "arrow":
        a = p["arrow"]
        parts.append(Polygon([(L - a, a * 0.9), (L + a * 0.6, 0), (L - a, -a * 0.9)]))
    elif tip == "fan":
        f = p["fan_w"]
        parts.append(Polygon([(L * 0.45, w / 2), (L, f / 2), (L, -f / 2), (L * 0.45, -w / 2)]))
        cut = Polygon([(L + 0.01, f * 0.28), (L - p["fan_cut"], 0), (L + 0.01, -f * 0.28)])

    arm = unary_union(parts)
    if cut is not None:
        arm = arm.difference(cut)
    return arm


def six_fold(geom):
    return unary_union([affinity.rotate(geom, 60 * i, origin=(0, 0)) for i in range(6)])


def silhouette(p):
    arm = make_arm(p)
    core = hexagon(p["hex_r"])
    if p.get("plate_r_center"):
        core = core.union(hexagon(p["plate_r_center"], rot=math.pi / 6))
    shape = unary_union([six_fold(arm), core])
    # soften tiny slivers so the bevel has clean edges to work with
    return shape.buffer(0.004, join_style="mitre").buffer(-0.004, join_style="mitre")


def surface_detail(p, sil):
    """Raised layers on top of the plate: inset plateau, centre hexagon ring, ribs."""
    rim = p.get("rim", 0.035)
    plateau = sil.buffer(-rim, join_style="mitre")
    layers = [(plateau, p["thick"], p["thick"] + p["plateau_h"])]

    top = p["thick"] + p["plateau_h"]
    rc = p["hex_r"] * 0.55
    ring = hexagon(rc).difference(hexagon(rc - p.get("ring_w", 0.03)))
    inner = hexagon(rc * 0.42, rot=math.pi / 6)
    rib_w = p.get("rib_w", 0.018)
    ribs = six_fold(bar(rc * 0.98, p["arm_len"] * 0.93, rib_w))
    for pos, length, _ in p.get("branches", []):
        for s in (1, -1):
            ribs = ribs.union(six_fold(branch(pos * p["arm_len"], length * 0.85, rib_w, 60 * s)))
    detail = unary_union([ring, inner, ribs]).intersection(plateau.buffer(-0.012))
    layers.append((detail, top - 0.004, top + p.get("rib_h", 0.012)))
    return layers


# ---------------------------------------------------------------- mesh

def polygons_of(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and g.area > 1e-7]


def prism_mesh(name, geom, z0, z1):
    verts, faces, index = [], [], {}

    def vid(x, y, z):
        key = (round(x, 6), round(y, 6), z)
        if key not in index:
            index[key] = len(verts)
            verts.append((x, y, z))
        return index[key]

    for poly in polygons_of(geom):
        poly = orient(poly.simplify(0.0005), 1.0)
        for tri in shapely.constrained_delaunay_triangles(poly).geoms:
            (ax, ay), (bx, by), (cx, cy) = list(tri.exterior.coords)[:3]
            if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
                (bx, by), (cx, cy) = (cx, cy), (bx, by)
            faces.append((vid(ax, ay, z1), vid(bx, by, z1), vid(cx, cy, z1)))
            faces.append((vid(ax, ay, z0), vid(cx, cy, z0), vid(bx, by, z0)))
        for ring in [poly.exterior, *poly.interiors]:
            c = list(ring.coords)[:-1]
            for i in range(len(c)):
                (ax, ay), (bx, by) = c[i], c[(i + 1) % len(c)]
                faces.append((vid(ax, ay, z0), vid(bx, by, z0), vid(bx, by, z1), vid(ax, ay, z1)))

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    return me


def build_snowflake(idx, p, material, location):
    sil = silhouette(p)
    base = bpy.data.objects.new(f"Snowflake_{idx:02d}", prism_mesh(f"sf{idx}_base", sil, 0.0, p["thick"]))
    bpy.context.scene.collection.objects.link(base)

    helpers = []
    for j, (geom, z0, z1) in enumerate(surface_detail(p, sil)):
        ob = bpy.data.objects.new(f"sf{idx}_layer{j}", prism_mesh(f"sf{idx}_layer{j}", geom, z0, z1))
        bpy.context.scene.collection.objects.link(ob)
        helpers.append(ob)
        mod = base.modifiers.new(f"union{j}", "BOOLEAN")
        mod.operation = "UNION"
        mod.solver = "MANIFOLD"
        mod.object = ob

    # bake the unions into one watertight mesh so the glass has no internal faces
    dg = bpy.context.evaluated_depsgraph_get()
    merged = bpy.data.meshes.new_from_object(base.evaluated_get(dg))
    old = base.data
    base.modifiers.clear()
    base.data = merged
    bpy.data.meshes.remove(old)
    for ob in helpers:
        me = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(me)

    # centre vertically so the plate's mid-plane is at z=0
    zs = [v.co.z for v in merged.vertices]
    mid = (min(zs) + max(zs)) / 2
    for v in merged.vertices:
        v.co.z -= mid

    bev = base.modifiers.new("bevel", "BEVEL")
    bev.width = p.get("bevel", 0.008)
    bev.segments = 3
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(30)
    bev.harden_normals = True
    base.modifiers.new("smooth", "WEIGHTED_NORMAL").keep_sharp = True
    for poly in merged.polygons:
        poly.use_smooth = True

    base.data.materials.clear()
    base.data.materials.append(material)
    base.location = location
    base.rotation_euler.z = math.radians(p.get("rot", 90))
    return base


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
    sc.cycles.caustics_refractive = True
    sc.cycles.caustics_reflective = True
    sc.cycles.blur_glossy = 0.5
    sc.cycles.use_denoising = False
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    sc.world = world
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.62, 0.64, 0.92, 1)
    bg.inputs["Strength"].default_value = 0.6

    # soft lavender backdrop with a gradient, like a microscope backlight
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
    mapn.inputs["Scale"].default_value = (0.35, 0.35, 0.35)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.42, 0.44, 0.78, 1)
    ramp.color_ramp.elements[1].color = (0.86, 0.87, 1.0, 1)
    nt.links.new(tc.outputs["Object"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    back.data.materials.append(m)

    bpy.ops.object.empty_add(location=(0, 0, 0))
    rig = bpy.context.object
    rig.name = "CameraRig"
    bpy.ops.object.camera_add(location=(0, -1.25, 4.6))
    cam = bpy.context.object
    cam.parent = rig
    cam.rotation_euler = (math.radians(15), 0, 0)
    cam.data.lens = 85
    cam.data.dof.use_dof = True
    cam.data.dof.focus_object = rig
    cam.data.dof.aperture_fstop = 5.6
    sc.camera = cam

    for loc, energy, size, color in [((-2.5, -2.0, 3.0), 90, 2.0, (1.0, 0.9, 0.85)),
                                     ((2.8, 1.5, 2.5), 60, 1.5, (0.75, 0.85, 1.0)),
                                     ((0.0, 3.0, 1.2), 40, 3.0, (0.9, 0.8, 1.0))]:
        bpy.ops.object.light_add(type="AREA", location=loc)
        lt = bpy.context.object
        lt.data.energy = energy
        lt.data.size = size
        lt.data.color = color
        lt.parent = rig
        lt.rotation_euler = (0, 0, 0)
        con = lt.constraints.new("TRACK_TO")
        con.target = rig
    return rig


# ---------------------------------------------------------------- variations
# arm_len and hex_r are in units where the flake is about 1 across from centre to tip.

BASE = dict(thick=0.055, plateau_h=0.022, hex_r=0.46, arm_w=0.38, arm_len=0.86)

VARIANTS = [
    # 01 reference: sectored plate, notched tips with hooks
    dict(BASE, tip="notch", tip_w=0.42, tip_d=0.16, notch_w=0.16, notch_d=0.10, hook=0.14,
         branches=[(0.55, 0.10, 0.12)], tint=(0.86, 0.92, 1.0), film=380),
    # 02 slender stellar dendrite
    dict(BASE, arm_w=0.10, arm_len=0.98, hex_r=0.18, tip="arrow", arrow=0.07,
         branches=[(0.35, 0.18, 0.06), (0.58, 0.24, 0.06), (0.80, 0.14, 0.05)],
         rib_w=0.012, rim=0.022, tint=(0.80, 0.93, 1.0), film=0),
    # 03 broad hexagonal plate with short notched lobes
    dict(BASE, hex_r=0.62, arm_w=0.34, arm_len=0.92, tip="notch", tip_w=0.38, tip_d=0.12,
         notch_w=0.14, notch_d=0.08, hook=0.07, tint=(0.92, 0.95, 1.0), film=450),
    # 04 fern: thin arms, dense short branches
    dict(BASE, arm_w=0.08, arm_len=1.0, hex_r=0.16, tip="arrow", arrow=0.05,
         branches=[(0.25 + 0.1 * k, 0.16 - 0.012 * k, 0.045) for k in range(7)],
         rib_w=0.010, rim=0.018, tint=(0.78, 0.95, 0.98), film=0),
    # 05 forked tips
    dict(BASE, arm_w=0.22, hex_r=0.26, tip="fork", prong=0.22,
         branches=[(0.45, 0.12, 0.10)], tint=(0.88, 0.88, 1.0), film=320),
    # 06 plates on stars (hexagonal plate on every tip)
    dict(BASE, arm_w=0.12, arm_len=0.78, hex_r=0.22, tip="plate", plate_r=0.20,
         rib_w=0.014, rim=0.026, tint=(0.84, 0.94, 1.0), film=0),
    # 07 arrowhead stellar with mid branches
    dict(BASE, arm_w=0.16, arm_len=0.92, hex_r=0.24, tip="arrow", arrow=0.16,
         branches=[(0.52, 0.16, 0.08)], tint=(0.93, 0.90, 1.0), film=500),
    # 08 two tiers of hooks plus notched tip
    dict(BASE, arm_w=0.24, arm_len=0.95, hex_r=0.26, tip="notch", tip_w=0.34, tip_d=0.12,
         notch_w=0.12, notch_d=0.08, hook=0.12, branches=[(0.42, 0.12, 0.10), (0.66, 0.14, 0.10)],
         tint=(0.80, 0.90, 1.0), film=0),
    # 09 tapered pointed star
    dict(BASE, arm_w=0.36, arm_len=1.0, hex_r=0.30, taper=0.45, tint=(0.90, 0.96, 1.0), film=420),
    # 10 fan sectors (broad sectored plate)
    dict(BASE, arm_w=0.22, arm_len=0.92, hex_r=0.30, tip="fan", fan_w=0.62, fan_cut=0.16,
         tint=(0.86, 0.92, 1.0), film=0),
    # 11 long arms with mid plates and fork tips
    dict(BASE, arm_w=0.10, arm_len=1.0, hex_r=0.20, tip="fork", prong=0.14,
         branches=[(0.55, 0.14, 0.14)], plate_r_center=0.26, rib_w=0.012, rim=0.022,
         tint=(0.82, 0.96, 1.0), film=360),
    # 12 compact chunky sectored plate
    dict(BASE, thick=0.08, plateau_h=0.03, hex_r=0.44, arm_w=0.40, arm_len=0.80, tip="notch",
         tip_w=0.50, tip_d=0.18, notch_w=0.20, notch_d=0.12, hook=0.10, rim=0.05,
         tint=(0.90, 0.90, 1.0), film=300),
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
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--res", type=int, default=1000)
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
