"""Export an evolved snowflake as a flat SVG.

Each shape is the original brand path (exact Bezier curves, not a traced outline), moved by
the same transform the flake uses, so the file stays faithful to the brand shape at any size.

Usage:
    python3 snowflakes/export_svg.py S06c-06 [--fill #00bdf2] [--size 1000] [--out PATH]
"""
import argparse
import itertools
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_snowflakes as base  # noqa: E402
from evolve_flakes import express, load_pool  # noqa: E402
from shapely import affinity  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402


def brand_segments():
    """The brand path as absolute segments: ('M', p), ('L', p), ('C', c1, c2, p)."""
    tokens = re.findall(r"[MHhLlCcVvZz]|-?\d*\.?\d+(?:e-?\d+)?", base.BRAND_PATH)
    segs, i, cmd, x, y = [], 0, None, 0.0, 0.0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                continue
        n = {"M": 2, "L": 2, "l": 2, "H": 1, "h": 1, "V": 1, "v": 1, "C": 6, "c": 6}[cmd]
        v = [float(t) for t in tokens[i:i + n]]
        i += n
        if cmd in "cC":
            if cmd == "c":
                v = [v[k] + (x if k % 2 == 0 else y) for k in range(6)]
            segs.append(("C", (v[0], v[1]), (v[2], v[3]), (v[4], v[5])))
            x, y = v[4], v[5]
            continue
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
        segs.append(("M" if cmd == "M" else "L", (x, y)))
    return segs


def svg_to_unit():
    """The affine map from brand-SVG coordinates to the unit brand shape (BRAND), built the
    same way as build_snowflakes.brand_unit: flip y, centre, turn the long diagonal onto x,
    scale the diagonal to 1."""
    poly = Polygon([(px, -py) for px, py in base.parse_svg_path(base.BRAND_PATH)]).buffer(0)
    cx, cy = poly.centroid.x, poly.centroid.y
    moved = affinity.translate(poly, -cx, -cy)
    p, q = max(itertools.combinations(moved.exterior.coords, 2), key=lambda pq: math.dist(*pq))
    th = -math.atan2(q[1] - p[1], q[0] - p[0])
    k = 1 / math.dist(p, q)
    c, s = math.cos(th), math.sin(th)

    def f(pt):
        x, y = pt[0] - cx, -pt[1] - cy
        return (k * (c * x - s * y), k * (s * x + c * y))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--fill", default="#00bdf2")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pool = load_pool()
    flake, _ = express(pool[args.code]["tiers"], args.code)
    shapes = base.layout(flake)
    extent = max(max(abs(v) for v in base.footprint(ga, gt).bounds) for ga, gt, _ in shapes)
    half = args.size / 2
    scale = half * 0.96 / extent
    to_unit = svg_to_unit()
    segs = brand_segments()

    def fmt(v):
        return f"{v:.2f}".rstrip("0").rstrip(".")

    paths = []
    for ga, gt, _ in shapes:
        def place(pt):
            ux, uy = to_unit(pt)
            x = ga[0][0] * ux + ga[0][1] * uy + gt[0]
            y = ga[1][0] * ux + ga[1][1] * uy + gt[1]
            # same orientation as the evolve gallery; SVG y runs downward
            return fmt(half + x * scale), fmt(half - y * scale)

        d = []
        for seg in segs:
            if seg[0] == "C":
                (a, b), (c2, d2), (e, f2) = (place(p) for p in seg[1:])
                d.append(f"C{a} {b} {c2} {d2} {e} {f2}")
            else:
                x, y = place(seg[1])
                d.append(f"{seg[0]}{x} {y}")
        paths.append(f'  <path d="{"".join(d)}Z"/>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {args.size} {args.size}" '
           f'width="{args.size}" height="{args.size}">\n'
           f'  <title>Snowflake {args.code}</title>\n'
           f'  <g fill="{args.fill}">\n' + "\n".join("  " + p for p in paths) + "\n  </g>\n</svg>\n")
    out = args.out or os.path.join(base.HERE, "evolve", f"{args.code}.svg")
    with open(out, "w") as fh:
        fh.write(svg)
    print(f"wrote {out}: {len(shapes)} shapes")


if __name__ == "__main__":
    main()
