"""Evolve snowflake designs from the crystal (G) and frond (H) parents, for picking by eye.

A genome is a list of tiers. Each tier is a V row of brand shapes (distance t from the
centre, opening angle a, size s, axis 0 = on an arm or 30 = between arms) with optional
fine prongs at its tip. Mutation nudges and adds or removes tiers; crossover mixes parents.
Selection keeps designs with a clear scale hierarchy and readable arms, then picks a
varied set.

Usage:
    python3 snowflakes/evolve_flakes.py [--seed N] [--count 24]
    python3 snowflakes/evolve_flakes.py --parents S06 --strength 0.6
Outputs in snowflakes/evolve/ (or evolve/<parents>/): gallery.png and genomes.json.
"""
import argparse
import copy
import json
import math
import os
import random
import sys

import shapely
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_snowflakes as base  # noqa: E402
from dance_flakes import Flake  # noqa: E402

OUT = os.path.join(base.HERE, "evolve")
HERO = 0.2

PARENTS = {
    "G": [dict(t=0.76, a=12, s=0.28, axis=0, crown=(0.3, 30)),
          dict(t=0.44, a=50, s=0.24, axis=0, crown=(0.36, 34)),
          dict(t=0.63, a=26, s=0.13, axis=0),
          dict(t=0.27, a=34, s=0.13, axis=0),
          dict(t=0.13, a=30, s=0.1, axis=30),
          dict(t=0.45, a=16, s=0.11, axis=30)],
    "H": [dict(t=0.79, a=14, s=0.25, axis=0, crown=(0.28, 34)),
          dict(t=0.5, a=50, s=0.21, axis=0, crown=(0.32, 36)),
          dict(t=0.66, a=24, s=0.12, axis=0),
          dict(t=0.35, a=46, s=0.15, axis=0),
          dict(t=0.22, a=40, s=0.1, axis=0),
          dict(t=0.13, a=30, s=0.11, axis=30),
          dict(t=0.5, a=14, s=0.09, axis=30)],
}


def load_pool():
    """Every saved design by code: S01.. from the first round, S06-03 style from later ones."""
    pool = {}
    for root, _, files in os.walk(OUT):
        if "genomes.json" in files:
            with open(os.path.join(root, "genomes.json")) as fh:
                for k, v in json.load(fh).items():
                    pool.setdefault(k, v)
    return pool


# ---------------------------------------------------------------- genome -> flake

def express(genome, name="x"):
    """Build the flake: largest tiers first so heroes claim their space. Returns the flake
    dict and how many tiers made it in."""
    f = Flake(name)
    placed = []
    for g in sorted(genome, key=lambda g: -g["s"]):
        spec = f.add(base.row(g["t"], g["a"], g["s"], g["axis"]), shrink=False)
        if spec:
            placed.append((g, spec))
    for g, spec in placed:
        if g.get("crown"):
            f.crown(spec, *g["crown"])
    return f.done(), placed


# ---------------------------------------------------------------- fitness

def measure(genome):
    p, placed = express(genome)
    if len(placed) < len(genome):
        return None  # a tier collided: reject rather than silently drop it
    shapes = base.layout(p)
    prints = [base.footprint(ga, gt) for ga, gt, _ in shapes]
    extent = max(max(abs(c) for c in g.bounds) for g in prints)
    heroes = sum(1 for g in genome if g["s"] >= HERO)
    union = unary_union(prints)
    fill = union.area / (math.pi * extent ** 2)
    # arm continuity: along the arm axis, find the largest radial stretch with no glass
    arm = [g for g in prints if abs(math.atan2(g.centroid.y, g.centroid.x)) < math.radians(24)]
    covered = sorted((math.hypot(*min(g.exterior.coords, key=lambda c: math.hypot(*c))),
                      math.hypot(*max(g.exterior.coords, key=lambda c: math.hypot(*c)))) for g in arm)
    gap, reach = 0.0, 0.12
    for lo, hi in covered:
        gap = max(gap, lo - reach)
        reach = max(reach, hi)
    sizes = sorted({round(g["s"], 2) for g in genome})
    # occupancy map of one 30 degree sector (radius x angle), scaled to the flake's extent
    occ = []
    for i in range(10):
        r = (i + 0.5) / 10 * extent
        for j in range(5):
            th = math.radians((j + 0.5) * 6)
            pt = shapely.Point(r * math.cos(th), r * math.sin(th)).buffer(0.035 * extent)
            occ.append(1.0 if union.intersects(pt) else 0.0)
    return dict(p=p, n=len(shapes), extent=extent, heroes=heroes, fill=fill, gap=gap,
                spread=sizes[-1] / sizes[0], tiers=len(genome), occ=occ)


def score(m):
    if m is None or m["heroes"] < 2 or not (0.82 < m["extent"] < 1.12) or m["n"] > 168:
        return None
    s = 1.0
    s -= 4.0 * max(0.0, m["gap"] - 0.06)            # arms must read as continuous
    s -= 2.0 * abs(m["fill"] - 0.27)                # neither crumbs nor a solid disc
    s += 0.15 * min(m["spread"], 4.0)               # clear big-to-small hierarchy
    s -= 0.04 * max(0, m["tiers"] - 8)              # restraint
    return s


# ---------------------------------------------------------------- variation

def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def mutate(genome, rng, strength=1.0):
    """strength < 1 keeps children close to the parent: smaller nudges, rarer bold moves."""
    g = copy.deepcopy(genome)
    for _ in range(rng.randint(1, 3)):
        op = rng.random()
        if op < 0.12 * strength:
            # bold hero move: new angle, new place, or onto the between-arm axis
            tier = rng.choice([x for x in g if x["s"] >= HERO])
            choice = rng.random()
            if choice < 0.4:
                tier["a"] = round(rng.uniform(8, 65), 1)
            elif choice < 0.75:
                tier["t"] = round(rng.uniform(0.3, 0.82), 3)
            else:
                tier["axis"] = 30 - tier["axis"]
        elif op < 0.45:
            tier = rng.choice(g)
            tier["t"] = round(clamp(tier["t"] + rng.gauss(0, 0.05 * strength), 0.1, 0.86), 3)
            tier["a"] = round(clamp(tier["a"] + rng.gauss(0, 9 * strength), 6, 72), 1)
            tier["s"] = round(clamp(tier["s"] * rng.uniform(1 - 0.15 * strength, 1 + 0.18 * strength),
                                    0.06, 0.32), 3)
        elif op < 0.6 and len(g) < 9:
            big = rng.random() < 0.35
            g.append(dict(t=round(rng.uniform(0.15, 0.8), 3), a=round(rng.uniform(10, 60), 1),
                          s=round(rng.uniform(0.2, 0.28) if big else rng.uniform(0.07, 0.14), 3),
                          axis=rng.choice((0, 0, 30))))
        elif op < 0.72 and len(g) > 4:
            small = [x for x in g if x["s"] < HERO] or g
            g.remove(rng.choice(small))
        elif op < 0.84:
            tier = rng.choice(g)
            tier["crown"] = None if tier.get("crown") else (round(rng.uniform(0.25, 0.42), 2),
                                                           round(rng.uniform(26, 46), 1))
        else:
            tier = rng.choice([x for x in g if x["s"] < 0.16] or g)
            tier["axis"] = 30 - tier["axis"]
    return g


def crossover(a, b, rng):
    """Heroes from one parent, supporting tiers drawn from both."""
    heroes = [x for x in rng.choice((a, b)) if x["s"] >= HERO]
    rest = [x for x in a + b if x["s"] < HERO]
    rng.shuffle(rest)
    return copy.deepcopy(heroes + rest[:rng.randint(3, 5)])


# ---------------------------------------------------------------- search

def evolve(seed, count, parents=None, strength=1.0, generations=14, brood=60):
    rng = random.Random(seed)
    parents = parents or PARENTS
    pop = []
    for name, g in parents.items():
        m = measure(g)
        pop.append((score(m) or 0, g, m, name))
    archive = list(pop)
    for gen in range(generations):
        children = []
        for _ in range(brood):
            if rng.random() < 0.3:
                a, b = rng.sample([p[1] for p in pop], 2) if len(pop) > 1 else (pop[0][1], pop[0][1])
                child = mutate(crossover(a, b, rng), rng, strength)
            else:
                child = mutate(rng.choice(pop)[1], rng, strength)
            m = measure(child)
            sc = score(m)
            if sc is not None:
                children.append((sc, child, m, f"g{gen}"))
        archive += children
        # keep the best, plus a few odd ones out so the search keeps exploring
        ranked = sorted(pop + children, key=lambda x: -x[0])
        pop = ranked[:10] + rng.sample(ranked[10:], min(6, len(ranked[10:])))
        print(f"generation {gen}: {len(children)} viable, best {pop[0][0]:.2f}", flush=True)
    return pick_varied(archive, count, keep=archive[:len(parents)] if parents is not PARENTS else ())


def features(m):
    return tuple(m["occ"])


def pick_varied(archive, count, keep=()):
    """Greedy: parents (if kept) or the best first, then whichever good design is most unlike
    those already picked."""
    good = sorted([x for x in archive if x[2] is not None], key=lambda x: -x[0])[:count * 8]
    picked = list(keep) or [good[0]]
    good = [c for c in good if c not in picked]
    while len(picked) < count and len(picked) < len(good):
        def novelty(c):
            return min(math.dist(features(c[2]), features(p[2])) for p in picked)
        best = max((c for c in good if c not in picked), key=lambda c: novelty(c) + 0.6 * c[0])
        picked.append(best)
    return picked


def gallery(picked, path, labels, cols=6):
    from PIL import Image, ImageDraw
    W = 360
    rows = math.ceil(len(picked) / cols)
    sheet = Image.new("RGB", (cols * W, rows * W), "white")
    d = ImageDraw.Draw(sheet)
    for i, (sc, g, m, origin) in enumerate(picked):
        ox, oy = (i % cols) * W + W / 2, (i // cols) * W + W / 2
        k = W * 0.46 / m["extent"]
        for ga, gt, s in base.layout(m["p"]):
            poly = base.footprint(ga, gt)
            d.polygon([(ox + x * k, oy - y * k) for x, y in poly.exterior.coords], fill=(0, 140, 220))
        d.text(((i % cols) * W + 8, (i // cols) * W + 6), labels[i], fill="black")
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--parents", default="", help="breed from picked designs, e.g. S06 or S06,S12")
    ap.add_argument("--strength", type=float, default=1.0, help="mutation size; below 1 stays close")
    args = ap.parse_args()
    parents, out, prefix = None, OUT, "S"
    if args.parents:
        pool = load_pool()
        parents = {k: pool[k]["tiers"] for k in args.parents.split(",")}
        prefix = "+".join(parents) + "-"
        out = os.path.join(OUT, "+".join(parents))
    os.makedirs(out, exist_ok=True)
    picked = evolve(args.seed, args.count, parents, args.strength)
    labels = [o if parents and o in parents else f"{prefix}{i:02d}" for i, (_, _, _, o) in enumerate(picked)]
    labels = [f"{x} (parent)" if parents and x in parents else x for x in labels]
    gallery(picked, os.path.join(out, "gallery.png"), labels)
    with open(os.path.join(out, "genomes.json"), "w") as fh:
        json.dump({lab.split()[0]: dict(score=round(sc, 3), origin=origin, tiers=g)
                   for lab, (sc, g, m, origin) in zip(labels, picked)}, fh, indent=1)
    print("wrote", len(picked), "designs", flush=True)


if __name__ == "__main__":
    main()
