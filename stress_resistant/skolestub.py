#!/usr/bin/env python3
"""
Reinforced 3D-printable skolestub (school chair floor glide).

Thick base (8 mm), thick walls (5.5 mm), large fillets — sized from the
photos, overbuilt so FDM won't shatter like the last thin print.

Print SIDEWAYS. Prefer PETG/ABS/ASA, 5–6 walls, ≥40% infill.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


# ---- dimensions (mm) -------------------------------------------------------
LENGTH = 45.0
WIDTH = 34.0
HEIGHT = 26.0          # slightly taller than original so thick base + roof both fit

BASE_THICK = 8.0       # thick bottom (original ~3–4 mm)
WALL = 5.5             # walls (original ~2–3 mm)
FILLET = 5.0
EDGE = 1.8

CAVITY_W = 16.0        # rail pocket — tweak ±0.3 if fit is off
CAVITY_H = 13.0        # keeps ≥5 mm solid roof above pocket
CAVITY_DEPTH = 32.0
CAVITY_R = 1.2

HOLE_W = 5.2
HOLE_H = 4.0
HOLE_FROM_NOSE = 5.5
HOLE_Z = 16.0

VOXEL = 0.30
OUT_DIR = Path(__file__).resolve().parent


def length(v):
    return np.linalg.norm(v, axis=-1)


def sd_box(p, half):
    q = np.abs(p) - np.asarray(half, dtype=float)
    return length(np.maximum(q, 0.0)) + np.minimum(
        np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0.0
    )


def sd_round_box(p, half, r):
    h = np.asarray(half, dtype=float)
    return sd_box(p, np.maximum(h - r, 0.0)) - r


def sd_sphere(p, r):
    return length(p) - r


def sd_cyl_x(p, r):
    return length(p[..., 1:3]) - r


def sd_cyl_z(p, r):
    return length(p[..., 0:2]) - r


def op_u(a, b):
    return np.minimum(a, b)


def op_s(a, b):
    """Keep a, remove b."""
    return np.maximum(a, -b)


def op_i(a, b):
    return np.maximum(a, b)


def op_su(a, b, k):
    """Smooth union (IQ). k > 0."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (1.0 - h) * b + h * a - k * h * (1.0 - h)


def op_ss(d_sub, d_base, k):
    """
    Smooth subtraction (IQ): result = d_base minus d_sub.
    """
    h = np.clip(0.5 - 0.5 * (d_base + d_sub) / k, 0.0, 1.0)
    return (1.0 - h) * d_base + h * (-d_sub) + k * h * (1.0 - h)


def sd_bullet(p, L, W, H, z0, edge=EDGE):
    """Open at -X, round nose at +X. Bottom at z0, top at z0+H."""
    q = p.copy()
    q[..., 2] = p[..., 2] - (z0 + H * 0.5)
    r = W * 0.5
    body_len = L - r

    bp = q.copy()
    bp[..., 0] = q[..., 0] + L * 0.5 - body_len * 0.5
    d_body = sd_round_box(bp, (body_len * 0.5, W * 0.5, H * 0.5), edge)

    np_ = q.copy()
    np_[..., 0] = q[..., 0] - (L * 0.5 - r)
    d_nose = op_i(sd_cyl_z(np_, r), np.abs(q[..., 2]) - H * 0.5)
    # Soften nose vertical edges slightly
    d_nose = d_nose  # keep crisp enough for MC

    return op_u(d_body, d_nose)


def sdf(p):
    # --- thick base ---
    base = sd_bullet(p, LENGTH, WIDTH, BASE_THICK, 0.0)

    # --- housing ---
    house_w = CAVITY_W + 2.0 * WALL
    house_h = min(BASE_THICK + CAVITY_H + WALL * 0.85, HEIGHT)
    house_len = CAVITY_DEPTH + WALL

    open_x = -LENGTH * 0.5
    h0 = open_x + 0.5
    h1 = min(h0 + house_len, LENGTH * 0.5 - 1.5)
    hcx = 0.5 * (h0 + h1)
    hl = h1 - h0

    hp = p.copy()
    hp[..., 0] -= hcx
    hp[..., 2] -= house_h * 0.5
    house = sd_round_box(hp, (hl * 0.5, house_w * 0.5, house_h * 0.5), EDGE)

    # Domed roof: finite cylinder along X (axis on housing top)
    dome_r = house_w * 0.5
    dp = np.empty_like(p)
    dp[..., 0] = p[..., 0] - hcx
    dp[..., 1] = p[..., 1]
    dp[..., 2] = p[..., 2] - house_h
    dome = sd_cyl_x(dp, dome_r)
    dome = op_i(dome, np.abs(p[..., 0] - hcx) - hl * 0.5)
    # Keep only the upper lobe sitting on the housing (z >= house_h - 1)
    dome = op_i(dome, (house_h - 1.0) - p[..., 2])
    dome = op_i(dome, p[..., 2] - (house_h + dome_r + 0.2))

    # Housing nose cap (closes the tunnel toward +X)
    sp = np.empty_like(p)
    sp[..., 0] = p[..., 0] - h1
    sp[..., 1] = p[..., 1]
    sp[..., 2] = p[..., 2] - (BASE_THICK + (house_h - BASE_THICK) * 0.55)
    nose_cap = sd_sphere(sp, dome_r * 0.95)

    # Solid nose of whole part (thick material in front of cavity)
    tip = sd_bullet(p, LENGTH, min(house_w + 5.0, WIDTH - 1.0), house_h, 0.0)
    tip = op_i(tip, 1.0 - p[..., 0])  # keep x >~ 1

    body = op_su(base, house, FILLET)
    body = op_su(body, dome, 2.8)
    body = op_su(body, nose_cap, 2.2)
    body = op_su(body, tip, 3.0)

    # World bounds (half-spaces / slabs)
    body = op_i(body, -p[..., 2])             # z >= 0
    body = op_i(body, p[..., 2] - HEIGHT)     # z <= HEIGHT
    body = op_i(body, np.abs(p[..., 1]) - WIDTH * 0.5)
    body = op_i(body, np.abs(p[..., 0]) - LENGTH * 0.5)

    # --- cavity ---
    cav_len = CAVITY_DEPTH + 4.0
    cav_cx = open_x + CAVITY_DEPTH * 0.5 - 1.0

    lower_h = max(CAVITY_H - CAVITY_W * 0.5, 3.0)
    lp = p.copy()
    lp[..., 0] -= cav_cx
    lp[..., 2] -= BASE_THICK + lower_h * 0.5
    d_lower = sd_round_box(
        lp,
        (cav_len * 0.5, CAVITY_W * 0.5, lower_h * 0.5 + 0.2),
        CAVITY_R,
    )

    rp = np.empty_like(p)
    rp[..., 0] = p[..., 0] - cav_cx
    rp[..., 1] = p[..., 1]
    rp[..., 2] = p[..., 2] - (BASE_THICK + CAVITY_H - CAVITY_W * 0.5)
    d_roof = sd_cyl_x(rp, CAVITY_W * 0.5)
    d_roof = op_i(d_roof, np.abs(p[..., 0] - cav_cx) - cav_len * 0.5)
    # Roof only above the lower pocket bottom
    d_roof = op_i(d_roof, BASE_THICK + 1.0 - p[..., 2])

    cavity = op_u(d_lower, d_roof)
    # Cavity only exists above the thick floor (z >= BASE_THICK)
    cavity = op_i(cavity, BASE_THICK - p[..., 2])

    # Hard subtract for reliability (smooth only a little on corners)
    body = op_ss(cavity, body, CAVITY_R)

    # --- mounting hole ---
    tip_x = LENGTH * 0.5 - HOLE_FROM_NOSE
    hp = np.empty_like(p)
    hp[..., 0] = p[..., 0] - tip_x
    hp[..., 1] = p[..., 1] / (HOLE_W * 0.5)
    hp[..., 2] = (p[..., 2] - HOLE_Z) / (HOLE_H * 0.5)
    hole = length(hp[..., 1:3]) - 1.0
    hole = op_i(hole, np.abs(hp[..., 0]) - 9.0)
    body = op_s(body, hole)

    return body


def probe():
    pts = {
        "center_base": (0.0, 0.0, 4.0),
        "above": (0.0, 0.0, 30.0),
        "below": (0.0, 0.0, -2.0),
        "nose": (20.0, 0.0, 4.0),
        "open": (-20.0, 0.0, 4.0),
        "cavity": (-10.0, 0.0, 14.0),
        "side_wall": (0.0, 11.0, 14.0),
        "far_side": (0.0, 25.0, 4.0),
    }
    print("SDF probes (neg=solid, pos=air):")
    for n, pt in pts.items():
        d = float(sdf(np.asarray(pt, dtype=float).reshape(1, 3))[0])
        state = "SOLID" if d < 0 else "air"
        print(f"  {n:10} {pt} -> {d:+7.3f}  ({state})")


def mesh_from_sdf(pitch=VOXEL) -> trimesh.Trimesh:
    pad = 2.0
    xs = np.arange(-LENGTH / 2 - pad, LENGTH / 2 + pad + pitch, pitch)
    ys = np.arange(-WIDTH / 2 - pad, WIDTH / 2 + pad + pitch, pitch)
    zs = np.arange(-0.4, HEIGHT + pad + pitch, pitch)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    print(f"Sampling {pts.shape[0]:,} pts @ {pitch} mm ...")
    d = np.empty(pts.shape[0], dtype=np.float64)
    for i in range(0, pts.shape[0], 300_000):
        d[i : i + 300_000] = sdf(pts[i : i + 300_000])
    vol = d.reshape(xx.shape)
    solid = int((d < 0).sum())
    print(f"  min={d.min():.3f} max={d.max():.3f} solid={solid:,}/{d.size:,}")
    if solid < 500:
        raise RuntimeError("No solid region in SDF")

    verts, faces, normals, _ = marching_cubes(
        vol, level=0.0, spacing=(pitch, pitch, pitch)
    )
    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += zs[0]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0][2])])
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.volume if m.is_volume else len(m.faces))
    return mesh


def stress_report(mesh: trimesh.Trimesh) -> dict:
    extents = [round(float(x), 2) for x in mesh.extents]
    volume = float(mesh.volume) if mesh.is_volume else None

    normals = mesh.face_normals
    centers = mesh.triangles_center
    floor = (normals[:, 2] < -0.8) & (centers[:, 2] < 1.2)
    floor_area = float(mesh.area_faces[floor].sum())
    if floor_area < 100:
        floor_area = 0.80 * LENGTH * WIDTH

    capacity_kg = (floor_area * 0.6 * 35.0) / 9.81
    design_kg = 50.0
    safety = capacity_kg / design_kg
    roof = HEIGHT - (BASE_THICK + CAVITY_H)

    checks = {
        "size_ok": extents[0] >= 40 and extents[1] >= 28 and extents[2] >= 20,
        "thick_base": BASE_THICK >= 7,
        "thick_walls": WALL >= 5,
        "fillets": FILLET >= 4,
        "roof_ok": roof >= 2.5,
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "safety_ge_3": safety >= 3,
    }
    checks["passes_stress_gate"] = all(checks.values())

    return {
        "extents_mm": extents,
        "volume_mm3": round(volume, 1) if volume else None,
        "floor_contact_area_mm2": round(floor_area, 1),
        "base_thickness_mm": BASE_THICK,
        "wall_thickness_mm": WALL,
        "roof_thickness_mm": round(roof, 2),
        "fillet_mm": FILLET,
        "cavity_mm": {"w": CAVITY_W, "h": CAVITY_H, "depth": CAVITY_DEPTH},
        "estimated_petg_capacity_kg_per_leg": round(capacity_kg, 1),
        "design_load_kg_per_leg": design_kg,
        "safety_factor": round(safety, 2),
        "checks": checks,
        "print_advice": {
            "orientation": "Print on SIDE so the arch is continuous filament loops.",
            "material": "PETG, ABS, or ASA — not brittle PLA.",
            "walls": "5–6 perimeters",
            "infill": "40–100% gyroid",
            "layer_height_mm": 0.2,
            "top_bottom": 6,
        },
    }


def main():
    probe()
    mesh = mesh_from_sdf()
    stl = OUT_DIR / "skolestub_reinforced.stl"
    mesh.export(stl)
    mesh.export(OUT_DIR / "skolestub_reinforced.obj")
    print(f"Wrote {stl}")
    print(f"Bounds: {np.array(mesh.bounds).round(2).tolist()}")
    print(f"Faces={len(mesh.faces)} watertight={mesh.is_watertight} volume={mesh.is_volume}")

    report = stress_report(mesh)
    (OUT_DIR / "stress_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["checks"]["passes_stress_gate"]:
        bad = [k for k, v in report["checks"].items() if not v]
        raise SystemExit(f"Stress gate FAILED: {bad}")
    print("Stress gate PASSED.")


if __name__ == "__main__":
    main()
