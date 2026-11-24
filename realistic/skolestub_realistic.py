#!/usr/bin/env python3
"""
Realistic skolestub — matched to reference photos.

Looks like the real part:
  • Flat open end with U-tunnel for the chair foot
  • Wider floor flange under a raised arched housing
  • Slipper profile: tall at the opening → lower rounded nose
  • Continuous arch roof over the cavity (not cut through!)
  • Oval hole in the closed nose tip

Chair foot fits in a ~15.5 × ~15 mm pocket at the open end.
Reinforced vs original: thicker base/walls + fillets.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# ---- mm from ruler photos + strength overbuild -----------------------------
LENGTH = 44.0
WIDTH = 34.0
HEIGHT = 24.0          # peak at open end

BASE_THICK = 5.0       # thick floor (original ~3)
WALL = 4.0             # shell around cavity (original ~2–3)
FILLET = 3.5
EDGE = 1.4

CAVITY_W = 15.5        # chair-foot width clearance
# Height at OPENING ≈ HEIGHT - WALL - BASE = 24-4-5 = 15
CAVITY_DEPTH = 28.0    # seat depth; stops before nose
CAVITY_R = 0.9

HOUSE_W = CAVITY_W + 2.0 * WALL  # 23.5
HOUSE_H_OPEN = HEIGHT
HOUSE_H_NOSE = 13.0

HOLE_W = 5.0
HOLE_H = 3.6
HOLE_FROM_NOSE = 5.0
HOLE_Z = 11.5

VOXEL = 0.26
OUT_DIR = Path(__file__).resolve().parent


def _len(v):
    return np.linalg.norm(v, axis=-1)


def sd_box(p, half):
    q = np.abs(p) - np.asarray(half, dtype=float)
    return _len(np.maximum(q, 0.0)) + np.minimum(
        np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0.0
    )


def sd_rbox(p, half, r):
    return sd_box(p, np.maximum(np.asarray(half, dtype=float) - r, 0.0)) - r


def sd_sphere(p, r):
    return _len(p) - r


def sd_cyl_x(p, r):
    return _len(p[..., 1:3]) - r


def sd_cyl_z(p, r):
    return _len(p[..., 0:2]) - r


def op_u(a, b):
    return np.minimum(a, b)


def op_s(a, b):
    return np.maximum(a, -b)


def op_i(a, b):
    return np.maximum(a, b)


def op_su(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0)
    return (1.0 - h) * b + h * a - k * h * (1.0 - h)


def sd_bullet(p, L, W, H, z0, edge=EDGE):
    q = p.copy()
    q[..., 2] = p[..., 2] - (z0 + H * 0.5)
    r = W * 0.5
    body = L - r
    bp = q.copy()
    bp[..., 0] = q[..., 0] + L * 0.5 - body * 0.5
    d0 = sd_rbox(bp, (body * 0.5, W * 0.5, H * 0.5), edge)
    np_ = q.copy()
    np_[..., 0] = q[..., 0] - (L * 0.5 - r)
    d1 = op_i(sd_cyl_z(np_, r), np.abs(q[..., 2]) - H * 0.5)
    return op_u(d0, d1)


def taper_h(x):
    """Outer housing height: tall at open (−X), lower at nose (+X)."""
    t = np.clip((x + LENGTH * 0.5) / LENGTH, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)
    return HOUSE_H_OPEN * (1.0 - s) + HOUSE_H_NOSE * s


def cavity_top(x):
    """Inner roof follows outer shell minus wall — keeps continuous arch."""
    return np.maximum(taper_h(x) - WALL, BASE_THICK + 4.0)


def sdf(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    open_x = -LENGTH * 0.5

    # --- floor flange ---
    base = sd_bullet(p, LENGTH, WIDTH, BASE_THICK, 0.0)
    lip = sd_bullet(p, LENGTH - 0.6, WIDTH - 0.6, BASE_THICK + 0.9, 0.0)
    base = op_su(base, lip, 0.9)

    # --- raised arched housing (slipper taper) ---
    h = taper_h(x)
    hw = HOUSE_W * 0.5
    top_r = hw
    lower = np.maximum(h - top_r, 2.0)

    cy = np.abs(y) - hw
    cz = np.maximum(-z, z - lower)
    d_lower = _len(np.stack([np.maximum(cy, 0), np.maximum(cz, 0)], axis=-1)) + np.minimum(
        np.maximum(cy, cz), 0
    )
    d_top = _len(np.stack([y, z - (h - top_r)], axis=-1)) - top_r
    d_top = op_i(d_top, (h - top_r - 0.6) - z)
    house = op_u(d_lower, d_top)

    house_end = LENGTH * 0.5 - 6.5
    house = op_i(house, open_x - 0.15 - x)
    house = op_i(house, x - house_end)

    # --- solid rounded nose ---
    tip = sd_sphere(
        np.stack([x - (LENGTH * 0.5 - 8.5), y * 1.06, z - 7.0], axis=-1),
        10.0,
    )
    tip = op_i(tip, 2.0 - x)
    tip = op_i(tip, -z)
    tip = op_i(tip, z - 15.5)

    toe_p = p.copy()
    toe_p[..., 0] = x - 11.0
    toe = sd_bullet(toe_p, 22.0, HOUSE_W + 1.5, HOUSE_H_NOSE, 0.0)
    toe = op_i(toe, 4.0 - x)

    body = op_su(base, house, FILLET)
    body = op_su(body, tip, 2.6)
    body = op_su(body, toe, 2.0)

    body = op_i(body, -z)
    body = op_i(body, z - (HEIGHT + 0.6))
    body = op_i(body, np.abs(y) - WIDTH * 0.5)
    body = op_i(body, np.abs(x) - LENGTH * 0.5)

    # --- chair-foot cavity: U-tunnel, roof follows shell ---
    cav_len = CAVITY_DEPTH + 5.0
    cav_cx = open_x + CAVITY_DEPTH * 0.5 - 1.5
    ctop = cavity_top(x)
    cav_h = np.maximum(ctop - BASE_THICK, 1.0)
    top_r_i = np.minimum(CAVITY_W * 0.5, cav_h * 0.55)
    lower_h = np.maximum(cav_h - top_r_i, 1.5)

    # Lower pocket (variable height via z-slab trick is hard in pure SDF;
    # approximate with round box to mean height + cylinder roof clamped by ctop)
    mean_lower = float(np.mean([15.5 - CAVITY_W * 0.5, 8.0]))  # unused
    # Use pointwise: outside if z > ctop OR z < BASE OR |y| > W/2 OR x out of range
    # Build as intersection of half-spaces + round top — classic arch SDF:

    # Infinite arch prism in YZ, then bound X
    # Lower: |y|<=cw/2, BASE<=z<=BASE+lower_h  where lower_h = cav_h - top_r_i
    cw = CAVITY_W * 0.5
    cyi = np.abs(y) - cw
    # z in [BASE, BASE+lower_h] with lower_h depending on x
    z0 = BASE_THICK
    z1 = BASE_THICK + lower_h
    czi = np.maximum(z0 - z, z - z1)
    d_low = _len(np.stack([np.maximum(cyi, 0), np.maximum(czi, 0)], axis=-1)) + np.minimum(
        np.maximum(cyi, czi), 0
    )
    d_low = d_low - CAVITY_R  # slight round (approx)

    # Round inner roof: circle center at z = ctop - top_r_i
    d_iroof = _len(np.stack([y, z - (ctop - top_r_i)], axis=-1)) - top_r_i
    d_iroof = op_i(d_iroof, (ctop - top_r_i - 0.5) - z)

    cavity = op_u(d_low, d_iroof)
    # X bounds: poke out of open face, stop before nose
    cavity = op_i(cavity, (open_x - 2.0) - x)          # x >= open-2
    cavity = op_i(cavity, x - (open_x + CAVITY_DEPTH))  # x <= open+depth
    # Never cut below thick floor
    cavity = op_i(cavity, BASE_THICK - 0.05 - z)

    # Only keep cavity where there is still enough clear height for a foot
    # (near nose taper, cavity naturally shrinks; stop when too low)
    cavity = op_i(cavity, 9.0 - cav_h)  # require cav_h >= 9

    body = op_s(body, cavity)

    # --- nose oval hole ---
    tip_x = LENGTH * 0.5 - HOLE_FROM_NOSE
    hp = np.stack(
        [x - tip_x, y / (HOLE_W * 0.5), (z - HOLE_Z) / (HOLE_H * 0.5)],
        axis=-1,
    )
    hole = _len(hp[..., 1:3]) - 1.0
    hole = op_i(hole, np.abs(hp[..., 0]) - 7.0)
    body = op_s(body, hole)

    return body


def mesh_from_sdf(pitch=VOXEL) -> trimesh.Trimesh:
    pad = 2.0
    xs = np.arange(-LENGTH / 2 - pad, LENGTH / 2 + pad + pitch, pitch)
    ys = np.arange(-WIDTH / 2 - pad, WIDTH / 2 + pad + pitch, pitch)
    zs = np.arange(-0.35, HEIGHT + pad + pitch, pitch)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    print(f"Sampling {pts.shape[0]:,} @ {pitch} mm")
    d = np.empty(pts.shape[0], dtype=np.float64)
    for i in range(0, len(pts), 300_000):
        d[i : i + 300_000] = sdf(pts[i : i + 300_000])
    print(f"  solid={(d < 0).sum():,}  min={d.min():.2f} max={d.max():.2f}")

    verts, faces, normals, _ = marching_cubes(
        d.reshape(xx.shape), level=0.0, spacing=(pitch,) * 3
    )
    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += zs[0]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.apply_translation([0, 0, -float(mesh.bounds[0][2])])
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.volume if m.is_volume else len(m.faces))
    return mesh


def cavity_opening_check() -> dict:
    x = -LENGTH * 0.5 + 2.0
    y_lim = HOUSE_W * 0.5
    ys = np.linspace(-y_lim, y_lim, 141)
    zs = np.linspace(BASE_THICK + 0.2, HEIGHT - 0.5, 100)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    pts = np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])
    air = sdf(pts) > 0.0
    # Restrict to likely cavity band
    central = air & (np.abs(yy.ravel()) <= CAVITY_W * 0.5 + 0.8)
    if not central.any():
        return {"opening_ok": False, "clear_width_mm": 0.0, "clear_height_mm": 0.0}
    ya, za = yy.ravel()[central], zz.ravel()[central]
    w = float(ya.max() - ya.min())
    h = float(za.max() - za.min())
    target_h = float(cavity_top(np.array([x]))[0] - BASE_THICK)
    return {
        "clear_width_mm": round(w, 2),
        "clear_height_mm": round(h, 2),
        "target_w": CAVITY_W,
        "target_h_at_opening": round(target_h, 2),
        "opening_ok": w >= CAVITY_W * 0.9 and h >= 13.0,
    }


def report(mesh: trimesh.Trimesh) -> dict:
    extents = [round(float(x), 2) for x in mesh.extents]
    opening = cavity_opening_check()

    # Roof present mid-cavity?
    x = -8.0
    zs = np.linspace(0, HEIGHT + 1, 120)
    pts = np.column_stack([np.full_like(zs, x), np.zeros_like(zs), zs])
    inside = sdf(pts) < 0
    zmax = float(zs[inside].max()) if inside.any() else 0.0
    roof_ok = zmax >= BASE_THICK + 12.0

    # Measure wall in the straight vertical section (not the round roof)
    ys = np.linspace(0, 16, 321)
    pts = np.column_stack([np.full_like(ys, -8.0), ys, np.full_like(ys, BASE_THICK + 3.5)])
    inside = sdf(pts) < 0
    segs, start = [], None
    for i, v in enumerate(inside):
        if v and start is None:
            start = float(ys[i])
        if (not v) and start is not None:
            segs.append((start, float(ys[i - 1])))
            start = None
    if start is not None:
        segs.append((start, float(ys[-1])))
    wall = max((b - a) for a, b in segs) if segs else 0.0

    checks = {
        "size_ok": extents[0] >= 40 and extents[1] >= 30 and 20 <= extents[2] <= 28,
        "thick_base": BASE_THICK >= 4.5,
        "thick_walls": wall >= 3.0,
        "continuous_roof": roof_ok,
        "chair_foot_fits": bool(opening["opening_ok"]),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
    }
    checks["passes"] = all(checks.values())
    return {
        "extents_mm": extents,
        "volume_mm3": round(float(mesh.volume), 1) if mesh.is_volume else None,
        "base_mm": BASE_THICK,
        "wall_mm": WALL,
        "wall_measured_mm": round(wall, 2),
        "mid_roof_zmax_mm": round(zmax, 2),
        "cavity_mm": {"w": CAVITY_W, "depth": CAVITY_DEPTH},
        "opening": opening,
        "checks": checks,
        "print_advice": {
            "orientation": "SIDEWAYS for strength",
            "material": "PETG / ABS / ASA",
            "walls": "4–6",
            "infill": "30–60%+",
            "fit_tweak": "CAVITY_W / WALL / HEIGHT if fit is off",
        },
    }


def preview(mesh: trimesh.Trimesh, path: Path) -> None:
    rng = np.random.default_rng(3)
    idx = rng.choice(len(mesh.faces), size=min(18000, len(mesh.faces)), replace=False)
    d = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)
    fig = plt.figure(figsize=(12, 4))
    views = [(22, -45, "iso"), (90, -90, "top"), (5, 180, "into opening")]
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        coll = Poly3DCollection(
            d.vertices[d.faces], alpha=0.95, linewidths=0.015, edgecolor="0.2"
        )
        coll.set_facecolor((0.08, 0.08, 0.1, 1))
        ax.add_collection3d(coll)
        b = mesh.bounds
        ax.set_xlim(b[0, 0], b[1, 0])
        ax.set_ylim(b[0, 1], b[1, 1])
        ax.set_zlim(b[0, 2], b[1, 2])
        ax.set_box_aspect(tuple(float(x) for x in mesh.extents))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
    fig.suptitle(
        f"Realistic skolestub  {LENGTH:.0f}×{WIDTH:.0f}×{HEIGHT:.0f} mm  "
        f"cavity {CAVITY_W} wide  base {BASE_THICK}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print("Wrote", path)


def main():
    print("Probes:")
    for name, pt in {
        "base": (0, 0, 2.5),
        "flange": (0, 14, 2.5),
        "cavity": (-8, 0, BASE_THICK + 6),
        "roof": (-8.0, 0.0, float(taper_h(np.array([-8.0]))[0]) - 1.0),
        "wall": (-8, CAVITY_W / 2 + WALL / 2, BASE_THICK + 5),
        "nose": (16, 0, 8),
    }.items():
        d = float(sdf(np.array(pt, float).reshape(1, 3))[0])
        print(f"  {name:7} {pt} -> {d:+.2f} ({'SOLID' if d < 0 else 'air'})")

    print("Opening:", json.dumps(cavity_opening_check(), indent=2))
    print("Heights along centerline:")
    for x in np.linspace(-20, 20, 11):
        zs = np.linspace(0, HEIGHT + 1, 80)
        pts = np.column_stack([np.full_like(zs, x), np.zeros_like(zs), zs])
        inside = sdf(pts) < 0
        zmax = float(zs[inside].max()) if inside.any() else 0.0
        oh = float(taper_h(np.array([x]))[0])
        print(f"  x={x:+6.1f}  zmax={zmax:5.1f}  outer_h={oh:5.1f}")

    mesh = mesh_from_sdf()
    mesh.export(OUT_DIR / "skolestub_realistic.stl")
    mesh.export(OUT_DIR / "skolestub_realistic.obj")
    print("Bounds", np.array(mesh.bounds).round(2).tolist())
    print("watertight", mesh.is_watertight, "volume", round(float(mesh.volume), 1) if mesh.is_volume else None)

    rep = report(mesh)
    (OUT_DIR / "fit_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    preview(mesh, OUT_DIR / "preview.png")

    if not rep["checks"]["passes"]:
        raise SystemExit(f"FAILED: {[k for k, v in rep['checks'].items() if not v]}")
    print("PASSED — realistic skolestub ready.")


if __name__ == "__main__":
    main()
