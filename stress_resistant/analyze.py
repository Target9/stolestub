#!/usr/bin/env python3
"""Section probes + preview image for the reinforced skolestub."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import skolestub as sk

ROOT = Path(__file__).resolve().parent


def solid_mask(pts: np.ndarray) -> np.ndarray:
    return sk.sdf(pts) < 0.0


def main() -> None:
    m = trimesh.load(ROOT / "skolestub_reinforced.stl", force="mesh")
    print("extents", m.extents)
    print("watertight", m.is_watertight, "volume_mm3", round(float(m.volume), 1))

    xs = np.linspace(-18, 18, 13)
    print("\nSection probes (Y solid span at z=12, Z solid span at y=0):")
    for x in xs:
        ys = np.linspace(-20, 20, 401)
        pts = np.column_stack([np.full_like(ys, x), ys, np.full_like(ys, 12.0)])
        inside = solid_mask(pts)
        y_span = float(ys[inside].max() - ys[inside].min()) if inside.any() else 0.0

        zs = np.linspace(-1, 28, 291)
        ptsz = np.column_stack([np.full_like(zs, x), np.zeros_like(zs), zs])
        inz = solid_mask(ptsz)
        if inz.any():
            z_span = float(zs[inz].max() - zs[inz].min())
            z_max = float(zs[inz].max())
        else:
            z_span = z_max = 0.0
        print(
            f"  x={x:+6.1f}  y_span@z12={y_span:5.1f}  "
            f"z_span@y0={z_span:5.1f}  z_max={z_max:5.1f}"
        )

    ys = np.linspace(0, 20, 401)
    pts = np.column_stack([np.full_like(ys, -8.0), ys, np.full_like(ys, 14.0)])
    inside = solid_mask(pts)
    segs = []
    start = None
    for i, v in enumerate(inside):
        if v and start is None:
            start = float(ys[i])
        if (not v) and start is not None:
            segs.append((start, float(ys[i - 1])))
            start = None
    if start is not None:
        segs.append((start, float(ys[-1])))
    print("\nSolid Y segments at x=-8, z=14 (half-side from centerline):")
    for a, b in segs:
        print(f"  {a:.2f} .. {b:.2f}  thickness={b - a:.2f} mm")

    # Min roof thickness above cavity centerline
    zs = np.linspace(sk.BASE_THICK + sk.CAVITY_H - 1, sk.HEIGHT + 1, 200)
    pts = np.column_stack([np.full_like(zs, -8.0), np.zeros_like(zs), zs])
    inz = solid_mask(pts)
    if inz.any():
        roof_solid = zs[inz]
        print(
            f"\nRoof solid Z at x=-8,y=0: {roof_solid.min():.2f} .. {roof_solid.max():.2f} "
            f"(thickness {roof_solid.max() - roof_solid.min():.2f} mm)"
        )

    idx = np.random.default_rng(0).choice(
        len(m.faces), size=min(12000, len(m.faces)), replace=False
    )
    d = trimesh.Trimesh(vertices=m.vertices, faces=m.faces[idx], process=False)

    fig = plt.figure(figsize=(12, 4))
    views = [(25, -60, "iso"), (90, -90, "top"), (5, 0, "side")]
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        tris = d.vertices[d.faces]
        coll = Poly3DCollection(tris, alpha=0.95, linewidths=0.02, edgecolor="0.2")
        coll.set_facecolor((0.12, 0.12, 0.14, 1.0))
        ax.add_collection3d(coll)
        b = m.bounds
        ax.set_xlim(b[0, 0], b[1, 0])
        ax.set_ylim(b[0, 1], b[1, 1])
        ax.set_zlim(b[0, 2], b[1, 2])
        ax.set_box_aspect(tuple(float(x) for x in m.extents))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    fig.suptitle("Skolestub reinforced — 45×34×26 mm · 8 mm base · 5.5 mm walls")
    fig.tight_layout()
    out = ROOT / "preview.png"
    fig.savefig(out, dpi=160)
    print(f"\nWrote {out}")

    report_path = ROOT / "stress_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["measured_wall_segments_mm"] = [
        {"y0": round(a, 2), "y1": round(b, 2), "thickness": round(b - a, 2)}
        for a, b in segs
    ]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
