# Reinforced skolestub (3D printable)

Stress-resistant school-chair floor glide, rebuilt from measurements of the original part. The previous thin print shattered — this version is intentionally **thick, round, and overbuilt**.

## Deliverables

| File | What it is |
|------|------------|
| `skolestub_reinforced.stl` | Ready-to-slice mesh |
| `skolestub_reinforced.obj` | Same mesh (preview / CAD) |
| `skolestub.py` | Parametric SDF generator + stress gate |
| `skolestub.scad` | OpenSCAD twin for quick dimension edits |
| `stress_report.json` | Geometry / strength sanity check |
| `preview.png` | Quick 3-view preview |
| `analyze.py` | Section thickness probes |

## Key dimensions (mm)

- Overall: **45 × 34 × 26**
- Bottom / floor base: **8** (original ≈ 3–4)
- Walls: **5.5** (original ≈ 2–3)
- Base↔housing fillet: **5**
- Rail cavity: **16 W × 13 H × 32 deep**
- Solid roof above cavity: **≈ 5**
- Nose mounting hole: oval ≈ **5.2 × 4**

Rebuild after edits:

```bash
python skolestub.py
python analyze.py
```

If the rail is tight/loose, change `CAVITY_W` / `CAVITY_H` in `skolestub.py` by ±0.3 mm and rebuild.

## Why the last print failed (and what we changed)

| Weak spot | Original / failed print | This model |
|-----------|-------------------------|------------|
| Floor base | ~3–4 mm, snaps under load | **8 mm solid** |
| Walls | ~2–3 mm | **5.5 mm** |
| Sharp corners | Stress risers → cracks | **Large fillets everywhere** |
| Layer orientation | Often printed on the floor face | Print **on its side** |
| Material | Often brittle PLA | Prefer **PETG / ABS / ASA** |

## Print settings (critical)

1. **Orientation:** lay it on its **side** so the arch is continuous loops of filament (not stacked layers that can delaminate under chair weight).
2. **Material:** PETG, ABS, or ASA. Avoid brittle PLA for load-bearing feet.
3. **Walls:** 5–6 perimeters  
4. **Infill:** 40–100% gyroid/cubic  
5. **Layers:** 0.2 mm, 6 top/bottom  
6. **Optional:** slight ironing on the floor face after you rotate the part back… or sand the contact face flat.

## Stress gate (included)

`python skolestub.py` refuses to finish unless:

- size ≈ 45×34×26
- base ≥ 7 mm, walls ≥ 5 mm, fillets ≥ 4 mm, roof ≥ 2.5 mm
- watertight manifold volume
- estimated PETG compressive capacity ≫ 50 kg/leg design load (100 kg person / 4 legs × dynamic 2×)

This is a **geometry sanity check**, not full FEA — but it catches the thin-wall mistakes that shattered the last print.
