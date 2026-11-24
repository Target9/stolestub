# Realistic skolestub

Shape-matched to the real chair-foot glide (slipper / tunnel-on-flange), with a pocket sized so the chair foot slides in.

## Slice this

`skolestub_realistic.stl`

## Shape (matches photos)

- Flat **open end** with **U-shaped tunnel** for the chair rail/foot  
- Wider **floor flange** around the raised housing  
- **Slipper side profile** — tall at the opening, sloping to a rounded nose  
- Small **oval hole** in the closed nose tip  
- Soft fillets where the tunnel meets the flange  

## Fit

| | mm |
|--|--|
| Overall | 44 × 34 × 24 |
| Chair-foot cavity | **15.5 W × ~15 H × 28 deep** (open U at flat end) |
| Base thickness | 5 |
| Walls | ~4 |

If the foot is tight/loose, edit `CAVITY_W` / `CAVITY_H` in `skolestub_realistic.py` by ±0.3 and run:

```bash
python skolestub_realistic.py
```

## Print

- Orient **on its side** (arch continuous)  
- PETG / ABS / ASA  
- 4–6 walls, 30–60%+ gyroid, 0.2 mm layers  
