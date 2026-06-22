# Understanding `jump_ratio`, `is_bimodal`, and the old `gap_ratio`

Both KPIs are about one thing: **do the gaps on a line form one pile (all spaces)
or two piles (spaces + gutters)?**

The trick to *seeing* it: take the line's gaps, **sort them small→large**, and look
at the **step (×) from each gap to the next**.

- Inside a pile, steps are tiny (≈ ×1.0–1.1) — gaps are all about the same size.
- Between two piles there's one big step (a **valley** / empty band).

`jump_ratio` = the biggest step. `is_bimodal` = "a big-enough step exists."

---

## Line 1 — justified text (one pile → TEXT)

```
(b)   AGRICULTURE,   RURAL   DEVELOPMENT,   FOOD
   └0.69┘        └0.69┘   └0.69┘        └0.69┘      (gaps in em)
```

Sorted gaps and the step to the next one:

```
  em    step
 0.69  ┐
 0.69  │  ← all four gaps identical
 0.69  │     steps are ×1.00
 0.69  ┘
```

No big step anywhere → **one pile** → not bimodal.
`jump_ratio = 1.00`, `is_bimodal = False` → fall back to content → **TEXT**.

---

## Line 2 — table header (two piles → TABLE)

```
Restructuring   Intangible Asset Amortisation   Other   Core
            └0.47┘         └0.25┘      └0.25┘ └0.47┘ └0.47┘
             gutter        space       space  gutter  gutter
```

Sorted gaps and steps:

```
  em    step
 0.25  ┐
 0.25  │  space pile   (steps ×1.00)
 0.25  ┘
   ╳   ←── VALLEY: step ×1.88  ← this is jump_ratio
 0.47  ┐
 0.47  │  gutter pile  (steps ×1.00)
 0.47  ┘
```

One clear valley (×1.88) with the upper pile gutter-sized (0.47 > 0.30 em)
→ `jump_ratio = 1.88`, `is_bimodal = True` → **TABLE**.

---

## Line 3 — your real line (THREE piles, still works)

```
For the quarter ended 30 September | Reported | Restructuring | Intangible Asset
  └0.17 each (5 spaces)┘      └4.01┘     └0.56┘        └0.48┘    └0.17┘
        spaces                 break     gutter        gutter    space
```

Sorted gaps and steps:

```
  em    step
 0.17  ┐
 0.17  │
 0.17  │  space pile  (steps ×1.0)
 0.18  │
 0.18  ┘
   ╳   ←── lowest valley: step ×2.7   ← split happens here
 0.48  ┐  gutter pile
 0.56  ┘
   ╳   ←── another valley: step ×7.2  (a wider "section break" tier)
 4.01     break pile
```

There are *two* valleys because gutters come in two sizes (column gutters ~0.5 em,
a big section break ~4 em). We take the **lowest** valley (×2.7) — everything above
it is "not a space," so all three of {0.48, 0.56, 4.01} become cell boundaries and
only the 0.17 spaces merge. `jump_ratio` reports the biggest step (7.2), but
`is_bimodal` only needs *a* qualifying valley to exist.

---

## Where does 1.8 come from?

It's a **chosen cut-off**, not a derived constant — sitting in the empty zone
between "normal space jitter" and "a real category change."

Observed steps from the examples above:

```
 ×1.00   ← uniform spaces (Line 1)
 ×1.00   ← within either pile
 ×1.06   ← natural space jitter (your real line, 0.17→0.18)
 ───────────────────── 1.8 cut-off lives in this gap ─────────────────────
 ×1.88   ← space → gutter (Line 2)
 ×2.70   ← space → gutter (your real line)
 ×7.16   ← gutter → section break
```

Within-line spaces barely vary (×1.0–1.3 — same space glyph, only justification
nudges them). A true gutter is a different *kind* of whitespace and lands ≥ ×1.8
away. So **1.8** says: "a step this big is a category change, not space jitter."
Lower it toward 1.5 to catch very tight tables (risks splitting loosely-justified
prose); raise it toward 2.0 to be conservative.

---

## How this differs from the old `gap_ratio = max / median`

`gap_ratio` compared the biggest gap to the *median* gap. It breaks when gutters
outnumber spaces, because the median then lands **on a gutter**:

```
Line: 2 spaces + 3 gutters → sorted: 0.25  0.25  [0.47] 0.47  0.47
                                                    ↑ median sits on a gutter
   gap_ratio = max/median = 0.47 / 0.47 = 1.00   ← looks "uniform" — WRONG
   jump_ratio (biggest step)            = 1.88   ← still sees the valley — RIGHT
```

| line | `gap_ratio` (old) | `jump_ratio` (new) |
|------|-------------------|--------------------|
| uniform text | 1.00 | 1.00 |
| header 4 spaces + 4 gutters | 1.31 | 1.88 |
| header 2 spaces + 3 gutters | **1.00 (fooled)** | 1.88 |
| your real line | 22.28 (outlier-inflated) | 7.16 |

So: when a line is truly uniform, **both read ~1.0**. But `gap_ratio ≈ 1` does *not*
guarantee uniform (row 3 proves it), which is why `jump_ratio` — the step between
*sorted neighbours* — replaced it.
```
