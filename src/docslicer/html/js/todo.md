### Possible TODO's for box_extractor.js ###

## 1. Better SEC page detection:

- Unify page detection around **repeating sibling patterns**, not generic div heuristics
- Prefer **direct children of a common parent** as page candidates
- Support two dominant layouts:
  - Batch siblings: `page, page, page…`
  - Interleaved: `page, pageBreak, page, pageBreak…`
- Explicitly treat `*PageBreak*` / `*BreakArea*` divs as **separators**, never as pages
- Enforce **minimum page dimensions** (e.g. width ≥ 300–400pt) to exclude small wrappers
- Use **consistent width clustering** to identify true page sequences
- Select the **highest DOM level** with the strongest repeating-page signal
- Avoid scanning full subtrees; operate on **direct children only**
- Run page numbering **once on the chosen container**, not per page
- Keep heuristic-based / CSS page-break detection only as a **fallback** for edge cases

## 2. Infer text alignment from geometry

- SEC may not always have "text-align:"
- Detect visual alignment (left / center / right) from absolute positioning  
- Use container width vs element x-position + text width
- Ignore `text-align` when elements are absolutely positioned
- Treat inline elements as non-alignable unless promoted to block/inline-block
- Add tolerance for PDF rounding errors (±2–3pt)
- Store:
  - `alignment_visual`
  - `alignment_confidence`
  - `alignment_method = "geometry" | "css"`

## 3. Detect “HR-like” borders (non-<hr>) incl. top + bottom

- Re-introduce deferred detection for separators that *look like* `<hr>` but are CSS borders
  - `border-bottom` **or** `border-top` on a block spanning (near) full page width

- Heuristics
  - Not a native `HR` tag
  - Border present (either side):
    - `borderBottomWidth > 0` AND style != `none` AND color != `transparent`
    - OR `borderTopWidth > 0` AND style != `none` AND color != `transparent`
  - Default exclude table cells (`TD`/`TH`) unless explicitly allowed
  - Width threshold: `rect.width >= 0.8 * pageWidth` (tune + tolerance)
  - Optional gating:
    - empty-ish text (preferred) OR token-like content (page label / footer patterns)
    - avoid catching normal underlined headings/links

- Performance / correctness
  - Use `textContent` for cheap gating
  - Only compute `innerText` if gating passes
  - Keep “deferred” (`pendingHrBorders`) so it can be resolved after page detection / layout grouping

- Output
  - Emit as `hr_row` with `hr_kind = "border_top" | "border_bottom" | "border_both"`
  - Preserve context: tableId / tableRowId ancestry if present


## 4. Text backround

function getEffectiveBackground(el) {
  let node = el;
  while (node) {
    const cs = getComputedStyle(node);
    const bg = cs.backgroundColor;
    if (bg && bg !== "rgba(0, 0, 0, 0)" && bg !== "transparent") {
      return bg;
    }
    node = node.parentElement;
  }
  return "transparent"; // or default white
}



## OPTIONAL: Increase iXBRL columns
- TBD if useful
- <ix:nonnumeric contextref="c-1" name="dei:DocumentType" id="f-1">10-Q</ix:nonnumeric>
- <ix:nonfraction unitref="usd" contextref="c-13" decimals="-5" name="us-gaap:CostOfGoodsAndServicesSold" format="ixt:num-dot-decimal" scale="6" id="f-61">3,008.3</ix:nonfraction>
