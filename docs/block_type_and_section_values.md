# Block Type and Section Values

## `block_type`

Classification of a block's content role. Set progressively through the shared pipeline — later steps fill nulls left by earlier ones.

| Value | Set by | Description |
|---|---|---|
| `page_label` | `pdf/step_08_page_label_detector`, `html/step_03_page_label_detector` | Running page number or label (e.g. "Page 1", "A-6") |
| `hr` | `html/step_02_box_cleaner` | Horizontal rule / divider element |
| `image` | `html/step_02_box_cleaner` | Image or figure block |
| `table` | `pdf/step_10_table_builder`, `html/step_04_line_builder` | Tabular data |
| `toc` | `shared/step_01_toc_detector` | Table of contents entry line |
| `toc_heading` | `shared/step_01_toc_detector` | Heading line of the TOC section (e.g. "Table of Contents") |
| `exhibits` | `shared/step_02_exhibit_detector` | Exhibit body content |
| `exhibit_heading` | `shared/step_02_exhibit_detector` | Heading line of an exhibit (e.g. "Exhibit A") |
| `navigation` | `shared/step_03_navigation_detector` | Repeated header/footer navigation text |
| `heading` | `shared/step_05_heading_detector` | Structural heading |
| `hybrid_heading_paragraph` | `shared/step_05_heading_detector` | Heading that also contains body text inline |
| `suppressed_repeated_heading` | `shared/step_05_heading_detector` | Heading suppressed because it repeats across pages |
| `paragraph` | `shared/step_05_heading_detector` | Default body text (fallback when no other type matches) |
| `vertical_text` | `pdf/step_08_cell_builder` | Text with BTT or TTB orientation (e.g. rotated sidebar labels, spine text) |
| `chart` | `pptx/step_05_paragraph_builder` | PPTX chart object (data points, axis labels, legends) |
| `shape` | `pptx/step_05_paragraph_builder` | PPTX non-text shape with associated text content |
| `speaker_notes` | `pptx/step_05_paragraph_builder` | PPTX speaker notes attached to a slide |
| `header` | `docx/step_04_paragraph_builder` | Page header text (top margin region); typically repeats across pages |
| `footer` | `docx/step_04_paragraph_builder` | Page footer text (bottom margin region); typically repeats across pages |
| `watermark` | _(reserved)_ | Watermark / background text overlay |
| `signature_block` | _(reserved)_ | Signature block at end of legal document |
Footnote
code
FormField
block_quote (picked up from pdf / html / docx) - can never be a heading


- TODO: Comment, Math, 
- A toc can be both in table as well as text form
- Maybe change navigation to bookmark
- We don't have ListItem because it's not mutually exclusive
**Example**
7.0 Documentation -> <li> that is heading
    7.1 Deviations: -> <li> that is heading
        7.1.1 Completed and electronically-signed Deviation Request Forms are maintained... -> <li> that is paragraph
- Same problem with caption: heading above a table, paragraph below a table (ambiguous)
- TOC can be a table or text, which we can distinguish by the presence of a table_id

**Noise types** (stripped before chunking): `hr`, `page_label`, `image`, `suppressed_repeated_heading`, `navigation`, `watermark`

**Heading types** (treated as hierarchy anchors): `heading`, `toc_heading`, `exhibit_heading`, `hybrid_heading_paragraph`

---

## `section`

Document-zone classification of a page. Set in `shared/step_04_section_classifier.py` in four passes.

| Value | Detection method | Description |
|---|---|---|
| `coverpage` | Blank page labels / TOC anchor / layout scoring / SEC heuristics | Leading cover page(s) before body content |
| `front_matter` | Roman page labels before body; blank pages before first TOC | Preface, foreword, executive summary |
| `body` | Longest contiguous run of arabic-numbered pages | Main document body |
| `back_matter` | Roman page labels after body; trailing blank-labeled pages | Appendices using roman numbering |
| `annex` | Alpha-prefixed page labels (e.g. "A-1", "B-3") | Annexes (general) |
| `financials` | Alpha-prefixed labels with prefix `F` (e.g. "F-1") | Financial statement annexes |
| `schedules` | Alpha-prefixed labels with prefix `S` (e.g. "S-1") | Schedule annexes |
| `last_page` | Blank label / low density / contact info signals | Standalone closing page (contact, disclaimer, colophon) |


| `toc` | Inherited from `block_type` (`toc`, `toc_heading`) | Pages containing the table of contents |
| `exhibits` | Inherited from `block_type` (`exhibits`, `exhibit_heading`) | Pages containing exhibit content |
