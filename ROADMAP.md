## PDF Extraction

### Known limitations
- Fix heading padding detection and merged cell handling
- Handle non-LTR text in table cells instead of stripping it
- Improve italic and slanted text detection for OCR pipelines

## PPTX Extraction

### Layout
- Text block layout is unreliable for infographic-heavy and image-dominant slides with the current line-based approach; investigate spatial clustering of text blocks as an alternative

### Shapes
- Extract XML SmartArt (org charts, process diagrams) as structured output (exploratory)

## HTML Parsing

### Page reconstruction
- Improve SEC HTML page boundary detection for repeating sibling-based layouts
- Better support interleaved page/page-break structures
- Reduce false page detection from wrapper divs and separators

### Layout understanding
- Infer visual text alignment from geometry when CSS alignment is unreliable
- Improve separator detection for border-based horizontal rules
- Add more reliable background-color inheritance detection

### Structured financial markup
- Evaluate whether additional iXBRL attributes should be surfaced in extracted output

## Heading Hierarchy

### Named headings
- Add heading detection for other languages

## Chart Data Extraction

### DOCX chart points
- Extend chart point extraction to DOCX using the shared DrawingML `c:` namespace
- Chart XML format is identical to PPTX; only the part-discovery traversal differs
- Share core series/cache parsing logic across PPTX and DOCX orchestrators

### PDF chart points (VLM-assisted)
- PDF charts are rendered as vector paths or rasterized images with no cached data
- VLM extraction (structured output from chart screenshot) is unavoidable for reliable results
- Deliberate exception to the no-LLM design principle; scope and cost should be opt-in

## Content Types

### Inline styling
- Detect subscript and superscript runs
- Detect strikethrough

### Rich content
- Detect and classify mathematical formulas
- Detect checkboxes and their checked/unchecked state

### Image narration (VLM-assisted)
- Populate the `text` field for image blocks using a VLM (opt-in)
- Deliberate exception to the no-LLM design principle; scope and cost should be opt-in

## Developer Experience

## Integrations
- Output connectors for vector stores and databases (e.g. PostgreSQL, chunking pipeline handshakes)

## Additional Input Formats
- Plain text (`.txt`) — line-based structure inference, heading and list detection
- Legacy Word (`.doc`, `.ppt`) — via format conversion or direct binary parsing
- Other formats (`.rtf`, `.odt`, `.epub`, …) — evaluated on demand based on use case
