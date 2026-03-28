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