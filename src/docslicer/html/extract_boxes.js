// extract_boxes.js
// Clean TreeWalker-based approach for extracting text boxes from HTML
// - isPageLabelToken is injected from YAML by step_01_box_extractor.py

(() => {
  // =========================
  // TAG DEFINITIONS
  // =========================
  
  // BUCKET 1: STRUCTURE TAGS
  // These are block-level elements that define text structure
  // We extract atomic structure elements (no structure descendants with text)
  const STRUCTURE_TAGS = new Set([
    "P", "DIV", "H1", "H2", "H3", "H4", "H5", "H6",
    "LI", "TD", "TH", "BLOCKQUOTE", "PRE", "ARTICLE",
    "SECTION", "HEADER", "FOOTER", "ASIDE", "MAIN",
    "FIGCAPTION", "CAPTION", "DT", "DD", "ADDRESS"
  ]);
  
  // BUCKET 2: INLINE SPLIT TAGS
  // These inline tags cause a new box to be created (split boundaries)
  // Examples: styling changes, links, line breaks
  const INLINE_SPLIT_TAGS = new Set([
    "STRONG", "B", "EM", "I", "U", "MARK",
    "A", "CODE", "KBD", "SAMP", "VAR",
    "DEL", "INS", "S", "STRIKE", "SMALL", "BIG",
    "SPAN", "FONT", "TT", "LABEL",
    "SUP", "SUB",  // split so script_type (superscript/subscript) can be tagged per box
    "BR"  // BR always creates a split
  ]);

  // BUCKET 3: INLINE TRANSPARENT TAGS
  // These inline tags do NOT cause splits (transparent to box creation)
  // Text flows through these as if they weren't there
  const INLINE_TRANSPARENT_TAGS = new Set([
    "WBR", "ABBR", "TIME",
    "Q", "CITE", "DFN", "BDI", "BDO", "NOBR"
  ]);

  // NOTE: PRE and CODE handled separately

  // =========================
  // UTILITIES
  // =========================
  
  const stripZeroWidth = (s) => (s || "").replace(/[\u200B\u200C\u200D\uFEFF]/g, "");
  const normalize = (s) => stripZeroWidth(s).replace(/\s+/g, " ").trim();
  
  const _csCache = new WeakMap();
  // Document-order unique id per element. Populated once in extractAll and read
  // by extractHierarchy so struct_ancestor_ids can reference the *same* ancestor
  // instance consistently across every box beneath it.
  const _elemUid = new WeakMap();
  const getCS = (el) => {
    if (!el) return null;
    let cs = _csCache.get(el);
    if (!cs) {
      cs = window.getComputedStyle(el);
      _csCache.set(el, cs);
    }
    return cs;
  };

  // =========================
  // ATOMIC STRUCTURE DETECTION
  // =========================
  
  // Find all atomic structure elements (structure tags with no structure descendants)
  // Uses ancestor-walk instead of querySelectorAll("*") per element to avoid O(n²).
  const findAtomicStructures = (root) => {
    const allCandidates = root.querySelectorAll(Array.from(STRUCTURE_TAGS).join(","));

    // First pass: filter to visible elements with text, build a Set for O(1) lookup
    const visible = [];
    for (const el of allCandidates) {
      const cs = getCS(el);
      if (!cs || cs.display === "none" || cs.visibility === "hidden") continue;
      if (!normalize(el.textContent || "")) continue;
      visible.push(el);
    }
    const visibleSet = new Set(visible);

    // Second pass: mark non-atomic via ancestor walk.
    // If an ancestor of el is also in visibleSet, that ancestor is non-atomic
    // (it has a structure descendant with text).
    const nonAtomic = new Set();
    for (const el of visible) {
      let parent = el.parentElement;
      while (parent && parent !== root) {
        if (visibleSet.has(parent)) {
          nonAtomic.add(parent);
        }
        parent = parent.parentElement;
      }
    }

    return visible.filter(el => !nonAtomic.has(el));
  };

  // =========================
  // TEXT ORIENTATION DETECTION
  // =========================
  
  const getTextOrientation = (el) => {
    const cs = getCS(el);
    if (!cs) return "LTR";
    
    const writingMode = cs.writingMode || cs.WebkitWritingMode || "";
    const direction = cs.direction || "ltr";
    
    // Check writing mode first (vertical text)
    if (writingMode.includes("vertical-rl") || writingMode.includes("tb-rl")) {
      return "TTB"; // Top-to-bottom (right-to-left columns)
    }
    if (writingMode.includes("vertical-lr") || writingMode.includes("tb-lr")) {
      return "TTB"; // Top-to-bottom (left-to-right columns)
    }
    if (writingMode.includes("sideways")) {
      return "TTB";
    }
    
    // Horizontal text - check direction
    if (direction === "rtl") {
      return "RTL"; // Right-to-left
    }
    
    return "LTR"; // Left-to-right (default)
  };

  // =========================
  // STYLE EXTRACTION
  // =========================
  
  const isBold = (cs) => {
    if (!cs) return false;
    const fw = cs.fontWeight;
    const n = parseInt(fw, 10);
    if (!isNaN(n)) return n >= 600;
    const s = String(fw).toLowerCase();
    return s === "bold" || s === "bolder";
  };
  
  const isItalic = (cs) => {
    if (!cs) return false;
    return cs.fontStyle === "italic" || cs.fontStyle === "oblique";
  };
  
  const isUnderlined = (cs, tagName) => {
    if (!cs) return false;
    const tdLine = (cs.textDecorationLine || "").toLowerCase();
    const td = (cs.textDecoration || "").toLowerCase();
    if (tdLine.includes("underline")) return true;
    if (td.includes("underline")) return true;
    if ((tagName || "").toUpperCase() === "U") return true;
    return false;
  };
  
  const rgbToHex = (rgb) => {
    if (!rgb) return "";
    // Handle hex already
    if (rgb.startsWith("#")) return rgb;
    // Handle rgb() or rgba()
    const match = rgb.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)/);
    if (!match) return rgb;
    const r = parseInt(match[1]);
    const g = parseInt(match[2]);
    const b = parseInt(match[3]);
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
  };
  
  const extractPrimaryFont = (fontFamilyStack) => {
    if (!fontFamilyStack) return "";
    
    // Split by comma to get individual fonts
    const fonts = fontFamilyStack.split(",");
    if (fonts.length === 0) return "";
    
    // Take the first font (the primary one)
    let primaryFont = fonts[0].trim();
    
    // Remove quotes if present
    primaryFont = primaryFont.replace(/^["']|["']$/g, "");
    
    return primaryFont;
  };
  
  const extractStyles = (el, tagName) => {
    const cs = getCS(el);
    if (!cs) {
      return {
        font_size: "",
        font_family: "",
        font_weight: 400,
        bold_ratio: 0,
        italic_ratio: 0,
        underlined_ratio: 0,
        non_stroking_color: "",
        stroking_color: "",
        text_align: ""
      };
    }
    
    // Font properties
    const fontSize = cs.fontSize || "";
    const fontFamily = extractPrimaryFont(cs.fontFamily || "");
    const fontWeight = parseInt(cs.fontWeight, 10) || 400;
    
    // Style ratios (1 or 0 at this granular level)
    const boldRatio = isBold(cs) ? 1.0 : 0.0;
    const italicRatio = isItalic(cs) ? 1.0 : 0.0;
    const underlinedRatio = isUnderlined(cs, tagName) ? 1.0 : 0.0;
    
    // Colors
    const nonStrokingColor = rgbToHex(cs.color || "");
    
    // Stroking color (text-stroke or -webkit-text-stroke)
    let strokingColor = "";
    const textStroke = cs.webkitTextStrokeColor || cs.textStrokeColor || "";
    const textStrokeWidth = cs.webkitTextStrokeWidth || cs.textStrokeWidth || "0px";
    if (textStroke && parseFloat(textStrokeWidth) > 0) {
      strokingColor = rgbToHex(textStroke);
    }
    
    return {
      font_size: fontSize,
      font_family: fontFamily,
      font_weight: fontWeight,
      bold_ratio: boldRatio,
      italic_ratio: italicRatio,
      underlined_ratio: underlinedRatio,
      non_stroking_color: nonStrokingColor,
      stroking_color: strokingColor
    };
  };

  // =========================
  // HELPER: FIND LINK URL
  // =========================
  
  // Helper to find the closest anchor tag and extract href
  const findLinkUrl = (node) => {
    let parent = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
    while (parent) {
      if (parent.tagName && parent.tagName.toUpperCase() === "A") {
        // Get the raw href attribute
        const rawHref = parent.getAttribute("href") || "";
        
        if (!rawHref) return "";
        
        // Hash-only anchors: keep as-is
        if (rawHref.startsWith("#")) {
          return rawHref;
        }
        
        // Special protocols: keep as-is
        const specialProtocols = ["javascript:", "mailto:", "tel:", "sms:", "data:"];
        for (const protocol of specialProtocols) {
          if (rawHref.toLowerCase().startsWith(protocol)) {
            return rawHref;
          }
        }
        
        // For everything else, use the resolved href
        // This handles relative URLs, absolute URLs, etc.
        const resolvedHref = parent.href || rawHref;
        
        // Don't return about:blank URLs (these are typically empty hrefs)
        if (resolvedHref.startsWith("about:blank")) {
          return rawHref;
        }
        
        return resolvedHref;
      }
      parent = parent.parentElement;
    }
    return "";
  };

  // =========================
  // HELPER: FIND IXBRL ID
  // =========================
  
  // Helper to find iXBRL element ID from immediate parent only
  const findIxbrlId = (node) => {
    // Only check the immediate parent element
    const parent = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
    if (!parent) return "";
    
    // Try different ways to get the tag name (handles namespace issues)
    const tagName = parent.tagName || parent.nodeName || "";
    const localName = parent.localName || "";
    const tag = tagName.toUpperCase();
    const local = localName.toUpperCase();
    
    // Check if it's an iXBRL element (ix:* or IX:*)
    const isIxbrl = tag.startsWith("IX:") || tag.startsWith("IX-") || 
                    local.startsWith("IX:") || local.startsWith("IX-") ||
                    tag.includes(":IX") || local.includes(":IX");
    
    if (isIxbrl) {
      // Try multiple ways to get the ID
      const id = parent.id || parent.getAttribute("id") || "";
      if (id) {
        return String(id).trim();
      }
    }
    
    return "";
  };

  // =========================
  // HELPER: PAGE DETECTION
  // =========================
  
  // Standard page sizes in points (72pt = 1 inch)
  const PAGE_FORMATS = {
    "US Letter": { width: 612, height: 792, tolerance: 5 },
    "US Legal": { width: 612, height: 1008, tolerance: 5 },
    "A4": { width: 595.3, height: 841.9, tolerance: 5 }
  };
  
  // Helper to parse dimension string (e.g., "612pt", "8.5in", "100px")
  const parseDimension = (dim) => {
    if (!dim) return null;
    const match = dim.match(/^([\d.]+)(pt|px|in|cm|mm)?$/);
    if (!match) return null;
    
    const value = parseFloat(match[1]);
    const unit = match[2] || 'px';
    
    // Convert to points
    switch (unit) {
      case 'pt': return value;
      case 'px': return value * 0.75; // 1px = 0.75pt (96 DPI)
      case 'in': return value * 72;
      case 'cm': return value * 28.35;
      case 'mm': return value * 2.835;
      default: return value;
    }
  };
  
  // Helper to detect page format from dimensions
  const detectPageFormat = (width, height) => {
    if (!width || !height) return "";
    
    for (const [format, dims] of Object.entries(PAGE_FORMATS)) {
      const widthMatch = Math.abs(width - dims.width) <= dims.tolerance;
      const heightMatch = Math.abs(height - dims.height) <= dims.tolerance;
      
      // Check both orientations
      if ((widthMatch && heightMatch) || 
          (Math.abs(width - dims.height) <= dims.tolerance && 
           Math.abs(height - dims.width) <= dims.tolerance)) {
        return format;
      }
    }
    
    return "";
  };
  
  // Helper to check if element has page break CSS
  const hasPageBreak = (el) => {
    const cs = getCS(el);
    if (!cs) return false;
    
    const pageBreakAfter = cs.pageBreakAfter || cs.breakAfter || "";
    const pageBreakBefore = cs.pageBreakBefore || cs.breakBefore || "";
    
    return pageBreakAfter.includes("always") || 
           pageBreakAfter.includes("page") ||
           pageBreakBefore.includes("always") ||
           pageBreakBefore.includes("page");
  };
  
  // Helper to check if element is a page container
  const isPageContainer = (el) => {
    // Check inline style first
    const style = el.style;
    let width = parseDimension(style.width);
    let height = parseDimension(style.height);
    
    // Check computed style if not in inline
    if (!width || !height) {
      const cs = getCS(el);
      if (cs) {
        width = width || parseDimension(cs.width);
        height = height || parseDimension(cs.height);
      }
    }
    
    // Check if dimensions match standard page formats
    if (width && height) {
      const format = detectPageFormat(width, height);
      if (format) {
        return { isPage: true, width, height, format };
      }
    }
    
    // Check for page-break CSS
    if (hasPageBreak(el)) {
      return { isPage: true, width: width || NaN, height: height || NaN, format: "" };
    }
    
    return { isPage: false, width: null, height: null, format: "" };
  };
  
  // WeakMaps to store page information
  const pageNumbers = new WeakMap();
  const pageInfo = new WeakMap();
  let pageCounter = 1;
  
  // Scan document and assign page numbers
  const assignPageNumbers = (root) => {
    const allDivs = root.querySelectorAll("div");
    
    for (const div of allDivs) {
      const pageCheck = isPageContainer(div);
      if (pageCheck.isPage) {
        pageNumbers.set(div, pageCounter++);
        pageInfo.set(div, {
          width: pageCheck.width,
          height: pageCheck.height,
          format: pageCheck.format
        });
      }
    }
  };
  
  // Helper to find page context
  const findPageContext = (node) => {
    let parent = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
    
    while (parent) {
      if (pageNumbers.has(parent)) {
        const pageNumber = pageNumbers.get(parent);
        const info = pageInfo.get(parent);
        return {
          page_number: pageNumber,
          page_width: info.width,
          page_height: info.height,
          page_format: info.format
        };
      }
      parent = parent.parentElement;
    }
    
    // Default: not in a page container
    return {
      page_number: 1,
      page_width: NaN,
      page_height: NaN,
      page_format: ""
    };
  };

  // =========================
  // HELPER: TABLE TRACKING
  // =========================
  
  // WeakMaps to store assigned IDs for tables and rows
  const tableIds = new WeakMap();
  const rowIds = new WeakMap();
  const rowCellCounts = new WeakMap();
  const cellIndexes = new WeakMap();
  let tableCounter = 1;
  let rowCounter = 1;

  const isTableCellTag = (tag) => tag === "TD" || tag === "TH";

  const ensureRowCellInfo = (rowEl) => {
    if (rowCellCounts.has(rowEl)) return;

    let index = 0;
    for (const child of rowEl.children) {
      const tag = child.tagName ? child.tagName.toUpperCase() : "";
      if (!isTableCellTag(tag)) continue;
      cellIndexes.set(child, index++);
    }
    rowCellCounts.set(rowEl, index);
  };
  
  // Helper to find table and row IDs
  // Only the outermost <table> (not itself inside a <td>/<th>) is treated as a
  // real table. Nested tables used for layout inside a cell are skipped — their
  // content is attributed to the enclosing real table/row/cell instead.
  const findTableContext = (node) => {
    let parent = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
    let tableId = NaN;
    let tableRowId = NaN;
    let tableCellIndex = NaN;
    let tableRowCellCount = NaN;

    // Accumulate row/cell context at the current nesting level.
    // Reset when we pass through a nested <table> boundary.
    let pendingCell = null;
    let pendingRow = null;

    while (parent) {
      const tag = parent.tagName ? parent.tagName.toUpperCase() : "";

      if (!pendingCell && isTableCellTag(tag)) {
        pendingCell = parent;
      }

      if (!pendingRow && tag === "TR") {
        pendingRow = parent;
      }

      if (tag === "TABLE") {
        if (parent.closest("td, th")) {
          // Nested layout table — skip it and reset pending context for outer level
          pendingCell = null;
          pendingRow = null;
        } else {
          // Outermost real table — commit context
          if (!tableIds.has(parent)) {
            tableIds.set(parent, tableCounter++);
          }
          tableId = tableIds.get(parent);

          if (pendingRow) {
            if (!rowIds.has(pendingRow)) {
              rowIds.set(pendingRow, rowCounter++);
            }
            tableRowId = rowIds.get(pendingRow);
            ensureRowCellInfo(pendingRow);
            tableRowCellCount = rowCellCounts.get(pendingRow);
            if (pendingCell && cellIndexes.has(pendingCell)) {
              tableCellIndex = cellIndexes.get(pendingCell);
            }
          }
          break;
        }
      }

      parent = parent.parentElement;
    }

    return { tableId, tableRowId, tableCellIndex, tableRowCellCount };
  };

  // =========================
  // HELPER: HIERARCHY EXTRACTION
  // =========================
  
  // Group tags that define semantic sections
  const ANCESTOR_TAGS = new Set([
    "SECTION", "ARTICLE", "MAIN", "HEADER", "NAV", "ASIDE", "DIV", "FOOTER"
  ]);
  
  // Extract DOM metadata from an element
  const extractDomMetadata = (el) => {
    const domId = el.id || "";
    const domClass = el.className || "";
    
    // Extract all data-* attributes
    const dataAttrs = {};
    if (el.dataset) {
      // Use the dataset API which automatically converts data-foo-bar to fooBar
      for (const key in el.dataset) {
        dataAttrs[key] = el.dataset[key];
      }
    } else if (el.attributes) {
      // Fallback for older browsers - manually parse data-* attributes
      for (let i = 0; i < el.attributes.length; i++) {
        const attr = el.attributes[i];
        if (attr.name.startsWith('data-')) {
          // Remove 'data-' prefix and convert to camelCase
          const key = attr.name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
          dataAttrs[key] = attr.value;
        }
      }
    }
    
    return { domId, domClass, dataAttrs };
  };
  
  // Extract hierarchy information for a node
  const extractHierarchy = (node) => {
    const ancestorIds = [];
    const ancestorClasses = [];
    const ancestorTags = [];
    const ancestorTagIds = [];
    const ancestorAriaRoles = [];

    // Start at the element itself so struct_ancestors INCLUDES the box's own tag
    // as the last entry, then walk the full chain up to the document root (includes
    // body and html). The document node has no tagName and is skipped by `if (tag)`.
    let parent = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;

    while (parent) {
      const tag = parent.tagName ? parent.tagName.toUpperCase() : "";

      // Collect IDs
      if (parent.id) {
        ancestorIds.push(parent.id);
      }

      // Collect classes
      if (parent.className && typeof parent.className === 'string') {
        const classes = parent.className.trim();
        if (classes) {
          ancestorClasses.push(classes);
        }
      }

      // Collect ALL ancestor tags. NOTE: the ANCESTOR_TAGS whitelist gate is
      // temporarily disabled so we can inspect the full, unfiltered ancestry.
      // struct_ancestor_ids / struct_ancestors / ancestor_aria_roles stay index-parallel.
      if (tag) {
        ancestorTags.push(tag.toLowerCase());
        ancestorTagIds.push(_elemUid.has(parent) ? _elemUid.get(parent) : -1);
        // aria roles stay sparse (like ancestor_ids / ancestor_classes) — only real roles
        const role = parent.getAttribute("role") || "";
        if (role) ancestorAriaRoles.push(role);
      }

      parent = parent.parentElement;
    }

    // Reverse arrays so they go from highest (body) to lowest (box)
    return {
      ancestor_ids: ancestorIds.reverse(),
      ancestor_classes: ancestorClasses.reverse(),
      struct_ancestors: ancestorTags.reverse(),
      struct_ancestor_ids: ancestorTagIds.reverse(),
      ancestor_aria_roles: ancestorAriaRoles.reverse()
    };
  };

  // =========================
  // HR (HORIZONTAL RULE) EXTRACTION
  // =========================
  
  const extractHrBox = (hr, boxIdObj) => {
    const cs = getCS(hr);
    if (!cs || cs.display === "none" || cs.visibility === "hidden") return null;
    
    const rect = hr.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    
    // Extract width from attributes or computed style
    let widthText = "";
    
    // Check width attribute first
    const widthAttr = hr.getAttribute("width");
    if (widthAttr) {
      widthText = widthAttr;
      // Add % if it's a number without unit
      if (/^\d+$/.test(widthAttr)) {
        widthText = widthAttr + "%";
      }
    } else if (cs.width) {
      // Check computed style width
      const width = cs.width;
      // Convert px to percentage if parent width is available
      if (width.endsWith("px")) {
        const parentEl = hr.parentElement;
        if (parentEl) {
          const parentCs = getCS(parentEl);
          if (parentCs && parentCs.width) {
            const parentWidth = parseFloat(parentCs.width);
            const hrWidth = parseFloat(width);
            if (parentWidth > 0) {
              const percentage = Math.round((hrWidth / parentWidth) * 100);
              widthText = percentage + "%";
            }
          }
        }
      } else if (width.endsWith("%")) {
        widthText = width;
      } else if (width !== "auto") {
        widthText = width;
      }
    }
    
    // Format text based on width
    const text = widthText ? `[[HR: ${widthText}]]` : "[[HR]]";
    
    // Extract DOM metadata
    const domMeta = extractDomMetadata(hr);
    
    // Extract hierarchy
    const hierarchy = extractHierarchy(hr);
    
    // Extract iXBRL ID
    const ixbrlId = findIxbrlId(hr);
    
    // Extract table context
    const tableContext = findTableContext(hr);
    
    // Extract page context
    const pageContext = findPageContext(hr);
    
    // Find parent structure tag for context
    let structEl = hr.parentElement;
    let structTag = "div"; // default
    while (structEl) {
      const tag = structEl.tagName.toUpperCase();
      if (STRUCTURE_TAGS.has(tag)) {
        structTag = tag.toLowerCase();
        break;
      }
      structEl = structEl.parentElement;
    }
    
    return {
      box_id: boxIdObj.value++,
      struct_tag: "hr",
      wrapping_tag: "hr",
      split_reason: "horizontal_rule",
      struct_tag_id: -1, // Will be set later if needed
      text: text,
      x_left: rect.left,
      x_right: rect.right,
      y_top: rect.top,
      y_bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
      text_orientation: "LTR",
      font_size: "",
      font_family: "",
      font_weight: 400,
      bold_ratio: 0,
      italic_ratio: 0,
      underlined_ratio: 0,
      strikethrough_ratio: 0,
      is_strikethrough: false,
      script_type: "",
      non_stroking_color: "",
      stroking_color: "",
      text_align: "",
      link_url: "",
      img_alt: "",
      img_src: "",
      dom_id: domMeta.domId,
      dom_class: domMeta.domClass,
      html_data_attrs: domMeta.dataAttrs,
      ixbrl_id: ixbrlId,
      table_id: tableContext.tableId,
      table_row_id: tableContext.tableRowId,
      table_cell_index: tableContext.tableCellIndex,
      table_row_cell_count: tableContext.tableRowCellCount,
      page_number: pageContext.page_number,
      page_width: pageContext.page_width,
      page_height: pageContext.page_height,
      page_format: pageContext.page_format,
      ancestor_ids: hierarchy.ancestor_ids,
      ancestor_classes: hierarchy.ancestor_classes,
      struct_ancestors: hierarchy.struct_ancestors,
      struct_ancestor_ids: hierarchy.struct_ancestor_ids,
      ancestor_aria_roles: hierarchy.ancestor_aria_roles
    };
  };

  // =========================
  // IMAGE EXTRACTION
  // =========================
  
  const extractImageBox = (img, boxIdObj) => {
    const cs = getCS(img);
    if (!cs || cs.display === "none" || cs.visibility === "hidden") return null;
    
    const rect = img.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    
    const alt = img.alt || "";
    const src = img.src || img.getAttribute("src") || "";
    const text = alt ? `[[IMAGE: ${alt}]]` : "[[IMAGE]]";
    
    // Extract DOM metadata
    const domMeta = extractDomMetadata(img);
    
    // Extract hierarchy
    const hierarchy = extractHierarchy(img);
    
    // Extract iXBRL ID
    const ixbrlId = findIxbrlId(img);
    
    // Extract table context
    const tableContext = findTableContext(img);
    
    // Extract page context
    const pageContext = findPageContext(img);
    
    // Find parent structure tag for context
    let structEl = img.parentElement;
    let structTag = "div"; // default
    while (structEl) {
      const tag = structEl.tagName.toUpperCase();
      if (STRUCTURE_TAGS.has(tag)) {
        structTag = tag.toLowerCase();
        break;
      }
      structEl = structEl.parentElement;
    }
    
    return {
      box_id: boxIdObj.value++,
      struct_tag: "img",
      wrapping_tag: "img",
      split_reason: "image",
      struct_tag_id: -1, // Will be set later if needed
      text: text,
      x_left: rect.left,
      x_right: rect.right,
      y_top: rect.top,
      y_bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
      text_orientation: "LTR",
      font_size: "",
      font_family: "",
      font_weight: 400,
      bold_ratio: 0,
      italic_ratio: 0,
      underlined_ratio: 0,
      strikethrough_ratio: 0,
      is_strikethrough: false,
      script_type: "",
      non_stroking_color: "",
      stroking_color: "",
      text_align: "",
      link_url: "",
      img_alt: alt,
      img_src: src,
      dom_id: domMeta.domId,
      dom_class: domMeta.domClass,
      html_data_attrs: domMeta.dataAttrs,
      ixbrl_id: ixbrlId,
      table_id: tableContext.tableId,
      table_row_id: tableContext.tableRowId,
      table_cell_index: tableContext.tableCellIndex,
      table_row_cell_count: tableContext.tableRowCellCount,
      page_number: pageContext.page_number,
      page_width: pageContext.page_width,
      page_height: pageContext.page_height,
      page_format: pageContext.page_format,
      ancestor_ids: hierarchy.ancestor_ids,
      ancestor_classes: hierarchy.ancestor_classes,
      struct_ancestors: hierarchy.struct_ancestors,
      struct_ancestor_ids: hierarchy.struct_ancestor_ids,
      ancestor_aria_roles: hierarchy.ancestor_aria_roles
    };
  };

  // =========================
  // BOX EXTRACTION FROM STRUCTURE ELEMENT
  // =========================
  
  const extractBoxesFromStructure = (structEl, structureTagId, boxIdObj) => {
    const boxes = [];
    const structTag = structEl.tagName.toUpperCase();
    const range = document.createRange();
    const textOrientation = getTextOrientation(structEl);
    
    // Detect list marker for LI elements
    let listMarker = "";
    let listMarkerAdded = false;
    if (structTag === "LI") {
      const parentList = structEl.parentElement;
      if (parentList) {
        const parentTag = parentList.tagName.toUpperCase();
        if (parentTag === "UL") {
          // Unordered list - use bullet
          listMarker = "• ";
        } else if (parentTag === "OL") {
          // Ordered list - find the index
          const listItems = Array.from(parentList.children).filter(
            child => child.tagName.toUpperCase() === "LI"
          );
          const index = listItems.indexOf(structEl);
          if (index >= 0) {
            // Get start attribute if present (defaults to 1)
            const start = parseInt(parentList.getAttribute("start") || "1", 10);
            listMarker = `${start + index}. `;
          }
        }
      }
    }
    
    // Extract text-align from structure element once (it's a block-level property)
    const structCs = getCS(structEl);
    let structTextAlign = "";
    if (structCs) {
      structTextAlign = structCs.textAlign || "";
      // Remove vendor prefixes
      structTextAlign = structTextAlign.replace(/^-webkit-|-moz-|-ms-|-o-/i, "");
      // Normalize logical values
      const direction = (structCs.direction || "ltr").toLowerCase();
      if (structTextAlign === "start") {
        structTextAlign = direction === "rtl" ? "right" : "left";
      } else if (structTextAlign === "end") {
        structTextAlign = direction === "rtl" ? "left" : "right";
      } else if (structTextAlign === "justify") {
        structTextAlign = "justified";
      }
    }
    
    // Invariant per structure element — compute once, reuse for every flushed box
    const hierarchy = extractHierarchy(structEl);
    const tableContext = findTableContext(structEl);
    const pageContext = findPageContext(structEl);

    // Track current inline context
    let currentText = "";
    let currentInlineTag = structTag; // Start with structure tag
    let currentInlineElement = structEl; // Track element for styling
    let currentLinkUrl = ""; // Track current link URL
    let currentIxbrlId = ""; // Track current iXBRL ID
    let splitReason = "new_structure";
    let rangeStart = null;
    
    // Helper to find the closest inline split tag ancestor
    const findInlineSplitParent = (node) => {
      let parent = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
      while (parent && parent !== structEl) {
        const tag = parent.tagName.toUpperCase();
        if (INLINE_SPLIT_TAGS.has(tag)) {
          return { tag, element: parent };
        }
        parent = parent.parentElement;
      }
      return { tag: structTag, element: structEl };
    };

    // Walk from an inline element up to structEl (inclusive) detecting decorations
    // that apply over descendant text via tag semantics or CSS (which is painted
    // over descendants, so getComputedStyle on the child alone would miss it).
    const detectStrikethrough = (startEl) => {
      let n = startEl;
      while (n) {
        const t = n.tagName ? n.tagName.toUpperCase() : "";
        if (t === "S" || t === "STRIKE" || t === "DEL") return true;
        const cs = getCS(n);
        if (cs) {
          const td = (cs.textDecorationLine || cs.textDecoration || "").toLowerCase();
          if (td.includes("line-through")) return true;
        }
        if (n === structEl) break;
        n = n.parentElement;
      }
      return false;
    };
    const detectScriptType = (startEl) => {
      let n = startEl;
      while (n) {
        const t = n.tagName ? n.tagName.toUpperCase() : "";
        if (t === "SUP") return "superscript";
        if (t === "SUB") return "subscript";
        const cs = getCS(n);
        const va = cs ? (cs.verticalAlign || "").toLowerCase() : "";
        if (va === "super") return "superscript";
        if (va === "sub") return "subscript";
        if (n === structEl) break;
        n = n.parentElement;
      }
      return "";
    };
    
    // =========================
    // PRE / CODE BLOCKS
    // =========================
    // Inside <pre>, whitespace IS the layout: split boxes on newline characters
    // (and <br>) instead of on inline tags, so syntax-highlighter token spans
    // (shiki, Pygments, highlight.js) are transparent. Leading indentation is
    // kept in the text; only zero-width chars and trailing whitespace are
    // stripped. split_reason "code_line" tells step_02 not to re-merge these
    // boxes by struct_tag_id (same contract as "br_tag").
    if (structTag === "PRE") {
      const styles = extractStyles(structEl, structTag);
      const domMeta = extractDomMetadata(structEl);
      const isStrike = detectStrikethrough(structEl);
      const scriptType = detectScriptType(structEl);
      const ixbrlId = findIxbrlId(structEl);

      // Anchor ancestry on a direct <code> wrapper when present (pre > code is
      // the standard highlighter structure), so struct_ancestors ends
      // [..., "pre", "code"] instead of stopping at the <pre>.
      const codeChild = structEl.querySelector(":scope > code");
      const preHierarchy = codeChild ? extractHierarchy(codeChild) : hierarchy;
      let lineStart = null; // { node, offset }
      let lineText = "";

      // end: { node, offset } at a "\n", { before: el } for <br>, or null at EOF
      const flushCodeLine = (end) => {
        const text = stripZeroWidth(lineText).replace(/\s+$/, "");
        const start = lineStart;
        lineText = "";
        lineStart = null;
        if (!start || !text.trim()) return;

        try {
          range.setStart(start.node, start.offset);
          if (!end) {
            range.setEndAfter(structEl.lastChild || structEl);
          } else if (end.before) {
            range.setEndBefore(end.before);
          } else {
            range.setEnd(end.node, end.offset);
          }

          const rect = range.getBoundingClientRect();
          if (rect.width <= 0 || rect.height <= 0) return;

          boxes.push({
            box_id: boxIdObj.value++,
            struct_tag: "pre",
            wrapping_tag: "pre",
            split_reason: "code_line",
            struct_tag_id: structureTagId,
            text: text,
            x_left: rect.left,
            x_right: rect.right,
            y_top: rect.top,
            y_bottom: rect.bottom,
            width: rect.width,
            height: rect.height,
            text_orientation: textOrientation,
            font_size: styles.font_size,
            font_family: styles.font_family,
            font_weight: styles.font_weight,
            bold_ratio: styles.bold_ratio,
            italic_ratio: styles.italic_ratio,
            underlined_ratio: styles.underlined_ratio,
            strikethrough_ratio: isStrike ? 1.0 : 0.0,
            is_strikethrough: isStrike,
            script_type: scriptType,
            non_stroking_color: styles.non_stroking_color,
            stroking_color: styles.stroking_color,
            text_align: structTextAlign,
            link_url: "",
            img_alt: "",
            img_src: "",
            dom_id: domMeta.domId,
            dom_class: domMeta.domClass,
            html_data_attrs: domMeta.dataAttrs,
            ixbrl_id: ixbrlId,
            table_id: tableContext.tableId,
            table_row_id: tableContext.tableRowId,
            table_cell_index: tableContext.tableCellIndex,
            table_row_cell_count: tableContext.tableRowCellCount,
            page_number: pageContext.page_number,
            page_width: pageContext.page_width,
            page_height: pageContext.page_height,
            page_format: pageContext.page_format,
            ancestor_ids: preHierarchy.ancestor_ids,
            ancestor_classes: preHierarchy.ancestor_classes,
            struct_ancestors: preHierarchy.struct_ancestors,
            struct_ancestor_ids: preHierarchy.struct_ancestor_ids,
            ancestor_aria_roles: preHierarchy.ancestor_aria_roles
          });
        } catch (e) {
          // Range error - skip this line
        }
      };

      const preWalker = document.createTreeWalker(
        structEl,
        NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
        {
          acceptNode: (node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              return node.tagName.toUpperCase() === "BR"
                ? NodeFilter.FILTER_ACCEPT
                : NodeFilter.FILTER_SKIP;
            }
            return NodeFilter.FILTER_ACCEPT;
          }
        }
      );

      while (preWalker.nextNode()) {
        const node = preWalker.currentNode;

        if (node.nodeType === Node.ELEMENT_NODE) { // <br>
          flushCodeLine({ before: node });
          continue;
        }

        const value = node.nodeValue || "";
        let from = 0;
        let idx;
        while ((idx = value.indexOf("\n", from)) !== -1) {
          if (!lineStart) lineStart = { node: node, offset: from };
          lineText += value.slice(from, idx);
          flushCodeLine({ node: node, offset: idx });
          from = idx + 1;
        }
        if (from < value.length) {
          if (!lineStart) lineStart = { node: node, offset: from };
          lineText += value.slice(from);
        }
      }
      flushCodeLine(null);

      return boxes;
    }

    const flushBox = (rangeEnd) => {
      const text = normalize(currentText);
      if (!text || !rangeStart) return;
      
      // Prepend list marker to first box only
      const finalText = listMarker && !listMarkerAdded ? listMarker + text : text;
      if (listMarker && !listMarkerAdded) {
        listMarkerAdded = true;
      }
      
      try {
        range.setStart(rangeStart.node, rangeStart.offset);
        if (rangeEnd) {
          range.setEnd(rangeEnd.node, rangeEnd.offset);
        } else {
          range.setEndAfter(structEl.lastChild || structEl);
        }
        
        const rect = range.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          // Extract styles from current context element
          const styles = extractStyles(currentInlineElement, currentInlineTag);

          // Strikethrough / script (super/subscript) — ancestor-aware, bounded to structEl
          const isStrike = detectStrikethrough(currentInlineElement);
          const scriptType = detectScriptType(currentInlineElement);

          // Extract DOM metadata from the inline element
          const domMeta = extractDomMetadata(currentInlineElement);
          
          
          boxes.push({
            box_id: boxIdObj.value++,
            struct_tag: structTag.toLowerCase(),
            wrapping_tag: currentInlineTag.toLowerCase(),
            split_reason: splitReason,
            struct_tag_id: structureTagId,
            text: finalText,
            x_left: rect.left,
            x_right: rect.right,
            y_top: rect.top,
            y_bottom: rect.bottom,
            width: rect.width,
            height: rect.height,
            text_orientation: textOrientation,
            font_size: styles.font_size,
            font_family: styles.font_family,
            font_weight: styles.font_weight,
            bold_ratio: styles.bold_ratio,
            italic_ratio: styles.italic_ratio,
            underlined_ratio: styles.underlined_ratio,
            strikethrough_ratio: isStrike ? 1.0 : 0.0,
            is_strikethrough: isStrike,
            script_type: scriptType,
            non_stroking_color: styles.non_stroking_color,
            stroking_color: styles.stroking_color,
            text_align: structTextAlign,
            link_url: currentLinkUrl,
            img_alt: "",
            img_src: "",
            dom_id: domMeta.domId,
            dom_class: domMeta.domClass,
            html_data_attrs: domMeta.dataAttrs,
            ixbrl_id: currentIxbrlId,
            table_id: tableContext.tableId,
            table_row_id: tableContext.tableRowId,
            table_cell_index: tableContext.tableCellIndex,
            table_row_cell_count: tableContext.tableRowCellCount,
            page_number: pageContext.page_number,
            page_width: pageContext.page_width,
            page_height: pageContext.page_height,
            page_format: pageContext.page_format,
            ancestor_ids: hierarchy.ancestor_ids,
            ancestor_classes: hierarchy.ancestor_classes,
            struct_ancestors: hierarchy.struct_ancestors,
            struct_ancestor_ids: hierarchy.struct_ancestor_ids,
            ancestor_aria_roles: hierarchy.ancestor_aria_roles
          });
        }
      } catch (e) {
        // Range error - skip this box
      }
      
      // Reset
      currentText = "";
      rangeStart = null;
    };
    
    // TreeWalker to process text nodes, inline elements, and images
    const walker = document.createTreeWalker(
      structEl,
      NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
      {
        acceptNode: (node) => {
          if (node === structEl) return NodeFilter.FILTER_SKIP;
          
          if (node.nodeType === Node.ELEMENT_NODE) {
            const tag = node.tagName.toUpperCase();
            
            // Skip structure descendants (shouldn't happen due to atomic filtering)
            if (STRUCTURE_TAGS.has(tag)) return NodeFilter.FILTER_REJECT;
            
            // Accept images and horizontal rules
            if (tag === "IMG" || tag === "HR") {
              return NodeFilter.FILTER_ACCEPT;
            }
            
            // Accept iXBRL elements (treat as transparent - skip but traverse children)
            if (tag.startsWith("IX:")) {
              return NodeFilter.FILTER_SKIP;
            }
            
            // Accept split and transparent inline tags, BR
            if (INLINE_SPLIT_TAGS.has(tag) || INLINE_TRANSPARENT_TAGS.has(tag)) {
              return NodeFilter.FILTER_ACCEPT;
            }
            
            // Skip other elements but traverse their children
            return NodeFilter.FILTER_SKIP;
          }
          
          if (node.nodeType === Node.TEXT_NODE) {
            return NodeFilter.FILTER_ACCEPT;
          }
          
          return NodeFilter.FILTER_SKIP;
        }
      }
    );
    
    while (walker.nextNode()) {
      const node = walker.currentNode;
      
      if (node.nodeType === Node.ELEMENT_NODE) {
        const tag = node.tagName.toUpperCase();
        
        // Handle images - create an image box
        if (tag === "IMG") {
          // Flush any pending text box first
          if (currentText.trim()) {
            flushBox({ node: node, offset: 0 });
          }
          
          // Create image box
          const imageBox = extractImageBox(node, boxIdObj);
          if (imageBox) {
            imageBox.struct_tag_id = structureTagId;
            // Check if image is inside a link
            const linkUrl = findLinkUrl(node);
            if (linkUrl) {
              imageBox.link_url = linkUrl;
            }
            boxes.push(imageBox);
          }
          
          // Reset context after image
          currentInlineTag = structTag;
          currentInlineElement = structEl;
          currentLinkUrl = "";
          currentIxbrlId = "";
          splitReason = "after_image";
          continue;
        }
        
        // Handle horizontal rules - create an HR box
        if (tag === "HR") {
          // Flush any pending text box first
          if (currentText.trim()) {
            flushBox({ node: node, offset: 0 });
          }
          
          // Create HR box
          const hrBox = extractHrBox(node, boxIdObj);
          if (hrBox) {
            hrBox.struct_tag_id = structureTagId;
            boxes.push(hrBox);
          }
          
          // Reset context after HR
          currentInlineTag = structTag;
          currentInlineElement = structEl;
          currentLinkUrl = "";
          currentIxbrlId = "";
          splitReason = "after_hr";
          continue;
        }
        
        // BR always causes a split
        if (tag === "BR") {
          flushBox({ node: node, offset: 0 });
          currentInlineTag = structTag; // Reset to structure
          currentInlineElement = structEl; // Reset to structure element
          currentLinkUrl = ""; // Reset link URL
          currentIxbrlId = ""; // Reset iXBRL ID
          splitReason = "br_tag";
          continue;
        }
        
        // For inline tags, context detection happens at text node level
        // Just skip element nodes for inline split and transparent tags
        if (INLINE_SPLIT_TAGS.has(tag) || INLINE_TRANSPARENT_TAGS.has(tag)) {
          continue;
        }
      }
      
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.nodeValue || "";
        if (!text.trim()) continue;
        
        // Check if we're in a different inline context
        const inlineContext = findInlineSplitParent(node);
        const linkUrl = findLinkUrl(node);
        const ixbrlId = findIxbrlId(node);
        
        // If context changed (element, link, or iXBRL), flush previous box and start new one
        const contextChanged = inlineContext.element !== currentInlineElement;
        const linkChanged = linkUrl !== currentLinkUrl;
        const ixbrlChanged = ixbrlId !== currentIxbrlId;
        
        if (contextChanged || linkChanged || ixbrlChanged) {
          if (currentText.trim()) {
            flushBox({ node: node, offset: 0 });
          }
          currentInlineTag = inlineContext.tag;
          currentInlineElement = inlineContext.element;
          currentLinkUrl = linkUrl;
          currentIxbrlId = ixbrlId;
          
          if (ixbrlChanged && ixbrlId) {
            splitReason = "ixbrl_change";
          } else if (linkChanged && linkUrl) {
            splitReason = "link_change";
          } else if (contextChanged) {
            splitReason = inlineContext.tag === structTag ? "inline_exit" : "inline_tag";
          }
        }
        
        // First text node in this context - set range start
        if (!rangeStart) {
          rangeStart = { node: node, offset: 0 };
        }
        
        currentText += text;
      }
    }
    
    // Flush final box
    flushBox(null);
    
    return boxes;
  };

  // =========================
  // MAIN EXTRACTION
  // =========================
  
  const extractAll = (root) => {
    const boxes = [];
    const boxIdObj = { value: 1 };
    let structureTagId = 1;

    // Stamp every element with a document-order unique id so struct_ancestor_ids
    // can reference shared ancestor instances consistently. Elements-only order
    // yields the expected gapped sequence (e.g. [0, 1, 2, 90, 92, 93, 94]) because
    // an ancestor chain skips over all the intervening sibling subtrees.
    // Stamp the whole document (html + head + body + descendants) so that body and
    // html — which are ancestors of every box — also receive ids.
    let _uidCounter = 0;
    _elemUid.set(document.documentElement, _uidCounter++);
    for (const el of document.documentElement.querySelectorAll("*")) {
      _elemUid.set(el, _uidCounter++);
    }

    // Scan document and assign page numbers
    assignPageNumbers(root);
    
    // Find all atomic structure elements
    const atomicStructures = findAtomicStructures(root);
    
    // Create sets to track processed elements and avoid duplicates
    const processedImages = new Set();
    const processedHrs = new Set();
    
    for (const structEl of atomicStructures) {
      const structBoxes = extractBoxesFromStructure(structEl, structureTagId, boxIdObj);
      
      // Track images and HRs that were processed within this structure
      const imgs = structEl.querySelectorAll("img");
      for (const img of imgs) {
        processedImages.add(img);
      }
      const hrs = structEl.querySelectorAll("hr");
      for (const hr of hrs) {
        processedHrs.add(hr);
      }
      
      // Only increment struct_tag_id if boxes were actually created
      if (structBoxes.length > 0) {
        boxes.push(...structBoxes);
        structureTagId++;
      }
    }
    
    // Find standalone images not within atomic structures
    const allImages = root.querySelectorAll("img");
    for (const img of allImages) {
      if (processedImages.has(img)) continue;
      
      const imageBox = extractImageBox(img, boxIdObj);
      if (imageBox) {
        imageBox.struct_tag_id = structureTagId++;
        
        // Check if image is inside a link
        const linkUrl = findLinkUrl(img);
        if (linkUrl) {
          imageBox.link_url = linkUrl;
        }
        
        boxes.push(imageBox);
      }
    }
    
    // Find standalone HRs not within atomic structures
    const allHrs = root.querySelectorAll("hr");
    for (const hr of allHrs) {
      if (processedHrs.has(hr)) continue;

      const hrBox = extractHrBox(hr, boxIdObj);
      if (hrBox) {
        hrBox.struct_tag_id = structureTagId++;
        boxes.push(hrBox);
      }
    }

    // Annotate every <table> element with its assigned JS table_id so that
    // Python can look up tables by attribute rather than relying on document-
    // order indexing (which diverges from JS assignment order for nested tables).
    const allTables = root.querySelectorAll("table");
    for (const tbl of allTables) {
      if (tableIds.has(tbl)) {
        tbl.setAttribute("data-docslicer-table-id", tableIds.get(tbl));
      }
    }

    return boxes;
  };

  // =========================
  // RUN
  // =========================
  
  return extractAll(document.body);
})();
