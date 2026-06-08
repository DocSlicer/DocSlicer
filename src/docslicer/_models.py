from typing import TypedDict, Literal, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
from typing import Set, get_type_hints

# ==========================================
# BASE SCHEMAS (Reusable Components)
# ==========================================

# --- Geometry and Styling --- #

class Geometry(TypedDict, total=False):
    """Base geometry for word-level objects (x_left/x_right)."""
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float
    width: float
    height: float
    text_align: Optional[Literal["left", "center", "right", "justify"]]  # HTML only
    x_center: float
    y_center: float
    top_bucket: Optional[int]
    center_bucket: Optional[int]
    bottom_bucket: Optional[int]


class Styling(TypedDict, total=False):
    """Text styling attributes."""
    font_size: float  # in pt (pdf) and px (html)
    font_name: str
    font_weight: int
    bold_ratio: float
    italic_ratio: float
    underlined_ratio: float
    underline_role: Literal["text_decoration", "border", "table_grid"]
    has_vertical_line: bool
    non_stroking_color: Optional[str]        # hex string (#rrggbb) or None
    stroking_color: Optional[str]            # hex string (#rrggbb) or None
    background_non_stroking_color: Any
    background_stroking_color: Any
    combined_shape_id_underline: int
    combined_shape_id_container: int
    tag: str  # html tag name

# Not implemented:
    # underline_type: Literal["single", "double", "dotted", "dashed", "wavy"]
    # superscript / subscript
    # font_family: str

# --- Content Metadata --- #

class BaseMetadata(TypedDict, total=False):
    """Metadata attributes."""
    has_link: bool
    link_url: str  # url for external, dest for internal
    link_type: Literal["external", "internal", "anchor"]
    ixbrl_id: str
    img_alt: str
    img_src: str
    is_name_strip: bool


class AggregatedMetadata(TypedDict, total=False):
    """Aggregated metadata attributes."""
    links_per_line: List[List[str]]
    ixbrl_ids: List[List[str]]


# --- Document Metadata --- #

class PageContext(TypedDict, total=False):
    """Page-level context information."""
    page_width: float
    page_height: float
    page_number: int
    page_label: str
    section: str


class DocumentIdentity(TypedDict, total=False):
    """Document identity fields."""
    document_name: str
    document_id: str  # uuidv4
    source_url: str


# --- Content Base --- #

class ContentBase(TypedDict, total=False):
    """Text content and statistics."""
    text: str
    char_count: int
    alpha_count: int
    digit_count: int
    uppercase_count: int
    token_count: int
    alpha_token_count: int
    capitalized_token_count: int


# ==========================================
# COMPOSED SCHEMAS (Inheriting from bases)
# ==========================================

class WordSchema(Geometry, Styling, ContentBase, BaseMetadata, PageContext, DocumentIdentity):
    """
    Complete schema for word-level DataFrames.

    Inherits:
    - DocumentIdentity: document_name, document_id
    - Geometry: x_left, x_right, y_top, y_bottom, width, height
    - Styling: font_name, font_size, bold_ratio, italic_ratio, colors, etc.
    - ContentBase: text, char_count, digit_count, etc.
    - PageContext: page_width, page_height, page_number
    """
    word_id: int


class ShapeSchema(Geometry, Styling, DocumentIdentity):
    """
    Schema for shape/rectangle DataFrames.

    Inherits:
    - DocumentIdentity: document_name, document_id
    - Geometry: x_left, x_right, y_top, y_bottom
    """
    shape_id: int
    shape_type: Literal["line", "rect", "curve"]
    linewidth: float
    fill: bool
    stroke: bool
    paint_op: str
    shape_orientation: Literal["horizontal", "vertical", "unknown"]


class LinkSchema(Geometry, DocumentIdentity):
    link_id: int
    link_url: str
    link_dest: str
    link_type: Literal["external", "internal", "anchor"]


class CellSchema(DocumentIdentity, Geometry, Styling, ContentBase):
    """
    Schema for cell-level DataFrames.

    Inherits:
    - DocumentIdentity: document_name, document_id
    - Geometry: x_left, x_right, y_top, y_bottom, width, height
    - ContentBase: text, char_count
    """
    cell_id: int
    temp_line_id: int
    line_id: int
    word_ids: List[int]


class LineSchema(DocumentIdentity, Geometry, Styling, ContentBase, PageContext):
    """
    Schema for line-level DataFrames.

    Inherits:
    - DocumentIdentity: document_name, document_id
    - Geometry: x_left, x_right, y_top, y_bottom, width, height
    - Styling: font_name, font_size, bold_ratio, italic_ratio, etc. (aggregated from words)
    - ContentBase: text, char_count, etc.
    - PageContext: page_width, page_height, page_number
    """
    line_id: int
    temp_line_id: int # Temporary line assignment to clean up OCR text (before analysis is done whether the line is text singlecol, multicol or table)
    word_ids: List[int]

    # Line-specific scores
    table_row_score: float
    heading_score: float

    # Line-specific layout
    median_x0x1_gap: float
    max_x0x1_gap: float
    gap_ratio: float
    capitalized_token_ratio: float


class BlockSchema(DocumentIdentity, Geometry, ContentBase):
    """Schema for block-level DataFrames."""
    block_id: int
    line_ids: List[int]
    page_label: str
    section: str

    # Style aggregates
    font_size: float
    is_bold: int
    is_italic: int
    is_underlined: int
    bold_ratio: float
    uppercase_ratio: float
    digit_ratio: float
    text_align: str
    has_link: int
    has_image: int
    is_single_line: bool

    # Classification
    block_type: Literal["page_label", "toc", "toc_heading", "exhibits", "exhibit_heading", "hr", "image",
                        "table", "heading", "paragraph"]
    heading_score: float

    # Optional metadata
    table_group_id: Optional[int]
    table_ids: List[int]
    links_per_line: Optional[List[List[str]]]
    ixbrl_ids: Optional[List[List[str]]]


# --- docx / pptx run level --- #

class _RunProvenance(TypedDict, total=False):
    """Fields that describe where an XML run sits in the document structure."""
    run_type: str               # text / tab / image_ref / chart_ref / section_break / …
    run_index: int              # position within parent paragraph
    order_index: int            # global sequential position in document
    source_part: str            # body / footnotes / endnotes / header / footer / …
    source_part_id: str         # ID of the item within that part
    header_footer_type: str     # body / header / footer  (docx)
    nested_table_depth: int     # 0 = top-level, >0 = inside table cell  (docx)
    page_break_before: bool     # paragraph-level page break flag  (docx)
    section_break_after: bool   # docx
    section_break_type: str     # nextPage / continuous / evenPage / oddPage  (docx)
    bookmark_id: str            # docx
    bookmark_ids: List[str]     # docx
    bookmark_names: List[str]   # docx
    comment_id: str             # docx
    footnote_id: str            # docx
    endnote_id: str             # docx
    placeholder_type: str       # title / body / subtitle / …  (pptx)


class _StyleInheritance(TypedDict, total=False):
    """Style-inheritance chain columns, resolved through the docx style graph."""
    paragraph_style_id: str
    paragraph_style_name: str
    effective_paragraph_style_id: str
    effective_paragraph_style_name: str
    character_style_id: str
    character_style_name: str
    effective_character_style_id: str
    effective_character_style_name: str


class _ListOutline(TypedDict, total=False):
    """List and outline-level fields shared by docx and pptx."""
    list_num_id: str            # docx numbering definition ID
    list_level: str             # 0-based level within the numbering definition
    list_label: str             # rendered label: "1.", "a)", "•", …
    list_type: str              # pptx: bullet / autoNumber / none
    list_auto_type: str         # pptx: arabicPeriod / romanUC / …
    list_start_at: int          # pptx: override start value
    outline_level: int          # 0-based heading depth from style (docx/pptx)


class RunSchema(
    _RunProvenance, _StyleInheritance, _ListOutline,
    Geometry, Styling, ContentBase, BaseMetadata, PageContext, DocumentIdentity,
):
    """
    Schema for run-level DataFrames produced by docx and pptx extractors.

    A run is the atomic text unit in Office XML — a sequence of characters
    sharing identical character-level formatting within one paragraph.

    Inherits:
    - DocumentIdentity: document_name, document_id
    - PageContext: page_number, page_label, section
    - Geometry: x_left / x_right / y_top / y_bottom (not always populated)
    - Styling: font_name, font_size, bold/italic flags, colors, text_align, …
    - ContentBase: text, char_count, …
    - BaseMetadata: has_link, link_url, …
    - _RunProvenance: run_type, order_index, source_part, header_footer_type, …
    - _StyleInheritance: paragraph_style_id, effective_paragraph_style_id, …
    - _ListOutline: list_num_id, list_level, list_label, outline_level, …
    """
    run_id: int
    paragraph_id: int
    section_id: int             # docx
    slide_index: int            # pptx


class ParagraphSchema(
    _ListOutline, _StyleInheritance,
    Geometry, Styling, ContentBase, PageContext, DocumentIdentity,
):
    """
    Schema for paragraph-level DataFrames (docx / pptx step_04/05).

    Paragraphs are produced by aggregating runs; one row = one XML <w:p> / <a:p>.

    Inherits:
    - DocumentIdentity, PageContext, Geometry, Styling, ContentBase
    - _ListOutline, _StyleInheritance
    """
    paragraph_id: int
    run_count: int              # number of runs aggregated into this paragraph
    block_type: Optional[str]   # image / table / None (set before line-building)


# ==========================================
# SCHEMA VALIDATION (works with inheritance)
# ==========================================

class SchemaValidator:
    """Validates DataFrame schemas against TypedDict definitions."""

    @staticmethod
    def get_all_columns(schema: type) -> Set[str]:
        """Extract all column names from TypedDict, including inherited ones."""
        return set(get_type_hints(schema).keys())

    @staticmethod
    def get_required_columns(schema: type) -> Set[str]:
        """Get only required columns (from TypedDict with total=True)."""
        if hasattr(schema, '__required_keys__'):
            return schema.__required_keys__
        return set(get_type_hints(schema).keys())

    @staticmethod
    def validate(
        df: pd.DataFrame,
        schema: type,
        check_required_only: bool = True,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Validate DataFrame against schema (including inherited fields).

        Args:
            df: DataFrame to validate
            schema: TypedDict schema class
            check_required_only: Only validate required fields
            strict: Fail if extra columns present

        Returns:
            Original DataFrame (for chaining)

        Raises:
            ValueError: If schema validation fails
        """
        expected_cols = (
            SchemaValidator.get_required_columns(schema)
            if check_required_only
            else SchemaValidator.get_all_columns(schema)
        )

        df_cols = set(df.columns)
        missing = expected_cols - df_cols
        extra = df_cols - expected_cols

        if missing:
            raise ValueError(
                f"Schema validation failed for {schema.__name__}\n"
                f"  Missing columns: {sorted(missing)}\n"
                f"  Available: {sorted(df_cols)}"
            )

        if strict and extra:
            raise ValueError(
                f"Schema validation failed for {schema.__name__}\n"
                f"  Unexpected columns: {sorted(extra)}"
            )

        return df

    @staticmethod
    def print_schema(schema: type) -> None:
        """Print all fields in schema."""
        all_cols = SchemaValidator.get_all_columns(schema)
        required_cols = SchemaValidator.get_required_columns(schema)
        optional_cols = all_cols - required_cols
        print(f"\n{schema.__name__}")
        print(f"  Required: {sorted(required_cols)}")
        print(f"  Optional: {sorted(optional_cols)}")
        print(f"  Total: {len(all_cols)} columns")


# ==========================================
# CONVENIENCE VALIDATORS
# ==========================================

def validate_word_df(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate word-level DataFrame."""
    return SchemaValidator.validate(df, WordSchema, strict=strict)


def validate_line_df(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate line-level DataFrame."""
    return SchemaValidator.validate(df, LineSchema, strict=strict)


def validate_cell_df(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate cell-level DataFrame."""
    return SchemaValidator.validate(df, CellSchema, strict=strict)


def validate_block_df(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate block-level DataFrame."""
    return SchemaValidator.validate(df, BlockSchema, strict=strict)


def validate_shape_df(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate shape-level DataFrame."""
    return SchemaValidator.validate(df, ShapeSchema, strict=strict)


def validate_run_df(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate run-level DataFrame (docx / pptx)."""
    return SchemaValidator.validate(df, RunSchema, strict=strict)


def validate_paragraph_df(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate paragraph-level DataFrame (docx / pptx)."""
    return SchemaValidator.validate(df, ParagraphSchema, strict=strict)


# ==========================================
# GEOMETRY NORMALIZATION (prevent drift)
# ==========================================

def normalize_geometry(df: pd.DataFrame, target_schema: type) -> pd.DataFrame:
    """
    Standardize geometry column names to x_left/x_right/y_top/y_bottom.

    Renames legacy names (x0/x1/top/bottom/left/right/x_min/x_max) to the
    canonical names used in Geometry and all composed schemas.

    Args:
        df: DataFrame with potentially non-standard names
        target_schema: WordSchema, LineSchema, CellSchema, etc.

    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    schema_cols = SchemaValidator.get_all_columns(target_schema)

    if "x_left" in schema_cols:
        for src in ("x0", "left", "x_min"):
            if src in df.columns and "x_left" not in df.columns:
                df["x_left"] = df[src]
        for src in ("x1", "right", "x_max"):
            if src in df.columns and "x_right" not in df.columns:
                df["x_right"] = df[src]

    if "y_top" in schema_cols and "top" in df.columns and "y_top" not in df.columns:
        df["y_top"] = df["top"]
    if "y_bottom" in schema_cols and "bottom" in df.columns and "y_bottom" not in df.columns:
        df["y_bottom"] = df["bottom"]

    return df


# ==========================================
# UTILITY: Show inheritance chain
# ==========================================

def show_schema_inheritance() -> None:
    """Print inheritance for all schemas."""
    schemas = [WordSchema, RunSchema, ParagraphSchema, LineSchema, CellSchema, BlockSchema, ShapeSchema]
    for schema in schemas:
        print(f"\n{'='*60}")
        SchemaValidator.print_schema(schema)
        print(f"  Inherits from: {[b.__name__ for b in schema.__bases__ if b != dict]}")


# ==========================================
# TYPE ALIASES (re-exported from constants for backwards compatibility)
# ==========================================

from .constants import (
    BlockType,
    ShapeType,
    ShapeOrientation,
    Orientation,
    SectionType,
    TextAlign,
)

__all__ = [
    "BlockType",
    "ShapeType",
    "ShapeOrientation",
    "Orientation",
    "SectionType",
    "TextAlign",
]


# ==========================================
# BOUNDING BOX UTILITY
# ==========================================

@dataclass
class BBox:
    """Standalone bounding box (for utility functions)."""
    x_left: float
    y_top: float
    x_right: float
    y_bottom: float

    @property
    def width(self) -> float:
        return self.x_right - self.x_left

    @property
    def height(self) -> float:
        return self.y_bottom - self.y_top

    @property
    def center_x(self) -> float:
        return (self.x_left + self.x_right) / 2

    @property
    def center_y(self) -> float:
        return (self.y_top + self.y_bottom) / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def overlaps(self, other: "BBox", tolerance: float = 0.0) -> bool:
        """Check if this bbox overlaps with another."""
        return not (
            self.x_right + tolerance < other.x_left
            or self.x_left > other.x_right + tolerance
            or self.y_bottom + tolerance < other.y_top
            or self.y_top > other.y_bottom + tolerance
        )

    @classmethod
    def from_series(cls, row: pd.Series) -> "BBox":
        """Create BBox from a DataFrame row, accepting multiple naming conventions."""
        x_left = float(row.get("x_left", row.get("x0", row.get("x_min", row.get("left", 0)))))
        x_right = float(row.get("x_right", row.get("x1", row.get("x_max", row.get("right", 0)))))
        y_top = float(row.get("y_top", row.get("top", 0)))
        y_bottom = float(row.get("y_bottom", row.get("bottom", 0)))
        return cls(x_left=x_left, y_top=y_top, x_right=x_right, y_bottom=y_bottom)


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    show_schema_inheritance()

    df_words = pd.DataFrame({
        "document_name": ["test.pdf"],
        "word_id": [1],
        "x_left": [100.0],
        "x_right": [150.0],
        "y_top": [200.0],
        "y_bottom": [220.0],
        "text": ["Hello"],
    })

    try:
        validate_word_df(df_words)
        print("Word DataFrame validated successfully!")
    except ValueError as e:
        print(f"Validation failed: {e}")
