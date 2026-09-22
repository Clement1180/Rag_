"""Représentation intermédiaire « plate » produite par l'extraction XML.

Étape 1 (extraction) -> liste de Block dans l'ordre de lecture
Étape 2 (classification) -> rôle de chaque ParaBlock (titre, liste, légende, sommaire…)
Étape 3 (assemblage) -> DocModel hiérarchique compatible Docling
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ImageRef:
    kind: str                     # image | chart | ole | smartart
    rel_id: str | None
    target: str | None            # partie zip (word/media/image1.png) ou URL externe
    external: bool = False
    mimetype: str | None = None
    name: str | None = None       # wp:docPr/@name
    descr: str | None = None      # texte alternatif
    title: str | None = None
    docpr_id: str | None = None
    placement: str = "inline"     # inline | anchor | vml | ole
    extent_emu: tuple[int, int] | None = None
    position: dict | None = None  # ancrage : offsets, relativeFrom, wrap
    decorative: bool = False
    char_offset: int = 0          # position dans le texte du paragraphe
    ole_prog_id: str | None = None
    chart_text: str | None = None


@dataclass(slots=True)
class ParaBlock:
    text: str
    style_id: str | None
    style_name: str
    style_names_chain: tuple[str, ...]
    outline_lvl: int | None
    num_id: str | None
    ilvl: int | None
    list_marker: str | None = None
    list_enumerated: bool = False
    list_lvl_text: str | None = None
    size: float | None = None        # taille dominante (pondérée par caractères)
    bold_ratio: float = 0.0
    italic_ratio: float = 0.0
    underline_ratio: float = 0.0
    caps: bool = False
    jc: str | None = None
    indent_left: int = 0
    in_table: bool = False
    in_textbox: bool = False
    in_toc_sdt: bool = False
    toc_field: bool = False
    seq_type: str | None = None      # champ SEQ Figure / Tableau
    anchors: list[str] = field(default_factory=list)      # liens internes (w:anchor)
    bookmarks: list[str] = field(default_factory=list)
    hyperlinks: list[str] = field(default_factory=list)
    footnote_ids: list[str] = field(default_factory=list)
    endnote_ids: list[str] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    textboxes: list[list["ParaBlock | TableBlock"]] = field(default_factory=list)
    page_break_before: bool = False
    page_break_after: bool = False
    is_math: bool = False
    page_hint: int = 1
    order: int = 0
    # rempli par la classification
    role: str = "text"               # text|heading|title|list_item|caption|toc|toc_title|furniture|empty|formula
    level: int | None = None
    heading_source: str | None = None
    confidence: float = 1.0
    caption_target: int | None = None   # index de bloc de la cible
    toc_level: int | None = None
    toc_page: str | None = None
    manual_marker: str | None = None


@dataclass(slots=True)
class CellData:
    start_row: int
    start_col: int
    row_span: int
    col_span: int
    text: str
    column_header: bool = False
    row_header: bool = False
    images: list[ImageRef] = field(default_factory=list)
    bold: bool = False


@dataclass(slots=True)
class TableBlock:
    cells: list[CellData]
    num_rows: int
    num_cols: int
    style_name: str | None = None
    nested: list["TableBlock"] = field(default_factory=list)
    nested_in: tuple[int, int] | None = None
    header_source: str | None = None
    in_textbox: bool = False
    page_hint: int = 1
    order: int = 0
    caption_target: int | None = None  # compat interface (inutilisé)


Block = ParaBlock | TableBlock
