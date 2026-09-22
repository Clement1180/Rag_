"""Sérialisation : JSON compatible DoclingDocument, Markdown."""
from __future__ import annotations

from .model import DocModel, GroupItem, Item, PictureItem, TableItem, TextItem

SCHEMA_VERSION = "1.10.0"   # version DoclingDocument ciblée (docling-core 2.x)
META_KEY = "docx__props"    # docling impose « namespace__champ » pour les métadonnées libres
SIDECAR_KEY = "x_docx"      # clé racine ignorée par DoclingDocument (extra="ignore")

# Pourquoi un « sidecar » par défaut ? Les sérialiseurs Docling (export Markdown, HierarchicalChunker,
# HybridChunker) recopient `item.meta` dans le texte produit : des métadonnées techniques inline
# pollueraient les embeddings. Elles sont donc rangées à la racine, indexées par self_ref.


class _Sink:
    """Destination des métadonnées (évite tout état global : sérialisation thread-safe)."""
    __slots__ = ("mode", "items")

    def __init__(self, mode: str):
        self.mode, self.items = mode, {}


def _ref(r: str | None) -> dict | None:
    return {"$ref": r} if r else None


def _meta(it: Item, extra: dict | None = None, sidecar: "_Sink | None" = None) -> dict | None:
    m = {k: v for k, v in it.meta.items() if v not in (None, [], {})}
    out = dict(extra or {})
    if m and sidecar is not None:
        if sidecar.mode == "inline":
            out[META_KEY] = m
        elif sidecar.mode == "sidecar":
            sidecar.items[it.self_ref] = m
    return out or None


def _node(it: Item) -> dict:
    return {"self_ref": it.self_ref, "parent": _ref(it.parent),
            "children": [{"$ref": c} for c in it.children], "content_layer": it.content_layer}


def _text(it: TextItem, sc: _Sink) -> dict:
    d = _node(it)
    d.update({"label": it.label, "prov": [], "orig": it.orig, "text": it.text})
    if it.label == "section_header":
        d["level"] = it.level or 1
    if it.label == "list_item":
        d["enumerated"] = bool(it.enumerated)
        d["marker"] = it.marker or ""
    if it.hyperlink:
        d["hyperlink"] = it.hyperlink
    m = _meta(it, sidecar=sc)
    if m:
        d["meta"] = m
    return d


def _cell(c, include_bbox: bool = True) -> dict:
    return {"bbox": None, "row_span": c.row_span, "col_span": c.col_span,
            "start_row_offset_idx": c.start_row, "end_row_offset_idx": c.start_row + c.row_span,
            "start_col_offset_idx": c.start_col, "end_col_offset_idx": c.start_col + c.col_span,
            "text": c.text, "column_header": c.column_header, "row_header": c.row_header,
            "row_section": False}


def table_grid(t: TableItem) -> list[list[dict]]:
    """Grille dense : une cellule fusionnée est répétée dans chaque case couverte."""
    empty = lambda r, c: {"bbox": None, "row_span": 1, "col_span": 1, "start_row_offset_idx": r,
                          "end_row_offset_idx": r + 1, "start_col_offset_idx": c,
                          "end_col_offset_idx": c + 1, "text": "", "column_header": False,
                          "row_header": False, "row_section": False}
    grid = [[empty(r, c) for c in range(t.num_cols)] for r in range(t.num_rows)]
    for cell in t.cells:
        d = _cell(cell)
        for r in range(cell.start_row, min(t.num_rows, cell.start_row + cell.row_span)):
            for c in range(cell.start_col, min(t.num_cols, cell.start_col + cell.col_span)):
                grid[r][c] = d
    return grid


def _table(it: TableItem, include_grid: bool, sc: _Sink) -> dict:
    d = _node(it)
    data = {"table_cells": [_cell(c) for c in it.cells], "num_rows": it.num_rows, "num_cols": it.num_cols}
    if include_grid:
        data["grid"] = table_grid(it)
    d.update({"label": "table", "prov": [], "captions": [{"$ref": c} for c in it.captions],
              "references": [], "footnotes": [], "image": None, "data": data})
    m = _meta(it, sidecar=sc)
    if m:
        d["meta"] = m
    return d


def _picture(it: PictureItem, sc: _Sink) -> dict:
    d = _node(it)
    image = None
    if it.image and it.image.get("uri"):
        image = {"mimetype": it.image.get("mimetype", "application/octet-stream"), "dpi": 96,
                 "size": it.image.get("size") or {"width": 0.0, "height": 0.0}, "uri": it.image["uri"]}
    d.update({"label": it.label, "prov": [], "captions": [{"$ref": c} for c in it.captions],
              "references": [], "footnotes": [], "image": image})
    extra = {}
    if it.description:
        extra["description"] = {"text": it.description["text"], "created_by": it.description.get("created_by")}
    meta = dict(it.meta)
    if it.image:   # métadonnées binaires complètes (hash, taille, partie zip…) hors ImageRef strict
        meta["image"] = {k: v for k, v in it.image.items() if k not in ("uri",) or not str(v).startswith("data:")}
    tmp = PictureItem(self_ref=it.self_ref, parent=it.parent, label=it.label, meta=meta)
    m = _meta(tmp, extra, sidecar=sc)
    if m:
        d["meta"] = m
    return d


def _group(it: GroupItem, sc: _Sink) -> dict:
    d = _node(it)
    d.update({"name": it.name, "label": it.label})
    m = _meta(it, sidecar=sc)
    if m:
        d["meta"] = m
    return d


def to_docling_dict(model: DocModel, include_grid: bool = False, meta_mode: str = "sidecar") -> dict:
    """meta_mode : "sidecar" (défaut, sûr pour le RAG), "inline" (item.meta.docx__props), "none"."""
    if meta_mode not in ("sidecar", "inline", "none"):
        raise ValueError(meta_mode)

    def root(g: GroupItem) -> dict:
        return {"self_ref": g.self_ref, "children": [{"$ref": c} for c in g.children],
                "content_layer": g.content_layer, "name": g.name, "label": g.label}
    sc = _Sink(meta_mode)
    out = {
        "schema_name": "DoclingDocument", "version": SCHEMA_VERSION, "name": model.name,
        "origin": model.origin, "furniture": root(model.furniture), "body": root(model.body),
        "groups": [_group(g, sc) for g in model.groups], "texts": [_text(t, sc) for t in model.texts],
        "pictures": [_picture(p, sc) for p in model.pictures],
        "tables": [_table(t, include_grid, sc) for t in model.tables],
        "key_value_items": [], "form_items": [], "pages": {},
    }
    if meta_mode == "sidecar":
        out[SIDECAR_KEY] = {"parser": "docxrag", "document": model.meta, "items": sc.items}
    return out


# ----------------------------------------------------------------- Markdown
def table_to_markdown(t: TableItem, rows: range | None = None, header_rows: int | None = None) -> str:
    grid = [[""] * t.num_cols for _ in range(t.num_rows)]
    for c in t.cells:
        txt = c.text.replace("\n", "<br>").replace("|", "\\|")
        for r in range(c.start_row, min(t.num_rows, c.start_row + c.row_span)):
            for k in range(c.start_col, min(t.num_cols, c.start_col + c.col_span)):
                grid[r][k] = txt   # valeur fusionnée répétée : chaque ligne reste autoportante
    if not grid or t.num_cols == 0:
        return ""
    hdr_n = header_rows if header_rows is not None else _header_rows(t)
    head = grid[:hdr_n] if hdr_n else [grid[0]]
    body_rows = [grid[i] for i in (rows if rows is not None else range(hdr_n or 1, t.num_rows)) if i >= (hdr_n or 1)]
    header = [" / ".join(dict.fromkeys(h for h in col if h)) for col in zip(*head)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * t.num_cols]
    lines += ["| " + " | ".join(r) + " |" for r in body_rows]
    return "\n".join(lines)


def _header_rows(t: TableItem) -> int:
    rows = {c.start_row for c in t.cells if c.column_header}
    n = 0
    while n in rows:
        n += 1
    return n


def to_markdown(model: DocModel) -> str:
    out: list[str] = []
    for it, _ in model.iterate():
        if isinstance(it, TextItem):
            if it.label == "title":
                out.append(f"# {it.text}")
            elif it.label == "section_header":
                out.append(f"{'#' * min(6, (it.level or 1) + 1)} {it.text}")
            elif it.label == "list_item":
                depth = 0
                p = model.resolve(it.parent)
                while isinstance(p, GroupItem) and p.parent and p.parent not in ("#/body",):
                    par = model.resolve(p.parent)
                    if par.label != "list_item":
                        break
                    depth += 1
                    p = model.resolve(par.parent)
                out.append(f"{'  ' * depth}{it.marker or '-'} {it.text}")
            elif it.label == "caption":
                out.append(f"*{it.text}*")
            elif it.label == "footnote":
                out.append(f"> [note] {it.text}")
            else:
                out.append(it.text)
        elif isinstance(it, TableItem) and not it.meta.get("nested_in_cell"):
            out.append(table_to_markdown(it))
        elif isinstance(it, PictureItem):
            alt = it.meta.get("alt_text") or it.meta.get("name") or it.label
            uri = (it.image or {}).get("uri", "")
            out.append(f"![{alt}]({'' if str(uri).startswith('data:') else uri})")
            if it.description:
                out.append(f"> {it.description['text']}")
    return "\n\n".join(x for x in out if x)
