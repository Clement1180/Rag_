"""Étape 3 : blocs classés -> DocModel hiérarchique (sections imbriquées, listes, légendes)."""
from __future__ import annotations

from .blocks import Block, ImageRef, ParaBlock, TableBlock
from .images import ImageStore
from .model import DocModel, GroupItem, Item, TextItem


class Assembler:
    def __init__(self, model: DocModel, images: ImageStore, notes: dict[str, dict[str, str]]):
        self.m, self.images, self.notes = model, images, notes
        self.stack: list[tuple[int, Item]] = []        # (niveau, titre)
        self.lists: list[tuple[int, GroupItem]] = []   # (ilvl, groupe)
        self.last_list_item: dict[int, TextItem] = {}
        self.toc_group: GroupItem | None = None
        self.bookmark_to_ref: dict[str, str] = {}
        self.toc_items: list[tuple[TextItem, ParaBlock]] = []
        self._notes_done: set[str] = set()
        self._title_seen = False

    # ----------------------------------------------------------- utilitaires
    def parent(self) -> Item:
        return self.stack[-1][1] if self.stack else self.m.body

    def heading_path(self) -> list[str]:
        return [it.text for _, it in self.stack]

    def _close_lists(self) -> None:
        self.lists.clear()
        self.last_list_item.clear()

    @staticmethod
    def _base_meta(b: Block) -> dict:
        meta = {"order": b.order, "page_hint": b.page_hint}
        if isinstance(b, ParaBlock):
            if b.style_name:
                meta["style"] = b.style_name
            if b.in_textbox:
                meta["in_textbox"] = True
            if b.in_table:
                meta["in_table"] = True
        return meta

    # --------------------------------------------------------------- run
    def build(self, blocks: list[Block]) -> None:
        captions_for: dict[int, list[ParaBlock]] = {}
        for b in blocks:
            if isinstance(b, ParaBlock) and b.role == "caption" and b.caption_target is not None:
                captions_for.setdefault(b.caption_target, []).append(b)
        for i, b in enumerate(blocks):
            if isinstance(b, TableBlock):
                self._close_lists()
                self._table(b, captions_for.get(i, []))
                continue
            role = b.role
            if role == "empty":
                continue
            if role not in ("list_item", "furniture"):
                self._close_lists()
            if role in ("toc", "toc_title"):
                self._toc(b)
            elif role == "furniture":
                label = "page_footer" if any(ch.isdigit() for ch in b.text) else "page_header"
                self.m.add_text(label, b.text, self.m.furniture, meta=self._base_meta(b))
            elif role in ("heading", "title"):
                self._heading(b, captions_for.get(i, []))
            elif role == "caption":
                if b.caption_target is None:
                    self._text(b, "caption")
                elif b.caption_target == i:            # image + légende dans le même paragraphe
                    self._pictures(b, self.parent(), [b], caption_inline=True)
            elif role == "list_item":
                self._list_item(b, captions_for.get(i, []))
            elif role == "formula":
                self._text(b, "formula")
            else:
                self._text(b, "text", captions_for.get(i, []))
        self._link_toc()

    # ------------------------------------------------------------ titres
    def _heading(self, b: ParaBlock, caps: list[ParaBlock]) -> None:
        level = 0 if b.role == "title" else (b.level or 1)
        if b.role == "title" and self._title_seen:
            level = 1
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        text = b.text
        meta = self._base_meta(b)
        meta.update({"heading_source": b.heading_source, "confidence": b.confidence})
        if b.list_marker and b.role == "heading":
            meta["number"] = b.list_marker.rstrip(".")
            text = f"{b.list_marker} {b.text}"
        if b.role == "title":
            self._title_seen = True
            it = self.m.add_text("title", text, self.parent(), meta=meta)
        else:
            it = self.m.add_text("section_header", text, self.parent(), level=level, meta=meta)
        for bm in b.bookmarks:
            self.bookmark_to_ref[bm] = it.self_ref
        self.stack.append((level, it))
        if b.images:
            self._pictures(b, it, caps)

    # ------------------------------------------------------------ texte
    def _text(self, b: ParaBlock, label: str, caps: list[ParaBlock] | None = None) -> TextItem | None:
        parent = self.parent()
        it = None
        if b.text:
            it = self.m.add_text(label, b.text, parent, meta=self._base_meta(b),
                                 hyperlink=b.hyperlinks[0] if b.hyperlinks else None)
            for bm in b.bookmarks:
                self.bookmark_to_ref.setdefault(bm, it.self_ref)
            self._footnotes(b, it)
        if b.images:
            self._pictures(b, parent, caps or [], para_ref=it.self_ref if it else None)
        return it

    def _footnotes(self, b: ParaBlock, it: TextItem) -> None:
        for kind, ids in (("footnote", b.footnote_ids), ("endnote", b.endnote_ids)):
            for fid in ids:
                key = f"{kind}:{fid}"
                txt = self.notes.get(kind, {}).get(fid)
                if txt and key not in self._notes_done:        # une note = un item, même si citée 2 fois
                    self._notes_done.add(key)
                    self.m.add_text("footnote", txt, it, meta={"note_id": fid, "note_kind": kind})

    # ------------------------------------------------------------ listes
    def _list_item(self, b: ParaBlock, caps: list[ParaBlock]) -> None:
        lvl = b.ilvl or 0
        while self.lists and self.lists[-1][0] > lvl:
            self.lists.pop()
        if not self.lists or self.lists[-1][0] < lvl:
            anchor = self.last_list_item.get(self.lists[-1][0]) if self.lists else None
            parent = anchor if anchor is not None else self.parent()
            grp = self.m.add_group(parent, "list", "list", meta={"order": b.order})
            self.lists.append((lvl, grp))
        grp = self.lists[-1][1]
        meta = self._base_meta(b)
        if b.num_id:
            meta["num_id"] = b.num_id
            meta["ilvl"] = lvl
        elif b.manual_marker:
            meta["manual_list"] = True
        it = self.m.add_text("list_item", b.text, grp, enumerated=b.list_enumerated,
                             marker=b.list_marker or "", meta=meta)
        self.last_list_item[lvl] = it
        self._footnotes(b, it)
        if b.images:
            self._pictures(b, it, caps, para_ref=it.self_ref)

    # ----------------------------------------------------------- images
    def _pictures(self, b: ParaBlock, parent: Item, caps: list[ParaBlock], *, para_ref: str | None = None,
                  caption_inline: bool = False) -> None:
        visible = [im for im in b.images if not im.decorative] or b.images
        for k, ref in enumerate(b.images):
            meta = self._base_meta(b)
            meta.update(_image_meta(ref))
            meta["heading_path"] = self.heading_path()
            if para_ref:
                meta["paragraph_ref"] = para_ref
            label = "chart" if ref.kind == "chart" else "picture"
            image = self.images.describe(ref) if ref.kind in ("image", "ole") and ref.target else None
            pic = self.m.add_picture(parent, label=label, image=image, meta=meta)
            # légendes : rattachées à la dernière image visible du paragraphe
            if caps and ref is visible[-1]:
                for c in caps:
                    cap = self.m.add_text("caption", c.text, pic, meta={"order": c.order, "style": c.style_name,
                                                                         "seq": c.seq_type})
                    pic.captions.append(cap.self_ref)

    # ---------------------------------------------------------- tableaux
    def _table(self, t: TableBlock, caps: list[ParaBlock], parent: Item | None = None,
               nested: bool = False) -> None:
        parent = parent or self.parent()
        meta = {"order": t.order, "page_hint": t.page_hint, "header_source": t.header_source}
        if t.style_name:
            meta["style"] = t.style_name
        if nested:
            meta["nested_in_cell"] = list(t.nested_in or ())
        else:
            meta["heading_path"] = self.heading_path()
        tab = self.m.add_table(parent, t.cells, t.num_rows, t.num_cols, meta=meta)
        for c in caps:
            cap = self.m.add_text("caption", c.text, tab, meta={"order": c.order, "style": c.style_name,
                                                                 "seq": c.seq_type})
            tab.captions.append(cap.self_ref)
        for cell in t.cells:
            for ref in cell.images:
                m = _image_meta(ref)
                m.update({"cell": [cell.start_row, cell.start_col], "table_ref": tab.self_ref,
                          "order": t.order, "page_hint": t.page_hint})
                image = self.images.describe(ref) if ref.kind in ("image", "ole") and ref.target else None
                self.m.add_picture(tab, label="chart" if ref.kind == "chart" else "picture",
                                   image=image, meta=m)
        for sub in t.nested:
            self._table(sub, [], parent=tab, nested=True)

    # --------------------------------------------------------- sommaire
    def _toc(self, b: ParaBlock) -> None:
        if self.toc_group is None:
            self.toc_group = self.m.add_group(self.parent(), "list", "table_of_contents", layer="furniture",
                                              meta={"order": b.order})
        if b.role == "toc_title":
            self.m.add_text("text", b.text, self.toc_group, meta={"role": "toc_title", "order": b.order})
            return
        it = self.m.add_text("list_item", b.text, self.toc_group, enumerated=False, marker="",
                             meta={"toc_level": b.toc_level, "page": b.toc_page, "anchors": b.anchors,
                                   "order": b.order})
        self.toc_items.append((it, b))

    def _link_toc(self) -> None:
        from .classify import norm_text
        by_text = {}
        for t in self.m.texts:
            if t.label == "section_header":
                by_text.setdefault(norm_text(t.text), t.self_ref)
        for it, b in self.toc_items:
            target = next((self.bookmark_to_ref[a] for a in b.anchors if a in self.bookmark_to_ref), None)
            it.meta["target_ref"] = target or by_text.get(norm_text(it.text))


def _image_meta(ref: ImageRef) -> dict:
    m = {"kind": ref.kind, "placement": ref.placement, "char_offset": ref.char_offset}
    for k in ("name", "descr", "title", "docpr_id", "position", "ole_prog_id", "chart_text"):
        v = getattr(ref, k)
        if v:
            m["alt_text" if k == "descr" else k] = v
    if ref.decorative:
        m["decorative"] = True
    if ref.target and not ref.external:
        m["part"] = ref.target
    return m
