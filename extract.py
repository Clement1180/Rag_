"""Étape 1 : extraction XML -> blocs intermédiaires, dans l'ordre de lecture.

Points clés anti-doublons traités ici :
  * mc:AlternateContent : seule la branche Choice est lue (le Fallback VML duplique
    systématiquement les zones de texte) ;
  * révisions : w:del / w:moveFrom / w:delText ignorés, w:ins / w:moveTo conservés ;
  * champs : seul le *résultat* est lu (jamais w:instrText) ;
  * w:sdt : seul w:sdtContent est lu (pas le texte d'invite de w:sdtPr) ;
  * tableaux fusionnés : une cellule logique par fusion (vMerge / gridSpan / hMerge).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterator

from lxml import etree

from .blocks import Block, CellData, ImageRef, ParaBlock, TableBlock
from .config import ParserConfig
from .ooxml import NS, DocxPackage, on_off, qn, wattr
from .styles import NumberingResolver, RunFormat, StyleResolver, normalize_bullet, parse_rpr

W_P, W_TBL, W_SDT, W_BODY = qn("w:p"), qn("w:tbl"), qn("w:sdt"), qn("w:body")
W_R, W_T, W_TAB, W_BR, W_CR = qn("w:r"), qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr")
W_FLDCHAR, W_INSTR = qn("w:fldChar"), qn("w:instrText")
MC_ALT, MC_CHOICE = qn("mc:AlternateContent"), qn("mc:Choice")
TRANSPARENT = {qn(t) for t in ("w:ins", "w:moveTo", "w:smartTag", "w:customXml", "w:dir", "w:bdo")}
SKIPPED = {qn(t) for t in ("w:del", "w:moveFrom", "w:pPr", "w:rPr", "w:proofErr", "w:permStart",
                            "w:permEnd", "w:commentRangeStart", "w:commentRangeEnd")}
EMU_PER_PX = 9525          # 96 dpi
EMU_PER_PT = 12700
SYM_MAP = {"F0B7": "•", "F0A7": "▪", "F0D8": "➢", "F0FC": "✓", "F0E0": "→", "F0B3": "≥", "F0A3": "≤"}


@dataclass(slots=True, frozen=True)
class Ctx:
    in_table: bool = False
    in_textbox: bool = False
    in_toc_sdt: bool = False

    def as_textbox(self) -> "Ctx":
        """Contexte des zones de texte, mis en cache (évite une copie par paragraphe)."""
        return _TEXTBOX_CTX.get(self) or _TEXTBOX_CTX.setdefault(self, replace(self, in_textbox=True))


_TEXTBOX_CTX: dict = {}


class _Field:
    __slots__ = ("instr", "result")

    def __init__(self):
        self.instr, self.result = "", False


_SPACES = re.compile(r"[ \u2002\u2003]+")
_MULTI_NL = re.compile(r"\n{2,}")
_WS = (" ", "\t", "\n", "\r", "\u00a0", "\u2009", "\u202f", "\u3000")


class _ParaAcc:
    """Accumulateur de texte et de statistiques de mise en forme d'un paragraphe."""
    __slots__ = ("parts", "chars", "bold", "italic", "under", "caps", "sizes", "block", "math_chars")

    def __init__(self, block: ParaBlock):
        self.parts: list[str] = []
        self.chars = self.bold = self.italic = self.under = self.caps = self.math_chars = 0
        self.sizes: dict[float, int] = {}
        self.block = block

    def add(self, text: str, fmt: RunFormat, math: bool = False) -> None:
        self.parts.append(text)
        n = len(text) - sum(map(text.count, _WS)) if not text.isalnum() else len(text)
        if not n:
            return
        self.chars += n
        if math:
            self.math_chars += n
        if fmt.bold:
            self.bold += n
        if fmt.italic:
            self.italic += n
        if fmt.underline:
            self.under += n
        if fmt.caps or (text.isupper() and any(c.isalpha() for c in text)):
            self.caps += n
        if fmt.size:
            self.sizes[fmt.size] = self.sizes.get(fmt.size, 0) + n


class Extractor:
    def __init__(self, pkg: DocxPackage, styles: StyleResolver, numbering: NumberingResolver,
                 cfg: ParserConfig, part: str | None = None):
        self.pkg, self.styles, self.numbering, self.cfg = pkg, styles, numbering, cfg
        self.part = part or pkg.main_part
        self.fields: list[_Field] = []          # pile de champs complexes (traverse les paragraphes)
        self.page = 1
        self._break_pending = False
        self._pending_bookmarks: list[str] = []
        self._order = 0
        self.warnings: list[str] = []

    # ================================================================ corps
    def iter_blocks(self) -> Iterator[Block]:
        for el in self._iter_body_elements():
            yield from self._block_element(el, Ctx())

    def _iter_body_elements(self) -> Iterator[etree._Element]:
        if self.cfg.streaming:
            for _ev, el in self.pkg.iterparse(self.part, events=("end",)):
                parent = el.getparent()
                if parent is None or parent.tag != W_BODY:
                    continue
                yield el
                el.clear(keep_tail=False)
                while el.getprevious() is not None:   # libère la mémoire au fil de l'eau
                    del parent[0]
        else:
            root = self.pkg.xml(self.part)
            body = root.find(qn("w:body")) if root is not None else None
            if body is not None:
                yield from list(body)

    def iter_part_blocks(self, part: str) -> Iterator[Block]:
        """Parties annexes (en-têtes, pieds, notes) : racine = conteneur de blocs."""
        root = self.pkg.xml(part)
        if root is not None:
            for el in root:
                yield from self._block_element(el, Ctx())

    def _block_element(self, el: etree._Element, ctx: Ctx) -> Iterator[Block]:
        tag = el.tag
        if tag == W_P:
            yield from self._paragraph(el, ctx)
        elif tag == W_TBL:
            yield from self._table(el, ctx)
        elif tag == W_SDT:
            gal = el.find(f"{qn('w:sdtPr')}/{qn('w:docPartObj')}/{qn('w:docPartGallery')}")
            is_toc = gal is not None and "table of contents" in (wattr(gal, "val") or "").lower()
            content = el.find(qn("w:sdtContent"))
            if content is not None:
                sub = replace(ctx, in_toc_sdt=ctx.in_toc_sdt or is_toc)
                for child in content:
                    yield from self._block_element(child, sub)
        elif tag == qn("w:customXml") or tag in TRANSPARENT:
            for child in el:
                yield from self._block_element(child, ctx)
        elif tag == qn("w:bookmarkStart"):
            name = wattr(el, "name")
            if name:
                self._pending_bookmarks.append(name)
        elif tag == MC_ALT:
            choice = el.find(MC_CHOICE)
            for child in (choice if choice is not None else []):
                yield from self._block_element(child, ctx)
        elif tag == qn("w:altChunk"):
            self.warnings.append("w:altChunk ignoré (contenu importé non converti)")

    def _next_order(self) -> int:
        self._order += 1
        return self._order

    # ============================================================ paragraphe
    def _paragraph(self, p: etree._Element, ctx: Ctx) -> Iterator[Block]:
        ppr = p.find(qn("w:pPr"))
        sid = None
        direct_outline = direct_num = direct_ilvl = jc = None
        indent = 0
        page_break_before = section_break = False
        mark_deleted = False
        if ppr is not None:
            ps = ppr.find(qn("w:pStyle"))
            sid = wattr(ps, "val") if ps is not None else None
            ol = ppr.find(qn("w:outlineLvl"))
            if ol is not None:
                try:
                    direct_outline = int(wattr(ol, "val"))
                except (TypeError, ValueError):
                    pass
            numpr = ppr.find(qn("w:numPr"))
            if numpr is not None:
                n = numpr.find(qn("w:numId"))
                lv = numpr.find(qn("w:ilvl"))
                direct_num = wattr(n, "val") if n is not None else None
                direct_ilvl = int(wattr(lv, "val") or 0) if lv is not None else None
            j = ppr.find(qn("w:jc"))
            jc = wattr(j, "val") if j is not None else None
            ind = ppr.find(qn("w:ind"))
            if ind is not None:
                try:
                    indent = int(wattr(ind, "left") or wattr(ind, "start") or 0)
                except ValueError:
                    indent = 0
            page_break_before = bool(on_off(ppr.find(qn("w:pageBreakBefore"))))
            sect = ppr.find(qn("w:sectPr"))
            if sect is not None:
                t = sect.find(qn("w:type"))
                section_break = (wattr(t, "val") if t is not None else "nextPage") in ("nextPage", "oddPage", "evenPage")
            mark_deleted = ppr.find(f"{qn('w:rPr')}/{qn('w:del')}") is not None

        rs = self.styles.paragraph(sid)
        outline = direct_outline if direct_outline is not None else rs.outline_lvl
        if outline is not None and not 0 <= outline <= 8:
            outline = None
        num_id = direct_num if direct_num is not None else rs.num_id
        ilvl = direct_ilvl if direct_ilvl is not None else rs.ilvl
        if num_id in (None, "0"):
            num_id = None
        elif ilvl is None:
            ilvl = 0
            for k in range(9):  # numérotation portée par le style : niveau lié au pStyle
                lv = self.numbering.level(num_id, k)
                if lv and lv.p_style and lv.p_style == rs.style_id:
                    ilvl = k
                    break

        if page_break_before:
            self._page_break()
        block = ParaBlock(text="", style_id=rs.style_id, style_name=rs.name,
                          style_names_chain=rs.names_chain, outline_lvl=outline,
                          num_id=num_id, ilvl=ilvl, jc=jc or rs.jc, indent_left=indent,
                          in_table=ctx.in_table, in_textbox=ctx.in_textbox,
                          in_toc_sdt=ctx.in_toc_sdt, page_break_before=page_break_before,
                          page_hint=self.page)
        if self._pending_bookmarks:
            block.bookmarks.extend(self._pending_bookmarks)
            self._pending_bookmarks.clear()
        acc = _ParaAcc(block)
        self._walk_inline(p, acc, rs.run, ctx.as_textbox())

        text = "".join(acc.parts)
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        block.text = _MULTI_NL.sub("\n", _SPACES.sub(" ", text).strip(" ")).strip()
        if acc.chars:
            block.bold_ratio = acc.bold / acc.chars
            block.italic_ratio = acc.italic / acc.chars
            block.underline_ratio = acc.under / acc.chars
            block.caps = acc.caps / acc.chars > 0.9
            block.is_math = acc.math_chars / acc.chars > 0.9
        block.size = max(acc.sizes, key=acc.sizes.get) if acc.sizes else rs.run.size
        if num_id is not None and (block.text or block.images):
            marker, lvl = self.numbering.next_marker(num_id, ilvl or 0)
            if lvl is not None:
                block.list_marker = marker
                block.list_enumerated = not lvl.is_bullet
                block.list_lvl_text = lvl.lvl_text
        if mark_deleted and not block.text and not block.images:
            return  # paragraphe supprimé en révision
        if section_break:
            self._page_break()
            block.page_break_after = True
        block.order = self._next_order()
        yield block
        # Zones de texte : émises juste après leur paragraphe d'ancrage (ordre de lecture)
        seen: set[str] = set()
        for tb in block.textboxes:
            key = "\n".join(b.text for b in tb if isinstance(b, ParaBlock))
            if key in seen:
                continue
            seen.add(key)
            for b in tb:
                b.order = self._next_order()
                yield b
        block.textboxes = []

    def _page_break(self) -> None:
        self.page += 1
        self._break_pending = True

    # ---------------------------------------------------------- inline
    def _field_hidden(self) -> bool:
        return any(not f.result for f in self.fields)

    def _in_toc_field(self) -> bool:
        return any(f.instr.lstrip().upper().startswith("TOC") for f in self.fields)

    def _emit(self, acc: _ParaAcc, text: str, fmt: RunFormat, math: bool = False) -> None:
        if not text or self._field_hidden() or fmt.vanish:
            return
        if self._in_toc_field():
            acc.block.toc_field = True
        if text.strip():
            self._break_pending = False
        acc.add(text, fmt, math)

    def _walk_inline(self, parent: etree._Element, acc: _ParaAcc, base: RunFormat, ctx: Ctx) -> None:
        for el in parent:
            tag = el.tag
            if tag == W_R:
                self._run(el, acc, base, ctx)
            elif tag in TRANSPARENT:
                self._walk_inline(el, acc, base, ctx)
            elif tag in SKIPPED:
                continue
            elif tag == qn("w:hyperlink"):
                anchor = wattr(el, "anchor")
                if anchor:
                    acc.block.anchors.append(anchor)
                rid = el.get(qn("r:id"))
                if rid:
                    rel = self.pkg.rels(self.part).get(rid)
                    if rel is not None and rel.external:
                        acc.block.hyperlinks.append(rel.target)
                self._walk_inline(el, acc, base, ctx)
            elif tag == qn("w:fldSimple"):
                instr = (wattr(el, "instr") or "").strip()
                self._note_field(instr, acc)
                f = _Field()
                f.instr, f.result = instr, True
                self.fields.append(f)
                self._walk_inline(el, acc, base, ctx)
                self.fields.pop()
            elif tag == W_SDT:
                content = el.find(qn("w:sdtContent"))
                if content is not None:
                    self._walk_inline(content, acc, base, ctx)
            elif tag == qn("w:bookmarkStart"):
                name = wattr(el, "name")
                if name:
                    acc.block.bookmarks.append(name)
            elif tag == MC_ALT:
                choice = el.find(MC_CHOICE)
                if choice is not None:
                    self._walk_inline(choice, acc, base, ctx)
            elif tag in (qn("m:oMathPara"), qn("m:oMath")):
                txt = "".join(t.text or "" for t in el.iter(qn("m:t")))
                self._emit(acc, txt, base, math=True)

    def _note_field(self, instr: str, acc: _ParaAcc) -> None:
        m = re.match(r"\s*SEQ\s+(\S+)", instr, re.I)
        if m:
            acc.block.seq_type = m.group(1).strip('"').lower()
            return
        m = re.match(r"\s*PAGEREF\s+(\S+)", instr, re.I)   # sommaire sans lien hypertexte (\\h absent)
        if m and m.group(1) not in acc.block.anchors:
            acc.block.anchors.append(m.group(1))

    def _run(self, r: etree._Element, acc: _ParaAcc, base: RunFormat, ctx: Ctx) -> None:
        rpr = r.find(qn("w:rPr"))
        fmt = base
        if rpr is not None:
            rst = rpr.find(qn("w:rStyle"))
            if rst is not None:
                fmt = fmt.merged(self.styles.character(wattr(rst, "val")))
            fmt = fmt.merged(parse_rpr(rpr))
        for el in r:
            tag = el.tag
            if tag == W_T:
                self._emit(acc, el.text or "", fmt)
            elif tag == W_TAB or tag == qn("w:ptab"):
                self._emit(acc, "\t", fmt)
            elif tag == W_BR:
                if wattr(el, "type") == "page":
                    self._page_break()
                    acc.block.page_break_after = True
                else:
                    self._emit(acc, "\n", fmt)
            elif tag == W_CR:
                self._emit(acc, "\n", fmt)
            elif tag == W_FLDCHAR:
                kind = wattr(el, "fldCharType")
                if kind == "begin":
                    self.fields.append(_Field())
                elif kind == "separate" and self.fields:
                    self.fields[-1].result = True
                    self._note_field(self.fields[-1].instr, acc)
                elif kind == "end" and self.fields:
                    f = self.fields.pop()
                    if not f.result:
                        self._note_field(f.instr, acc)
            elif tag == W_INSTR:
                if self.fields:
                    self.fields[-1].instr += el.text or ""
            elif tag == qn("w:noBreakHyphen"):
                self._emit(acc, "-", fmt)
            elif tag == qn("w:sym"):
                code = (wattr(el, "char") or "").upper()
                self._emit(acc, SYM_MAP.get(code, ""), fmt)
            elif tag == qn("w:lastRenderedPageBreak"):
                if self._break_pending:
                    self._break_pending = False    # déjà compté via un saut explicite
                else:
                    self.page += 1
            elif tag in (qn("w:footnoteReference"), qn("w:endnoteReference")):
                fid = wattr(el, "id")
                if fid:
                    (acc.block.footnote_ids if "footnote" in tag else acc.block.endnote_ids).append(fid)
            elif tag == qn("w:drawing"):
                self._drawing(el, acc, ctx)
            elif tag == qn("w:pict"):
                self._vml(el, acc, ctx, placement="vml")
            elif tag == qn("w:object"):
                self._vml(el, acc, ctx, placement="ole")
            elif tag == MC_ALT:
                choice = el.find(MC_CHOICE)
                if choice is not None:
                    fake = etree.Element(W_R)   # contenu de run encapsulé
                    for child in choice:
                        fake.append(child)
                    self._run(fake, acc, fmt, ctx)
            elif tag == qn("m:oMath"):
                self._emit(acc, "".join(t.text or "" for t in el.iter(qn("m:t"))), fmt, math=True)

    # ---------------------------------------------------------- images
    def _offset(self, acc: _ParaAcc) -> int:
        return sum(len(p) for p in acc.parts)

    def _rel(self, rid: str | None):
        return self.pkg.rels(self.part).get(rid) if rid else None

    def _drawing(self, d: etree._Element, acc: _ParaAcc, ctx: Ctx) -> None:
        for cont in d:
            placement = etree.QName(cont).localname  # inline | anchor
            ext = cont.find(qn("wp:extent"))
            extent = None
            if ext is not None:
                try:
                    extent = (int(ext.get("cx", 0)), int(ext.get("cy", 0)))
                except ValueError:
                    extent = None
            docpr = cont.find(qn("wp:docPr"))
            name = descr = title = dpid = None
            decorative = False
            if docpr is not None:
                name, descr, title, dpid = docpr.get("name"), docpr.get("descr"), docpr.get("title"), docpr.get("id")
                dec = docpr.find(f".//{qn('adec:decorative')}")
                decorative = dec is not None and dec.get("val") in ("1", "true")
            position = None
            if placement == "anchor":
                position = {"wrap": next((etree.QName(c).localname for c in cont
                                          if etree.QName(c).localname.startswith("wrap")), None),
                            "behind_doc": cont.get("behindDoc") == "1"}
                for axis in ("positionH", "positionV"):
                    pe = cont.find(qn(f"wp:{axis}"))
                    if pe is not None:
                        off = pe.find(qn("wp:posOffset"))
                        al = pe.find(qn("wp:align"))
                        position[axis] = {"relative_from": pe.get("relativeFrom"),
                                          "offset_emu": int(off.text) if off is not None and off.text and off.text.lstrip("-").isdigit() else None,
                                          "align": al.text if al is not None else None}
            gd = cont.find(f"{qn('a:graphic')}/{qn('a:graphicData')}")
            if gd is None:
                continue
            common = dict(name=name, descr=descr, title=title, docpr_id=dpid, placement=placement,
                          extent_emu=extent, position=position, decorative=decorative,
                          char_offset=self._offset(acc))
            found = False
            for blip in gd.iter(qn("a:blip")):
                rid = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
                rel = self._rel(rid)
                acc.block.images.append(ImageRef(
                    kind="image", rel_id=rid, target=rel.target if rel else None,
                    external=bool(rel and rel.external),
                    mimetype=None if rel is None or rel.external else self.pkg.content_type(rel.target),
                    **common))
                found = True
            chart = gd.find(qn("c:chart"))
            if chart is not None:
                rel = self._rel(chart.get(qn("r:id")))
                acc.block.images.append(ImageRef(kind="chart", rel_id=chart.get(qn("r:id")),
                                                 target=rel.target if rel else None,
                                                 chart_text=self._part_text(rel.target) if rel else None,
                                                 **common))
                found = True
            dgm = gd.find(qn("dgm:relIds"))
            if dgm is not None:
                rel = self._rel(dgm.get(qn("r:dm")))
                acc.block.images.append(ImageRef(kind="smartart", rel_id=dgm.get(qn("r:dm")),
                                                 target=rel.target if rel else None,
                                                 chart_text=self._part_text(rel.target) if rel else None,
                                                 **common))
                found = True
            if self.cfg.include_textboxes:
                for txbx in gd.iter(qn("w:txbxContent")):
                    self._textbox(txbx, acc, ctx)
                    found = True

    def _part_text(self, part: str, limit: int = 4000) -> str | None:
        """Texte en cache d'un graphique / SmartArt (titres, catégories, valeurs)."""
        try:
            root = self.pkg.xml(part)
        except Exception:  # partie corrompue : on n'échoue pas tout le document
            return None
        if root is None:
            return None
        vals, total = [], 0
        for el in root.iter(qn("a:t"), qn("c:v")):
            t = (el.text or "").strip()
            if t:
                vals.append(t)
                total += len(t)
                if total > limit:
                    break
        return " | ".join(vals) or None

    def _vml(self, el: etree._Element, acc: _ParaAcc, ctx: Ctx, placement: str) -> None:
        prog = None
        ole = el.find(f".//{qn('o:OLEObject')}")
        if ole is not None:
            prog = ole.get("ProgID")
        for shape in el.iter(qn("v:shape"), qn("v:rect"), qn("v:image")):
            imd = shape.find(qn("v:imagedata"))
            if imd is not None:
                rid = imd.get(qn("r:id")) or imd.get(qn("o:relid")) or imd.get(qn("r:pict"))
                rel = self._rel(rid)
                extent = _vml_extent(shape.get("style", ""))
                acc.block.images.append(ImageRef(
                    kind="ole" if placement == "ole" else "image", rel_id=rid,
                    target=rel.target if rel else None, external=bool(rel and rel.external),
                    mimetype=None if rel is None or rel.external else self.pkg.content_type(rel.target),
                    name=shape.get("id"), descr=shape.get("alt") or imd.get(qn("o:title")),
                    title=imd.get(qn("o:title")), placement=placement, extent_emu=extent,
                    char_offset=self._offset(acc), ole_prog_id=prog))
        if self.cfg.include_textboxes:
            for txbx in el.iter(qn("w:txbxContent")):
                self._textbox(txbx, acc, ctx)

    def _textbox(self, txbx: etree._Element, acc: _ParaAcc, ctx: Ctx) -> None:
        blocks: list[Block] = []
        for child in txbx:
            blocks.extend(self._block_element(child, ctx))
        blocks = [b for b in blocks if not isinstance(b, ParaBlock) or b.text or b.images]
        if blocks:
            acc.block.textboxes.append(blocks)

    # ============================================================= tableaux
    @staticmethod
    def _children(el: etree._Element, tag: str) -> Iterator[etree._Element]:
        """Enfants `tag`, en traversant les enveloppes sdt/customXml (fréquentes dans les modèles)."""
        for c in el:
            if c.tag == tag:
                yield c
            elif c.tag == W_SDT:
                sc = c.find(qn("w:sdtContent"))
                if sc is not None:
                    yield from Extractor._children(sc, tag)
            elif c.tag == qn("w:customXml") or c.tag in TRANSPARENT:
                yield from Extractor._children(c, tag)

    def _table(self, tbl: etree._Element, ctx: Ctx) -> Iterator[Block]:
        rows = list(self._children(tbl, qn("w:tr")))
        if self.cfg.unwrap_layout_tables and len(rows) == 1:
            cells = list(self._children(rows[0], qn("w:tc")))
            if len(cells) == 1:  # tableau 1x1 = cadre de mise en page (très fréquent en .doc)
                for child in cells[0]:
                    yield from self._block_element(child, ctx)
                return
        yield self._parse_table(tbl, rows, ctx)

    def _parse_table(self, tbl: etree._Element, rows: list[etree._Element], ctx: Ctx) -> TableBlock:
        tpr = tbl.find(qn("w:tblPr"))
        style_name = None
        first_row_look = False
        if tpr is not None:
            ts = tpr.find(qn("w:tblStyle"))
            style_name = self.styles.name_of(wattr(ts, "val")) if ts is not None else None
            look = tpr.find(qn("w:tblLook"))
            if look is not None:
                fr = wattr(look, "firstRow")
                if fr is None and wattr(look, "val"):
                    try:
                        fr = "1" if int(wattr(look, "val"), 16) & 0x0020 else "0"
                    except ValueError:
                        fr = None
                first_row_look = fr in ("1", "true")
        grid = tbl.find(qn("w:tblGrid"))
        grid_cols = len(grid.findall(qn("w:gridCol"))) if grid is not None else 0
        page = self.page
        cells: list[CellData] = []
        col_owner: dict[int, CellData] = {}     # colonne -> dernière cellule qui la couvre
        nested: list[TableBlock] = []
        header_rows: list[int] = []
        max_col = 0
        cell_ctx = replace(ctx, in_table=True)
        for r, tr in enumerate(rows):
            trpr = tr.find(qn("w:trPr"))
            col = 0
            if trpr is not None:
                gb = trpr.find(qn("w:gridBefore"))
                col = int(wattr(gb, "val") or 0) if gb is not None else 0
                if trpr.find(qn("w:tblHeader")) is not None and on_off(trpr.find(qn("w:tblHeader"))):
                    header_rows.append(r)
            prev_in_row: CellData | None = None
            for tc in self._children(tr, qn("w:tc")):
                tcpr = tc.find(qn("w:tcPr"))
                span, vmerge, hmerge = 1, None, None
                if tcpr is not None:
                    gs = tcpr.find(qn("w:gridSpan"))
                    if gs is not None:
                        span = max(1, int(wattr(gs, "val") or 1))
                    vm = tcpr.find(qn("w:vMerge"))
                    if vm is not None:
                        vmerge = wattr(vm, "val") or "continue"
                    hm = tcpr.find(qn("w:hMerge"))
                    if hm is not None:
                        hmerge = wattr(hm, "val") or "continue"
                if vmerge == "continue":
                    owner = col_owner.get(col)
                    if owner is not None and owner.start_col == col:
                        owner.row_span = r - owner.start_row + 1
                        col += span
                        prev_in_row = owner
                        continue
                if hmerge == "continue" and prev_in_row is not None and prev_in_row.start_row == r:
                    prev_in_row.col_span += span
                    for c in range(col, col + span):
                        col_owner[c] = prev_in_row
                    col += span
                    continue
                texts, images, bold_chars, all_chars = [], [], 0.0, 0
                for child in tc:
                    for b in self._block_element(child, cell_ctx):
                        if isinstance(b, TableBlock):
                            b.nested_in = (r, col)
                            nested.append(b)
                            texts.append(table_to_text(b))
                        else:
                            t = b.text
                            if b.list_marker and t:
                                t = f"{b.list_marker} {t}"
                            if t:
                                texts.append(t)
                                bold_chars += b.bold_ratio * len(t)
                                all_chars += len(t)
                            images.extend(b.images)
                cell = CellData(start_row=r, start_col=col, row_span=1, col_span=span,
                                text="\n".join(texts), images=images,
                                bold=all_chars > 0 and bold_chars / all_chars > 0.9)
                cells.append(cell)
                for c in range(col, col + span):
                    col_owner[c] = cell
                prev_in_row = cell
                col += span
            max_col = max(max_col, col)
        num_rows = len(rows)
        num_cols = max(grid_cols if max_col == 0 else 0, max_col)
        header_source = None
        if header_rows and header_rows[0] == 0:
            hdr = set(header_rows)
            header_source = "tblHeader"
        else:
            first = [c for c in cells if c.start_row == 0]
            rest = [c for c in cells if c.start_row > 0 and c.text]
            first_bold = bool(first) and all(c.bold or not c.text for c in first) and any(c.text for c in first)
            rest_bold = bool(rest) and all(c.bold for c in rest)
            hdr = {0} if num_rows > 1 and first_bold and not rest_bold else set()
            if hdr:
                header_source = "bold_first_row"
            elif num_rows > 1 and first_row_look and all(c.text for c in first):
                hdr, header_source = {0}, "tblLook"
        for c in cells:
            if c.start_row in hdr:
                c.column_header = True
        return TableBlock(cells=cells, num_rows=num_rows, num_cols=num_cols, style_name=style_name,
                          nested=nested, header_source=header_source,
                          in_textbox=ctx.in_textbox, page_hint=page, order=self._next_order())


def _vml_extent(style: str) -> tuple[int, int] | None:
    vals = {}
    for part in style.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            vals[k.strip().lower()] = v.strip()

    def to_emu(v: str | None) -> int | None:
        if not v:
            return None
        m = re.match(r"([\d.]+)\s*(pt|in|cm|mm|px)?", v)
        if not m:
            return None
        n, unit = float(m.group(1)), m.group(2) or "px"
        factor = {"pt": EMU_PER_PT, "in": 914400, "cm": 360000, "mm": 36000, "px": EMU_PER_PX}[unit]
        return int(n * factor)
    w, h = to_emu(vals.get("width")), to_emu(vals.get("height"))
    return (w, h) if w and h else None


def table_to_text(t: TableBlock) -> str:
    """Rendu texte compact (utilisé pour les tableaux imbriqués dans une cellule)."""
    rows: dict[int, list[str]] = {}
    for c in sorted(t.cells, key=lambda c: (c.start_row, c.start_col)):
        rows.setdefault(c.start_row, []).append(c.text.replace("\n", " "))
    return "\n".join(" | ".join(v) for _, v in sorted(rows.items()))
