"""Étape 2 : attribution d'un rôle à chaque paragraphe.

Ordre de priorité (le premier qui s'applique gagne) :
  vide -> sommaire -> mobilier répété -> légende -> formule -> titre (règles de headings)
  -> élément de liste (numPr) -> liste manuelle -> texte
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from .blocks import Block, ParaBlock, TableBlock
from .config import HeadingConfig, ParserConfig

# --------------------------------------------------------------- motifs
HEADING_STYLE_RE = re.compile(
    r"^(?:heading|titre|title|level|niveau|lvl|niv|[üu]berschrift|titolo|t[ií]tulo|kop|rubrik|"
    r"nag[łl][óo]wek|otsikko|overskrift|заголовок|h)\s*[_\-.]?\s*([1-9])\b", re.I)
HEADING_KEYWORD_RE = re.compile(r"\b(heading|titre|title|chapitre|chapter|section|partie|part|"
                                r"rubrique|intertitre|[üu]berschrift)\b", re.I)
TITLE_STYLE_NAMES = {"title", "titre", "titel", "titolo", "título", "titulo", "document title"}
NOT_HEADING_STYLES = re.compile(r"\b(toc|tm|caption|l[ée]gende|footer|header|pied|en-t[êe]te|"
                                r"list|liste|table|quote|citation|subtitle|sous-titre|body|corps|normal)\b", re.I)
TOC_STYLE_RE = re.compile(r"^(?:toc|tm|contents|table des mati[èe]res|sommaire|"
                          r"verzeichnis|inhaltsverzeichnis)\s*([1-9])$", re.I)
TOC_HEADING_STYLE_RE = re.compile(r"^(toc heading|en-t[êe]te de table des mati[èe]res|"
                                  r"contents heading|inhaltsverzeichnisüberschrift)$", re.I)
TOC_TITLE_TEXT_RE = re.compile(r"^(sommaire|table des mati[èe]res|contents|table of contents|"
                               r"inhalt(sverzeichnis)?|indice|índice|summary of contents)\s*:?$", re.I)
TOC_LINE_RE = re.compile(r"^(?P<title>.*?\S)\s*(?:\t+|\.{3,}|…+|[._·]{3,}|\s{3,})[\s._·…]*"
                         r"(?P<page>\d{1,4}|[ivxlcdm]{1,7})\s*$", re.I)
TOC_PAGE_TAIL_RE = re.compile(r"^(?P<title>.*?\S)[\s._·…]*\t[\s._·…]*(?P<page>\d{1,4}|[ivxlcdm]{1,7})\s*$", re.I)
CAPTION_STYLE_RE = re.compile(r"^(caption|l[ée]gende|beschriftung|didascalia|ep[íi]grafe|leyenda|"
                              r"bijschrift|figure|table|illustration|drawing|image caption|"
                              r"table caption|figure caption)$", re.I)
CAPTION_TEXT_RE = re.compile(
    r"^(?P<kind>figure|fig\.?|tableau|table|tab\.|illustration|image|graphique|graphe|sch[ée]ma|photo|"
    r"diagramme|carte|abbildung|abb\.|tabelle|chart|exhibit|planche)\s*(?:n[°o]\.?\s*)?"
    r"(?P<num>[\dIVXivx]+(?:[.\-–]\d+)*)\s*(?:[:.\-–—]|$|\s)", re.I)
TABLE_WORDS = {"tableau", "table", "tab", "tab.", "tabelle"}
TEXT_NUMBER_RE = re.compile(r"^(?P<num>(?:\d{1,3})(?:[.\-]\d{1,3}){0,5})[.)]?\s+(?=\S)")
ROMAN_RE = re.compile(r"^(?P<num>[IVXLC]{1,6})[.)\-–]\s+(?=\S)")
LETTER_RE = re.compile(r"^(?P<num>[A-Z])[.)]\s+(?=\S)")
CHAPTER_RE = re.compile(r"^(chapitre|chapter|partie|part|section|titre|livre|annexe|appendix|"
                        r"article|kapitel|teil)\s+([\dIVXLC]+|premier|unique|[A-Z])\b", re.I)
PAGE_FURNITURE_RE = re.compile(r"^(?:page\s*\d+(?:\s*(?:/|sur|of|de|von)\s*\d+)?|\d+\s*(?:/|sur|of)\s*\d+|"
                               r"[-–—]\s*\d+\s*[-–—])$", re.I)
MANUAL_BULLET_RE = re.compile(r"^(?P<m>[•●▪■◦○➢►✓✔❖→·*\-–—]|o(?=\s)|[\uf000-\uf0ff])\s*\t?\s*(?=\S)")
# « 1) », « (a) », « iv) », mais aussi « 1. » / « a. » (sans « 1.2 » : ce cas relève
# des titres numérotés, testés AVANT les listes et exigeant une mise en emphase).
MANUAL_ENUM_RE = re.compile(r"^(?P<m>\(?[a-z]\)|\(?\d{1,2}\)|[ivx]{1,4}\)|\d{1,2}\.(?!\d)|[a-z]\.(?![a-z]\.))\s+(?=\S)")
TERMINAL_PUNCT = tuple(".;,!?")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = TEXT_NUMBER_RE.sub("", s.strip())
    s = ROMAN_RE.sub("", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# ===================================================================== API
def classify(blocks: list[Block], cfg: ParserConfig) -> dict:
    paras = [b for b in blocks if isinstance(b, ParaBlock)]
    for p in paras:
        if not p.text and not p.images:
            p.role = "empty"
    toc_info = _detect_toc(blocks, cfg)
    if cfg.detect_repeated_furniture:
        _detect_repeated_furniture(blocks, cfg)
    _detect_captions(blocks)
    for p in paras:
        if p.role == "text" and p.is_math and len(p.text) < 500:
            p.role = "formula"
    stats = _detect_headings(blocks, cfg.headings, toc_info)
    _detect_lists(blocks, cfg)
    for p in paras:  # tabulations : utiles au sommaire, bruit ailleurs
        if p.role not in ("toc",):
            p.text = re.sub(r"[ \t]+", " ", p.text).strip()
    return {"toc": toc_info, **stats}


# ================================================================ sommaire
def _toc_split(text: str) -> tuple[str, str | None]:
    m = TOC_PAGE_TAIL_RE.match(text) or TOC_LINE_RE.match(text)
    if m:
        return m.group("title").strip(" \t.·…_"), m.group("page")
    return text.strip(), None


def _detect_toc(blocks: list[Block], cfg: ParserConfig) -> dict:
    entries: list[ParaBlock] = []
    n = len(blocks)
    for i, b in enumerate(blocks):
        if not isinstance(b, ParaBlock) or b.role == "empty":
            continue
        style = b.style_names_chain[0] if b.style_names_chain else ""
        m = TOC_STYLE_RE.match(style) or TOC_STYLE_RE.match((b.style_id or "").lower())
        if TOC_HEADING_STYLE_RE.match(style) or (b.in_toc_sdt and TOC_TITLE_TEXT_RE.match(b.text)):
            b.role = "toc_title"
        elif b.in_toc_sdt or b.toc_field or m:
            b.role = "toc"
            b.toc_level = int(m.group(1)) if m else None
            entries.append(b)
    # Sommaire « texte » (DOC converti, champ perdu) : >= 3 lignes consécutives « titre ..... 12 »
    if cfg.detect_text_toc and not entries:
        run: list[ParaBlock] = []
        for b in blocks + [None]:
            ok = (isinstance(b, ParaBlock) and b.role in ("text",) and not b.in_table
                  and len(b.text) < 250 and TOC_LINE_RE.match(b.text) is not None)
            if ok:
                run.append(b)
                continue
            if isinstance(b, ParaBlock) and b.role == "empty":
                continue
            if len(run) >= 3:
                for e in run:
                    e.role = "toc"
                entries.extend(run)
            run = []
    if not entries:
        return {"entries": [], "last_order": -1}
    first = entries[0]
    idx = blocks.index(first)
    for j in range(idx - 1, max(-1, idx - 4), -1):   # titre du sommaire juste avant
        b = blocks[j]
        if isinstance(b, ParaBlock) and b.role != "empty":
            if TOC_TITLE_TEXT_RE.match(b.text.strip()):
                b.role = "toc_title"
            break
    # niveaux : style tocN > indentation > profondeur de numérotation
    indents = sorted({e.indent_left for e in entries})
    for e in entries:
        title, page = _toc_split(e.text)
        e.text, e.toc_page = title, page
        if e.toc_level is None:
            m = TEXT_NUMBER_RE.match(title)
            if len(indents) > 1:
                e.toc_level = indents.index(e.indent_left) + 1
            elif m:
                e.toc_level = m.group("num").count(".") + m.group("num").count("-") + 1
            else:
                e.toc_level = 1
    return {"entries": entries, "last_order": max(e.order for e in entries)}


# ============================================================== mobilier
def _detect_repeated_furniture(blocks: list[Block], cfg: ParserConfig) -> None:
    occurrences: dict[str, list[int]] = defaultdict(list)
    for i, b in enumerate(blocks):
        if isinstance(b, ParaBlock) and b.role == "text" and not b.in_table:
            if PAGE_FURNITURE_RE.match(b.text.strip()):
                b.role = "furniture"
            elif len(b.text) <= 100:
                occurrences[norm_text(b.text)].append(i)
    for key, idxs in occurrences.items():
        if not key or len(idxs) < cfg.repeated_min_count:
            continue
        near_break = 0
        for i in idxs:
            window = [blocks[j] for j in range(max(0, i - 2), min(len(blocks), i + 3))]
            if any(isinstance(w, ParaBlock) and (w.page_break_before or w.page_break_after) for w in window):
                near_break += 1
        if near_break / len(idxs) >= 0.5:     # répété ET collé aux sauts de page => en-tête/pied injecté
            for i in idxs:
                blocks[i].role = "furniture"


# =============================================================== légendes
def _is_picture_block(b: Block) -> bool:
    return isinstance(b, ParaBlock) and any(not im.decorative for im in b.images)


def _neighbors(blocks: list[Block], i: int, step: int):
    """Premier bloc non vide avant/après i."""
    j = i + step
    while 0 <= j < len(blocks):
        b = blocks[j]
        if isinstance(b, ParaBlock) and b.role == "empty":
            j += step
            continue
        return j
    return None


def _detect_captions(blocks: list[Block]) -> None:
    taken: set[int] = set()
    for i, b in enumerate(blocks):
        if not isinstance(b, ParaBlock) or b.role != "text" or len(b.text) > 400:
            continue
        style = b.style_names_chain[0] if b.style_names_chain else ""
        by_style = bool(CAPTION_STYLE_RE.match(style)) or any(CAPTION_STYLE_RE.match(s) for s in b.style_names_chain[1:2])
        m = CAPTION_TEXT_RE.match(b.text)
        if not (by_style or b.seq_type or m):
            continue
        kind_word = (b.seq_type or (m.group("kind") if m else "") or "").lower()
        want = "table" if kind_word in TABLE_WORDS or kind_word.startswith("tab") else ("picture" if kind_word else None)
        if b.images and b.text:             # image + légende dans le même paragraphe
            b.role, b.caption_target = "caption", i
            continue
        prev_i, next_i = _neighbors(blocks, i, -1), _neighbors(blocks, i, +1)

        def fits(j):
            if j is None or j in taken:
                return False
            t = blocks[j]
            if want == "table":
                return isinstance(t, TableBlock)
            if want == "picture":
                return _is_picture_block(t)
            return isinstance(t, TableBlock) or _is_picture_block(t)
        # convention : légende de tableau au-dessus, de figure en dessous
        order = (next_i, prev_i) if want == "table" else (prev_i, next_i)
        target = next((j for j in order if fits(j)), None)
        if target is None and (by_style or b.seq_type):
            target = next((j for j in order if j is not None and j not in taken and
                           (isinstance(blocks[j], TableBlock) or _is_picture_block(blocks[j]))), None)
        if target is not None:
            b.role, b.caption_target = "caption", target
            taken.add(target)
        elif by_style or b.seq_type:
            b.role = "caption"                 # légende orpheline : gardée comme légende sans cible


# ================================================================== titres
def _emphasis(p: ParaBlock, body: float, hc: HeadingConfig) -> bool:
    big = p.size is not None and p.size >= body * hc.size_ratio
    return (p.bold_ratio >= 0.9 or big or p.underline_ratio >= 0.9
            or (p.caps and len(p.text) >= 4 and (p.bold_ratio >= 0.5 or big or p.jc == "center")))


def _shape_ok(p: ParaBlock, hc: HeadingConfig) -> bool:
    t = p.text.strip()
    return (1 < len(t) <= hc.max_chars and len(t.split()) <= hc.max_words
            and not t.endswith(TERMINAL_PUNCT) and not t.endswith(":")
            and any(c.isalpha() for c in t) and "\n" not in t)


def _signature(p: ParaBlock) -> tuple:
    size = round((p.size or 0) * 2) / 2
    return (size, p.bold_ratio >= 0.9, p.caps, p.italic_ratio >= 0.9, p.underline_ratio >= 0.9)


def _prominence(sig: tuple, centered: bool = False) -> float:
    size, bold, caps, italic, under = sig
    return size * 10 + 4 * bold + 2 * caps + 1 * under - 0.5 * italic + (0.5 if centered else 0)


def _body_size(paras: list[ParaBlock]) -> float:
    c: Counter = Counter()
    for p in paras:
        if p.role == "text" and not p.in_table and len(p.text) >= 40 and p.size:
            c[p.size] += len(p.text)
    if not c:
        for p in paras:
            if p.size and p.role == "text":
                c[p.size] += len(p.text)
    return c.most_common(1)[0][0] if c else 11.0


def _detect_headings(blocks: list[Block], hc: HeadingConfig, toc_info: dict) -> dict:
    paras = [b for b in blocks if isinstance(b, ParaBlock)]
    body = _body_size(paras)
    toc_by_anchor: dict[str, int] = {}
    toc_by_text: dict[str, int] = {}
    for e in toc_info["entries"]:
        for a in e.anchors:
            toc_by_anchor[a] = e.toc_level or 1
        key = norm_text(e.text)
        if key:
            toc_by_text.setdefault(key, e.toc_level or 1)
    toc_end = toc_info["last_order"]

    pending: list[ParaBlock] = []                 # titres sans niveau (mise en forme / mot-clé)
    for p in paras:
        if p.role != "text" or not p.text:
            continue
        if (p.in_table and not hc.allow_in_tables) or (p.in_textbox and not hc.allow_in_textboxes):
            continue
        style = p.style_names_chain[0] if p.style_names_chain else ""
        text = p.text.strip()
        # (a) style Title
        if style in TITLE_STYLE_NAMES or (p.style_id or "").lower() == "title":
            p.role, p.level, p.heading_source = "title", 0, "style_title"
            continue
        long_text = len(text) > 2 * hc.max_chars
        # (b) niveau hiérarchique explicite (pPr/outlineLvl ou hérité du style)
        if p.outline_lvl is not None and not long_text:
            _set_heading(p, p.outline_lvl + 1, "outline_level", 1.0)
            continue
        # (c) nom de style numéroté : « Level 1 », « Titre 2 », « Niveau 3 », « H4 »…
        if hc.detect_from_style_names and not long_text:
            lvl = None
            for s in (*p.style_names_chain, (p.style_id or "").lower()):
                m = HEADING_STYLE_RE.match(s.strip())
                if m and not NOT_HEADING_STYLES.search(s):
                    lvl = int(m.group(1))
                    break
            if lvl is not None:
                _set_heading(p, lvl, "style_name", 0.95)
                continue
        # (d) ancre ou texte présent dans le sommaire
        if hc.use_toc and p.order > toc_end and (toc_by_anchor or toc_by_text) and _shape_ok(p, hc):
            lvl = next((toc_by_anchor[b] for b in p.bookmarks if b in toc_by_anchor), None)
            if lvl is not None:
                _set_heading(p, lvl, "toc_anchor", 0.95)
                continue
            lvl = toc_by_text.get(norm_text(text))
            if lvl is not None and len(text) <= hc.max_chars:
                _set_heading(p, lvl, "toc_text", 0.85)
                continue
        # (e) numérotation Word multi-niveaux (« %1.%2 ») sur paragraphe court
        if hc.detect_from_numbering and p.num_id and p.list_enumerated and _shape_ok(p, hc):
            multi = (p.list_lvl_text or "").count("%") >= 2
            if (multi and (p.ilvl or 0) >= 1 and (_emphasis(p, body, hc) or p.size and p.size > body)) or \
               ((p.ilvl or 0) == 0 and _emphasis(p, body, hc) and p.bold_ratio >= 0.9):
                _set_heading(p, (p.ilvl or 0) + 1, "numbering", 0.8)
                continue
        if not _shape_ok(p, hc) or (p.num_id and not p.list_enumerated):
            continue
        # (f) numérotation saisie « 2.3.1 Titre », « IV. Titre », « Chapitre 3 »
        if hc.detect_from_text_numbering and _emphasis(p, body, hc):
            m = TEXT_NUMBER_RE.match(text)
            if m:
                depth = len(re.split(r"[.\-]", m.group("num").strip(".-")))
                _set_heading(p, depth, "text_numbering", 0.8)
                continue
            if CHAPTER_RE.match(text) or ROMAN_RE.match(text):
                pending.append(p)
                p.heading_source, p.confidence = "text_numbering", 0.75
                continue
        # (g) mot-clé dans le nom de style (« Titre chapitre », « Section title »)
        if hc.detect_from_style_names and HEADING_KEYWORD_RE.search(style) and not NOT_HEADING_STYLES.search(style):
            pending.append(p)
            p.heading_source, p.confidence = "style_keyword", 0.7
            continue
        # (h) mise en forme seule
        if hc.detect_from_formatting and not p.num_id and _emphasis(p, body, hc):
            pending.append(p)
            p.heading_source, p.confidence = "formatting", 0.6

    pending = _guard_formatting(pending, paras, hc)
    _assign_levels(pending, paras, body)
    return {"body_font_size": body}


def _set_heading(p: ParaBlock, level: int, source: str, conf: float) -> None:
    p.role, p.level, p.heading_source, p.confidence = "heading", max(1, min(9, level)), source, conf


def _guard_formatting(pending: list[ParaBlock], paras: list[ParaBlock], hc: HeadingConfig) -> list[ParaBlock]:
    """Écarte les faux positifs : signature trop fréquente, blocs gras consécutifs, dernier paragraphe."""
    fmt = [p for p in pending if p.heading_source == "formatting"]
    if not fmt:
        return pending
    textual = [p for p in paras if p.role in ("text", "heading") and p.text and not p.in_table]
    sig_count = Counter(_signature(p) for p in fmt)
    rejected: set[int] = set()
    if len(textual) >= hc.min_paragraphs_for_share_guard:
        for sig, cnt in sig_count.items():
            if cnt / len(textual) > hc.max_signature_share:
                rejected |= {id(p) for p in fmt if _signature(p) == sig}
    # séries de >= 4 paragraphes consécutifs de même signature = bloc mis en valeur, pas des titres
    order_idx = {id(p): i for i, p in enumerate(textual)}
    fmt_sorted = sorted(fmt, key=lambda p: order_idx.get(id(p), 0))
    run: list[ParaBlock] = []
    for p in fmt_sorted + [None]:
        if run and p is not None and order_idx.get(id(p)) == order_idx.get(id(run[-1])) + 1 \
                and _signature(p) == _signature(run[-1]):
            run.append(p)
            continue
        if len(run) >= 4:
            rejected |= {id(x) for x in run}
        run = [p] if p is not None else []
    if textual and fmt and textual[-1] is fmt_sorted[-1]:
        rejected.add(id(textual[-1]))             # un titre ne termine pas un document
    out = []
    for p in pending:
        if id(p) in rejected:
            p.heading_source, p.confidence = None, 1.0
        else:
            out.append(p)
    return out


def _assign_levels(pending: list[ParaBlock], paras: list[ParaBlock], body: float) -> None:
    if not pending:
        return
    explicit: dict[int, list[float]] = defaultdict(list)
    for p in paras:
        if p.role == "heading" and p.level:
            explicit[p.level].append(_prominence(_signature(p), p.jc == "center"))
    calib = sorted((lvl, sum(v) / len(v)) for lvl, v in explicit.items())
    sigs = sorted({_signature(p) for p in pending}, key=_prominence, reverse=True)
    if not calib:
        level_of = {s: i + 1 for i, s in enumerate(sigs)}
    else:
        max_lvl = max(l for l, _ in calib)
        level_of, extra = {}, 0
        for s in sigs:
            prom = _prominence(s)
            lvl = next((l for l, pr in calib if pr <= prom + 0.25), None)
            if lvl is None:
                extra += 1
                lvl = max_lvl + extra
            level_of[s] = lvl
    for p in pending:
        _set_heading(p, level_of[_signature(p)], p.heading_source, p.confidence)


# =================================================================== listes
def _detect_lists(blocks: list[Block], cfg: ParserConfig) -> None:
    prev_indent: list[int] = []
    for b in blocks:
        if not isinstance(b, ParaBlock):
            prev_indent = []
            continue
        if b.role == "text" and b.num_id and b.list_marker is not None:
            b.role = "list_item"
            continue
        if b.role != "text" or not cfg.detect_manual_lists or b.in_table:
            if b.role != "empty":
                prev_indent = []
            continue
        m = MANUAL_BULLET_RE.match(b.text)
        enumerated = False
        if not m:
            m = MANUAL_ENUM_RE.match(b.text)
            enumerated = m is not None
        if not m:
            prev_indent = []
            continue
        marker = m.group("m")
        b.role, b.manual_marker = "list_item", marker
        b.list_marker = marker if enumerated else _bullet(marker)
        b.list_enumerated = enumerated
        b.text = b.text[m.end():].strip()
        if b.indent_left not in prev_indent:
            prev_indent = sorted(set(prev_indent + [b.indent_left]))
        b.ilvl = prev_indent.index(b.indent_left)


def _bullet(m: str) -> str:
    from .styles import normalize_bullet
    return normalize_bullet(m) if m not in "-–—*" else "-"
