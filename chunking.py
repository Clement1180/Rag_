"""Chunking hiérarchique orienté RAG.

Principes :
  * frontières = titres (jamais de chunk à cheval sur deux sections, sauf fusion
    explicite de petites sections sœurs) ;
  * unités insécables : paragraphe (découpé en phrases seulement s'il dépasse le budget),
    liste (découpée entre items), tableau (découpé par lignes, en-tête répété),
    image (légende + texte alternatif + description) ;
  * `text` = contenu brut (affichage / citation) ; `embed_text` = fil d'Ariane + contenu
    (vectorisation) : le titre n'est jamais dupliqué dans les deux ;
  * sommaire, en-têtes / pieds et légendes déjà rattachées ne produisent pas de chunk.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

from .model import DocModel, GroupItem, Item, PictureItem, TableItem, TextItem
from .serialize import _header_rows, table_to_markdown

SENT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÀ-ÖØ-Þ0-9«\"(])")


def approx_tokens(text: str) -> int:
    """Estimation sans dépendance (~4 caractères / token en français). Remplaçable par tiktoken/HF."""
    return max(1, (len(text) + 3) // 4)


@dataclass(slots=True)
class ChunkerConfig:
    max_tokens: int = 512
    min_tokens: int = 80                 # sections plus petites fusionnées avec leurs sœurs
    tokenizer: Callable[[str], int] = approx_tokens
    merge_small_sections: bool = True
    table_mode: str = "markdown"         # markdown | rows (ligne = « col: valeur ; … »)
    tables_as_own_chunks: bool = True
    table_rows_per_chunk: int | None = None  # None = autant que le budget permet
    picture_min_tokens_own_chunk: int = 120  # description longue -> chunk dédié
    include_footnotes: bool = True
    dedup_identical: bool = True
    overlap_sentences: int = 0


@dataclass(slots=True)
class Unit:
    kind: str                 # text | list | table | picture | heading
    text: str
    refs: list[str]
    path: tuple[str, ...]
    order: int
    page: int
    tokens: int = 0
    item: Item | None = None
    level: int = 0


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    embed_text: str
    heading_path: list[str]
    kind: str
    item_refs: list[str]
    order_start: int
    order_end: int
    page_start: int
    page_end: int
    token_count: int
    content_hash: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


class HierarchicalChunker:
    def __init__(self, cfg: ChunkerConfig | None = None):
        self.cfg = cfg or ChunkerConfig()
        self._model: DocModel | None = None

    # ============================================================ unités
    def units(self, model: DocModel) -> list[Unit]:
        units: list[Unit] = []
        path: list[tuple[int, str]] = []
        tok = self.cfg.tokenizer

        def cur_path() -> tuple[str, ...]:
            return tuple(t for _, t in path)

        consumed: set[str] = set()
        for it, _depth in model.iterate():
            if it.self_ref in consumed:
                continue
            order = it.meta.get("order", 0)
            page = it.meta.get("page_hint", 1)
            if isinstance(it, TextItem) and it.label in ("title", "section_header"):
                lvl = 0 if it.label == "title" else (it.level or 1)
                while path and path[-1][0] >= lvl:
                    path.pop()
                path.append((lvl, it.text))
                units.append(Unit("heading", it.text, [it.self_ref], cur_path(), order, page, tok(it.text),
                                  item=it, level=lvl))
            elif isinstance(it, GroupItem) and it.label == "list":
                lines, refs = self._list_lines(model, it, 0, consumed)
                if lines:
                    units.append(Unit("list", "\n".join(lines), refs, cur_path(), order, page,
                                      tok("\n".join(lines)), item=it))
            elif isinstance(it, TableItem):
                self._consume_children(model, it, consumed)
                if it.meta.get("nested_in_cell"):
                    continue          # déjà présent dans le texte de la cellule parente
                units.append(Unit("table", "", [it.self_ref], cur_path(), order, page, 0, item=it))
                for ref in it.children:  # images contenues dans les cellules
                    child = model.resolve(ref)
                    if isinstance(child, PictureItem):
                        u = self._picture_unit(model, child, cur_path())
                        if u:
                            units.append(u)
            elif isinstance(it, PictureItem):
                self._consume_children(model, it, consumed)
                u = self._picture_unit(model, it, cur_path())
                if u:
                    units.append(u)
            elif isinstance(it, TextItem):
                if it.label == "caption" and it.parent and not it.parent.startswith("#/texts"):
                    parent = model.resolve(it.parent)
                    if isinstance(parent, (TableItem, PictureItem)):
                        continue      # rattachée à sa cible
                if it.label == "footnote":
                    if self.cfg.include_footnotes and units:
                        units[-1].text += f"\n[note] {it.text}"
                        units[-1].refs.append(it.self_ref)
                        units[-1].tokens = tok(units[-1].text)
                    continue
                if it.text.strip():
                    units.append(Unit("text", it.text, [it.self_ref], cur_path(), order, page, tok(it.text), item=it))
        return units

    def _consume_children(self, model: DocModel, it: Item, consumed: set[str]) -> None:
        for ref in it.children:
            consumed.add(ref)
            self._consume_children(model, model.resolve(ref), consumed)

    def _list_lines(self, model: DocModel, grp: GroupItem, depth: int, consumed: set[str]):
        lines, refs = [], [grp.self_ref]
        consumed.add(grp.self_ref)
        for ref in grp.children:
            li = model.resolve(ref)
            consumed.add(ref)
            if isinstance(li, TextItem):
                marker = li.marker or "-"
                lines.append(f"{'  ' * depth}{marker} {li.text}")
                refs.append(ref)
                for cref in li.children:
                    child = model.resolve(cref)
                    consumed.add(cref)
                    if isinstance(child, GroupItem):
                        sub_lines, sub_refs = self._list_lines(model, child, depth + 1, consumed)
                        lines += sub_lines
                        refs += sub_refs
                    elif isinstance(child, TextItem) and child.label == "footnote" and self.cfg.include_footnotes:
                        lines.append(f"{'  ' * (depth + 1)}[note] {child.text}")
        return lines, refs

    def _picture_unit(self, model: DocModel, pic: PictureItem, path) -> Unit | None:
        parts = []
        caps = [model.resolve(c).text for c in pic.captions]
        if caps:
            parts.append(" ".join(caps))
        alt = pic.meta.get("alt_text")
        if alt and alt not in caps:
            parts.append(f"Texte alternatif : {alt}")
        if pic.meta.get("chart_text"):
            parts.append(f"Données du graphique : {pic.meta['chart_text'][:1500]}")
        if pic.description:
            parts.append(f"Description : {pic.description['text']}")
        if not parts or pic.meta.get("decorative"):
            return None   # image sans aucun signal textuel : rien à indexer
        label = "Graphique" if pic.label == "chart" else "Figure"
        text = f"[{label}] " + "\n".join(parts)
        return Unit("picture", text, [pic.self_ref] + list(pic.captions), path,
                    pic.meta.get("order", 0), pic.meta.get("page_hint", 1), self.cfg.tokenizer(text), item=pic)

    # ========================================================= découpage
    def chunk(self, model: DocModel, doc_id: str | None = None) -> list[Chunk]:
        doc_id = doc_id or model.meta.get("sha256", model.name)[:16]
        self._model = model
        units = self.units(model)
        sections: list[list[Unit]] = []
        for u in units:
            if u.kind == "heading" or not sections:
                sections.append([])
            sections[-1].append(u)
        if self.cfg.merge_small_sections:
            sections = self._merge_small(sections)
        raw: list[tuple[list[Unit], str, tuple[str, ...]]] = []
        for sec in sections:
            raw.extend(self._pack(sec))
        return self._finalize(raw, doc_id)

    def _section_tokens(self, sec: list[Unit]) -> int:
        return sum(u.tokens if u.kind != "table" else self._table_tokens(u) for u in sec)

    def _table_tokens(self, u: Unit) -> int:
        return self.cfg.tokenizer(table_to_markdown(u.item))

    def _merge_small(self, sections: list[list[Unit]]) -> list[list[Unit]]:
        """Fusionne une petite section avec la suivante si elle est sa sœur ou sa fille."""
        out: list[list[Unit]] = []
        for sec in sections:
            if out:
                prev = out[-1]
                prev_head = prev[0] if prev[0].kind == "heading" else None
                head = sec[0] if sec[0].kind == "heading" else None
                prev_tok, cur_tok = self._section_tokens(prev), self._section_tokens(sec)
                related = (prev_head is not None and head is not None and
                           (head.path[:-1] == prev_head.path[:-1] or head.path[:-1] == prev_head.path))
                has_table = any(u.kind == "table" for u in prev + sec) and self.cfg.tables_as_own_chunks
                if (related and prev_tok < self.cfg.min_tokens and not has_table
                        and prev_tok + cur_tok <= self.cfg.max_tokens):
                    out[-1] = prev + sec
                    continue
            out.append(sec)
        return out

    def _pack(self, sec: list[Unit]):
        """Remplit des chunks jusqu'au budget, sans couper les unités sauf nécessité."""
        cfg, tok = self.cfg, self.cfg.tokenizer
        base_path = sec[0].path if sec[0].kind == "heading" else sec[0].path
        out, cur, cur_tok = [], [], 0

        def flush():
            nonlocal cur, cur_tok
            if cur:   # un titre seul (section vide) donne un chunk « heading » minimal
                out.append((cur, "mixed" if any(x.kind != "heading" for x in cur) else "heading", base_path))
            cur, cur_tok = [], 0

        for u in sec:
            if u.kind == "table":
                if cfg.tables_as_own_chunks:
                    if cur and not all(x.kind == "heading" for x in cur):
                        flush()
                    heads = [x for x in cur if x.kind == "heading"]
                    cur, cur_tok = [], 0
                    for piece in self._split_table(u):
                        out.append((heads + [piece], "table", u.path))
                        heads = []
                    continue
                u = Unit("text", table_to_markdown(u.item), u.refs, u.path, u.order, u.page, 0, item=u.item)
                u.tokens = tok(u.text)
            if u.tokens > cfg.max_tokens:
                if cur:
                    flush()
                pieces = self._split_unit(u)
                for p in pieces[:-1]:
                    out.append(([p], "mixed", u.path))
                cur, cur_tok = [pieces[-1]], pieces[-1].tokens
                continue
            if cur_tok + u.tokens > cfg.max_tokens and any(x.kind != "heading" for x in cur):
                flush()
            if u.kind == "picture" and u.tokens >= cfg.picture_min_tokens_own_chunk:
                if cur and any(x.kind != "heading" for x in cur):
                    flush()
                out.append((cur + [u], "picture", u.path))
                cur, cur_tok = [], 0
                continue
            cur.append(u)
            cur_tok += u.tokens
        flush()
        return out

    def _split_unit(self, u: Unit) -> list[Unit]:
        cfg, tok = self.cfg, self.cfg.tokenizer
        if u.kind == "list":
            atoms = u.text.split("\n")
            sep = "\n"
        else:
            atoms = [s for s in SENT_RE.split(u.text) if s]
            sep = " "
        pieces, buf = [], []
        for a in atoms:
            while tok(a) > cfg.max_tokens:   # phrase monstre : coupe dure par caractères
                cut = cfg.max_tokens * 4
                atoms_head, a = a[:cut], a[cut:]
                if buf:
                    pieces.append(sep.join(buf))
                    buf = []
                pieces.append(atoms_head)
            if buf and tok(sep.join(buf + [a])) > cfg.max_tokens:
                pieces.append(sep.join(buf))
                buf = buf[-cfg.overlap_sentences:] if cfg.overlap_sentences and u.kind != "list" else []
            buf.append(a)
        if buf:
            pieces.append(sep.join(buf))
        return [Unit(u.kind, p, list(u.refs), u.path, u.order, u.page, tok(p), item=u.item) for p in pieces]

    def _split_table(self, u: Unit) -> list[Unit]:
        t: TableItem = u.item
        tok = self.cfg.tokenizer
        hdr = _header_rows(t)
        caption = self._caption_text(t)
        prefix = f"[Tableau] {caption}\n" if caption else "[Tableau]\n"
        render = self._render_rows
        full = prefix + render(t, range(hdr or (1 if t.num_rows > 1 else 0), t.num_rows), hdr)
        if tok(full) <= self.cfg.max_tokens and not self.cfg.table_rows_per_chunk:
            return [Unit("table", full, list(u.refs) + t.captions, u.path, u.order, u.page, tok(full), item=t)]
        start = hdr or 1
        pieces, rows = [], []
        for r in range(start, t.num_rows):
            trial = prefix + render(t, range(rows[0] if rows else r, r + 1), hdr)
            too_many = self.cfg.table_rows_per_chunk and len(rows) >= self.cfg.table_rows_per_chunk
            if rows and (tok(trial) > self.cfg.max_tokens or too_many):
                pieces.append(range(rows[0], rows[-1] + 1))
                rows = []
            rows.append(r)
        if rows:
            pieces.append(range(rows[0], rows[-1] + 1))
        out = []
        for i, rg in enumerate(pieces):
            txt = prefix.rstrip("\n") + f" (lignes {rg.start + 1}-{rg.stop} / {t.num_rows})\n" + render(t, rg, hdr)
            out.append(Unit("table", txt, list(u.refs) + t.captions, u.path, u.order, u.page, tok(txt), item=t))
        return out

    def _render_rows(self, t: TableItem, rows: range, hdr: int) -> str:
        if self.cfg.table_mode == "rows" and hdr:
            grid = [[""] * t.num_cols for _ in range(t.num_rows)]
            for c in t.cells:
                for r in range(c.start_row, min(t.num_rows, c.start_row + c.row_span)):
                    for k in range(c.start_col, min(t.num_cols, c.start_col + c.col_span)):
                        grid[r][k] = c.text.replace("\n", " ")
            heads = [" / ".join(dict.fromkeys(grid[h][k] for h in range(hdr) if grid[h][k])) for k in range(t.num_cols)]
            return "\n".join(" ; ".join(f"{heads[k] or f'col{k+1}'}: {grid[r][k]}" for k in range(t.num_cols)
                                        if grid[r][k]) for r in rows)
        return table_to_markdown(t, rows, hdr)

    def _caption_text(self, t: TableItem) -> str:
        return " ".join(self._model.resolve(c).text for c in t.captions)

    # ========================================================= finalisation
    def _finalize(self, raw, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: dict[str, str] = {}
        for units, kind, path in raw:
            body_units = [u for u in units if u.kind != "heading"]
            heads_in = [u for u in units if u.kind == "heading"]
            if not body_units and not heads_in:
                continue
            lines = []
            for u in units:
                if u.kind == "heading":
                    lines.append(f"{'#' * min(6, u.level + 1)} {u.text}")
                else:
                    lines.append(u.text)
            text = "\n\n".join(lines).strip()
            if not body_units:
                # titre seul (section vide) : utile pour les requêtes de navigation, pas de doublon
                kind = "heading"
            # fil d'Ariane sans les titres déjà présents dans le texte (pas de duplication)
            full_path = list(max((u.path for u in units), key=len)) if units else list(path)
            if heads_in:
                full_path = list(heads_in[0].path)
            in_text = {u.text for u in heads_in}
            crumb = [h for h in full_path if h not in in_text]
            embed = (" > ".join(crumb) + "\n\n" + text) if crumb else text
            norm = re.sub(r"\s+", " ", text).strip().lower()
            h = hashlib.sha256(norm.encode()).hexdigest()
            refs = list(dict.fromkeys(r for u in units for r in u.refs))
            if self.cfg.dedup_identical and h in seen:
                prev = next(c for c in chunks if c.chunk_id == seen[h])
                prev.meta.setdefault("duplicate_refs", []).extend(refs)
                continue
            cid = hashlib.sha1(f"{doc_id}:{len(chunks)}:{h}".encode()).hexdigest()[:20]
            seen[h] = cid
            chunks.append(Chunk(
                chunk_id=cid, doc_id=doc_id, text=text, embed_text=embed, heading_path=full_path,
                kind=kind if kind != "mixed" else _dominant(body_units), item_refs=refs,
                order_start=min(u.order for u in units), order_end=max(u.order for u in units),
                page_start=min(u.page for u in units), page_end=max(u.page for u in units),
                token_count=self.cfg.tokenizer(embed), content_hash=h))
        for i, c in enumerate(chunks):
            c.meta["prev"] = chunks[i - 1].chunk_id if i else None
            c.meta["next"] = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None
        return chunks


def _dominant(units: list[Unit]) -> str:
    kinds = {u.kind for u in units}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def chunk_document(model: DocModel, cfg: ChunkerConfig | None = None, doc_id: str | None = None) -> list[Chunk]:
    return HierarchicalChunker(cfg).chunk(model, doc_id)
