"""Orchestrateur : paquet -> extraction -> classification -> assemblage."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import IO

from .assemble import Assembler
from .blocks import ParaBlock
from .classify import classify, norm_text
from .config import ParserConfig
from .extract import Ctx, Extractor
from .images import ImageStore
from .model import DocModel
from .ooxml import DocxPackage, wattr
from .styles import NumberingResolver, StyleResolver

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@dataclass(slots=True)
class ParseResult:
    model: DocModel
    source: str | None
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class DocxParser:
    def __init__(self, config: ParserConfig | None = None):
        self.cfg = config or ParserConfig()

    def parse(self, source: str | os.PathLike | IO[bytes], name: str | None = None) -> ParseResult:
        t = {"start": time.perf_counter()}
        src_path = os.fspath(source) if isinstance(source, (str, os.PathLike)) else None
        binary_hash, sha = _file_hash(source)
        with DocxPackage(src_path or source, self.cfg.max_uncompressed_bytes, self.cfg.huge_tree) as pkg:
            main = pkg.main_part
            styles = StyleResolver(pkg.xml(pkg.rel_target_by_type(main, "/styles") or "word/styles.xml"))
            num_part = pkg.rel_target_by_type(main, "/numbering") or "word/numbering.xml"
            numbering = NumberingResolver(pkg.xml(num_part))
            notes = self._notes(pkg, styles, num_part)
            t["setup"] = time.perf_counter()

            ex = Extractor(pkg, styles, numbering, self.cfg)
            blocks = list(ex.iter_blocks())
            t["extract"] = time.perf_counter()

            stats = classify(blocks, self.cfg)
            t["classify"] = time.perf_counter()

            fname = name or (os.path.basename(src_path) if src_path else "document.docx")
            model = DocModel(os.path.splitext(fname)[0],
                             {"mimetype": DOCX_MIME, "binary_hash": binary_hash, "filename": fname})
            images = ImageStore(pkg, self.cfg)
            Assembler(model, images, notes).build(blocks)
            if self.cfg.include_furniture:
                self._furniture(pkg, styles, num_part, model)
            t["assemble"] = time.perf_counter()

            model.meta = {
                "sha256": sha,
                "core_properties": pkg.core_properties(),
                "body_font_size": stats.get("body_font_size"),
                "heading_sources": _count(p.heading_source for p in blocks
                                          if isinstance(p, ParaBlock) and p.role == "heading"),
                "image_occurrences": images.occurrences,
                "approx_pages": max((b.page_hint for b in blocks), default=1),
                "blocks": len(blocks),
            }
        steps = ["setup", "extract", "classify", "assemble"]
        prev, timings = t["start"], {}
        for s in steps:
            timings[s] = round((t[s] - prev) * 1000, 2)
            prev = t[s]
        return ParseResult(model=model, source=src_path, warnings=ex.warnings, timings_ms=timings)

    # ----------------------------------------------------------------- notes
    def _notes(self, pkg: DocxPackage, styles: StyleResolver, num_part: str) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for kind, enabled in (("footnote", self.cfg.include_footnotes), ("endnote", self.cfg.include_endnotes)):
            part = pkg.rel_target_by_type(pkg.main_part, f"/{kind}s")
            if not enabled or not part:
                continue
            root = pkg.xml(part)
            if root is None:
                continue
            ex = Extractor(pkg, styles, NumberingResolver(pkg.xml(num_part)), self.cfg, part=part)
            notes = {}
            for fn in root:
                if wattr(fn, "type") in ("separator", "continuationSeparator", "continuationNotice"):
                    continue
                texts = [b.text for child in fn for b in ex._block_element(child, Ctx())
                         if isinstance(b, ParaBlock) and b.text]
                if texts:
                    notes[wattr(fn, "id")] = " ".join(" ".join(texts).split())
            out[kind] = notes
        return out

    # ------------------------------------------------------------- mobilier
    def _furniture(self, pkg: DocxPackage, styles: StyleResolver, num_part: str, model: DocModel) -> None:
        seen: set[str] = set()
        for rel in pkg.rels(pkg.main_part).values():
            kind = "page_header" if rel.type.endswith("/header") else "page_footer" if rel.type.endswith("/footer") else None
            if kind is None or rel.external:
                continue
            ex = Extractor(pkg, styles, NumberingResolver(pkg.xml(num_part)), self.cfg, part=rel.target)
            for b in ex.iter_part_blocks(rel.target):
                text = getattr(b, "text", "") or ""
                key = f"{kind}:{norm_text(text)}"
                if text.strip() and key not in seen:       # même en-tête sur N sections -> 1 item
                    seen.add(key)
                    model.add_text(kind, " ".join(text.split()), model.furniture, meta={"part": rel.target})


def parse_docx(source, config: ParserConfig | None = None, **kw) -> ParseResult:
    return DocxParser(config).parse(source, **kw)


def _file_hash(source) -> tuple[int, str]:
    h = hashlib.sha256()
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    else:
        pos = source.tell()
        h.update(source.read())
        source.seek(pos)
    digest = h.hexdigest()
    return int(digest[:16], 16), digest


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out
