"""Passe d'enrichissement des images (désactivée par défaut).

Filtrage (taille, décoratif, logos répétés, format), cache par (sha256, modèle, prompt),
concurrence bornée, tolérance aux pannes : une erreur fournisseur ne casse jamais le parsing.
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from ..model import DocModel, PictureItem, TextItem
from .base import PROMPT_VERSION, ImageContext, ImageDescriber, ImageInput

log = logging.getLogger(__name__)


@dataclass(slots=True)
class VisionConfig:
    enabled: bool = False                    # OFF par défaut
    min_pixels: int = 64 * 64               # icônes, puces graphiques : ignorées
    max_images: int = 200
    skip_decorative: bool = True
    skip_repeated_min: int = 3              # même binaire >= 3 fois = logo / filigrane
    concurrency: int = 4
    language: str = "français"
    convert: Callable[[bytes, str], tuple[bytes, str] | None] | None = None  # EMF/WMF/TIFF -> PNG


class DescriptionCache:
    """Cache mémoire + fichier JSONL optionnel. Clé = sha256|modèle|version de prompt."""

    def __init__(self, path: str | None = None):
        self.path, self._lock, self._d = path, threading.Lock(), {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    self._d[rec["key"]] = rec["value"]

    def get(self, key: str) -> dict | None:
        return self._d.get(key)

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._d[key] = value
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def default_convert(data: bytes, mime: str) -> tuple[bytes, str] | None:
    """Pillow sait convertir TIFF/BMP (et WMF/EMF sous Windows seulement) en PNG."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            buf = io.BytesIO()
            im.convert("RGBA" if im.mode in ("P", "RGBA", "LA") else "RGB").save(buf, "PNG")
            return buf.getvalue(), "image/png"
    except Exception:
        return None


def _heading_path(model: DocModel, pic: PictureItem) -> tuple[str, ...]:
    return tuple(pic.meta.get("heading_path") or ())


def enrich_pictures(model: DocModel, source: str, describer: ImageDescriber,
                    cfg: VisionConfig | None = None, cache: DescriptionCache | None = None) -> dict:
    cfg = cfg or VisionConfig(enabled=True)
    stats = {"described": 0, "cached": 0, "skipped": 0, "errors": 0}
    if not cfg.enabled:
        return stats
    cache = cache or DescriptionCache()
    occ: dict[str, int] = model.meta.get("image_occurrences", {})
    groups: dict[str, list[PictureItem]] = {}
    for pic in model.pictures:
        img = pic.image or {}
        sha, part = img.get("sha256"), img.get("part")
        px = img.get("pixel_size") or img.get("display_size_px") or {}
        area = (px.get("width") or 0) * (px.get("height") or 0)
        if (not sha or not part or img.get("external") or (cfg.skip_decorative and pic.meta.get("decorative"))
                or (area and area < cfg.min_pixels) or occ.get(sha, 0) >= cfg.skip_repeated_min):
            stats["skipped"] += 1
            continue
        groups.setdefault(sha, []).append(pic)          # un appel par binaire unique
    todo = list(groups.items())[: cfg.max_images]

    with zipfile.ZipFile(source) as zf:
        payloads = {}
        for sha, pics in todo:
            img = pics[0].image
            try:
                payloads[sha] = (zf.read(img["part"]), img["mimetype"])
            except KeyError:
                stats["errors"] += 1

    def work(item):
        sha, pics = item
        key = f"{sha}|{describer.id}|{PROMPT_VERSION}"
        hit = cache.get(key)
        if hit:
            return pics, hit, True
        if sha not in payloads:
            return pics, None, False
        data, mime = payloads[sha]
        if mime not in describer.supported_mimetypes:
            conv = (cfg.convert or default_convert)(data, mime)
            if conv is None:
                return pics, None, False
            data, mime = conv
        first = pics[0]
        caption = " ".join(model.resolve(c).text for c in first.captions)
        ctx = ImageContext(caption=caption, alt_text=first.meta.get("alt_text", ""),
                           heading_path=_heading_path(model, first),
                           neighbor_text=_neighbor_text(model, first), language=cfg.language)
        try:
            desc = describer.describe(ImageInput(data=data, mimetype=mime, sha256=sha, context=ctx))
        except Exception as exc:  # fournisseur indisponible, quota, image refusée…
            log.warning("Description d'image échouée (%s) : %s", sha[:12], exc)
            return pics, {"error": str(exc)}, False
        value = {"text": desc.text, "created_by": desc.created_by, "prompt_version": desc.prompt_version}
        cache.put(key, value)
        return pics, value, False

    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        for pics, value, cached in pool.map(work, todo):
            if not value or "error" in value:
                stats["errors" if value else "skipped"] += len(pics)
                continue
            for p in pics:
                p.description = value
            stats["cached" if cached else "described"] += len(pics)
    return stats


def _neighbor_text(model: DocModel, pic: PictureItem, max_chars: int = 400) -> str:
    """Texte du paragraphe porteur, sinon texte courant qui précède / suit l'image (ordre de lecture)."""
    ref = pic.meta.get("paragraph_ref")
    if ref:
        it = model.resolve(ref)
        if isinstance(it, TextItem) and it.text.strip():
            return it.text[:max_chars]
    before, after, found = "", "", False
    for it, _ in model.iterate():
        if it is pic:
            found = True
            continue
        if isinstance(it, TextItem) and it.label in ("text", "list_item") and it.text.strip():
            if not found:
                before = it.text
            else:
                after = it.text
                break
    return " […] ".join(t for t in (before[-max_chars // 2:], after[: max_chars // 2]) if t)
