"""Métadonnées et stockage des images. Un binaire = un hash, quel que soit le nombre d'occurrences."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import posixpath

from .blocks import ImageRef
from .config import ParserConfig
from .ooxml import DocxPackage

EMU_PER_PX = 9525
_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/bmp": "bmp",
        "image/tiff": "tif", "image/x-emf": "emf", "image/x-wmf": "wmf", "image/svg+xml": "svg",
        "image/webp": "webp"}


class ImageStore:
    def __init__(self, pkg: DocxPackage, cfg: ParserConfig):
        self.pkg, self.cfg = pkg, cfg
        self._cache: dict[str, dict] = {}       # partie zip -> infos binaires
        self.occurrences: dict[str, int] = {}   # sha256 -> nb d'occurrences (logos répétés…)

    def _binary_info(self, target: str) -> dict:
        if target in self._cache:
            return self._cache[target]
        info: dict = {"part": target, "bytes": self.pkg.size(target)}
        need_bytes = self.cfg.hash_images or self.cfg.read_pixel_size or self.cfg.image_mode != "reference"
        data = self.pkg.read(target) if need_bytes and self.pkg.has(target) else None
        if data is not None:
            if self.cfg.hash_images:
                info["sha256"] = hashlib.sha256(data).hexdigest()
            if self.cfg.read_pixel_size:
                info.update(_pixel_size(data))
            mime = self.pkg.content_type(target)
            if self.cfg.image_mode == "embed":
                info["uri"] = f"data:{mime};base64,{base64.b64encode(data).decode()}"
            elif self.cfg.image_mode == "export" and self.cfg.image_export_dir:
                os.makedirs(self.cfg.image_export_dir, exist_ok=True)
                key = info.get("sha256") or hashlib.sha256(data).hexdigest()
                ext = _EXT.get(mime) or posixpath.splitext(target)[1].lstrip(".") or "bin"
                path = os.path.join(self.cfg.image_export_dir, f"{key[:20]}.{ext}")
                if not os.path.exists(path):          # déduplication physique
                    with open(path, "wb") as fh:
                        fh.write(data)
                info["uri"] = path
        self._cache[target] = info
        return info

    def describe(self, ref: ImageRef) -> dict:
        """Dictionnaire `image` (format Docling ImageRef + extensions)."""
        img: dict = {"mimetype": ref.mimetype or "application/octet-stream", "dpi": 96}
        if ref.extent_emu:
            img["display_size_px"] = {"width": round(ref.extent_emu[0] / EMU_PER_PX, 1),
                                      "height": round(ref.extent_emu[1] / EMU_PER_PX, 1)}
            img["display_size_cm"] = {"width": round(ref.extent_emu[0] / 360000, 2),
                                      "height": round(ref.extent_emu[1] / 360000, 2)}
        if ref.external:
            img["uri"] = ref.target
            img["external"] = True
        elif ref.target:
            info = self._binary_info(ref.target)
            img.update({k: v for k, v in info.items() if k != "uri"})
            img["uri"] = info.get("uri") or ref.target
            if "sha256" in info:
                self.occurrences[info["sha256"]] = self.occurrences.get(info["sha256"], 0) + 1
        size = img.get("pixel_size") or img.get("display_size_px")
        img["size"] = size or {"width": 0.0, "height": 0.0}
        return img


def _pixel_size(data: bytes) -> dict:
    try:
        from PIL import Image  # optionnel
    except ImportError:
        return {}
    try:
        with Image.open(io.BytesIO(data)) as im:   # ne décode que l'en-tête
            return {"pixel_size": {"width": float(im.width), "height": float(im.height)}}
    except Exception:
        return {}
