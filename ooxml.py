"""Accès bas niveau au paquet OPC/Open XML : zip, relations, parsing XML sécurisé.

Aucune dépendance à python-docx : lxml uniquement.
"""
from __future__ import annotations

import posixpath
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from typing import IO, Iterator

from lxml import etree

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "wpg": "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "v": "urn:schemas-microsoft-com:vml",
    "o": "urn:schemas-microsoft-com:office:office",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a16": "http://schemas.microsoft.com/office/drawing/2014/main",
    "adec": "http://schemas.microsoft.com/office/drawing/2017/decorative",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}

REL_OFFICE_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
REL_STRICT_OFFICE_DOC = "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument"


@lru_cache(maxsize=512)
def qn(tag: str) -> str:
    """'w:p' -> '{http://...}p' (mis en cache : appelé des millions de fois)."""
    prefix, local = tag.split(":")
    return f"{{{NS[prefix]}}}{local}"


W = NS["w"]


def wattr(el: etree._Element | None, name: str, default: str | None = None) -> str | None:
    """Lit un attribut w:xxx (gère aussi l'absence de namespace, fréquente après conversion)."""
    if el is None:
        return default
    v = el.get(f"{{{W}}}{name}")
    if v is None:
        v = el.get(name)
    return default if v is None else v


def on_off(el: etree._Element | None) -> bool | None:
    """Sémantique ST_OnOff : <w:b/> = True, <w:b w:val="0"/> = False, absent = None."""
    if el is None:
        return None
    v = wattr(el, "val")
    return v not in ("0", "false", "off", "none")


def make_parser(huge_tree: bool = False) -> etree.XMLParser:
    # Sécurité : pas d'entités externes, pas de réseau (DOCX = entrée non fiable).
    return etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=huge_tree,
                           remove_comments=True, remove_pis=True)


class PackageError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class Relationship:
    rid: str
    type: str
    target: str          # chemin résolu dans le zip (ou URL si externe)
    external: bool


class DocxPackage:
    """Paquet DOCX ouvert en lecture. Les parties sont lues à la demande."""

    def __init__(self, source: str | IO[bytes], max_uncompressed_bytes: int = 1_000_000_000,
                 huge_tree: bool = False):
        try:
            self.zip = zipfile.ZipFile(source)
        except zipfile.BadZipFile as exc:  # .doc binaire renommé, fichier corrompu…
            raise PackageError(f"Pas un paquet OOXML valide (DOC non converti ?) : {exc}") from exc
        total = sum(i.file_size for i in self.zip.infolist())
        if total > max_uncompressed_bytes:
            raise PackageError(f"Archive trop volumineuse une fois décompressée ({total} octets)")
        self._names = {n.lstrip("/"): n for n in self.zip.namelist()}
        self._names_ci = {n.lower(): n for n in self._names}
        self._rels_cache: dict[str, dict[str, Relationship]] = {}
        self.huge_tree = huge_tree
        self.content_types = self._load_content_types()
        self.main_part = self._find_main_part()

    # ------------------------------------------------------------------ parts
    def has(self, part: str) -> bool:
        return self._resolve_name(part) is not None

    def _resolve_name(self, part: str) -> str | None:
        part = part.lstrip("/")
        if part in self._names:
            return self._names[part]
        return self._names_ci.get(part.lower())  # certains convertisseurs changent la casse

    def read(self, part: str) -> bytes:
        name = self._resolve_name(part)
        if name is None:
            raise KeyError(part)
        return self.zip.read(name)

    def open(self, part: str) -> IO[bytes]:
        name = self._resolve_name(part)
        if name is None:
            raise KeyError(part)
        return self.zip.open(name)

    def size(self, part: str) -> int:
        name = self._resolve_name(part)
        return self.zip.getinfo(name).file_size if name else 0

    def xml(self, part: str) -> etree._Element | None:
        if not self.has(part):
            return None
        return etree.fromstring(self.read(part), make_parser(self.huge_tree))

    def iterparse(self, part: str, events=("start", "end")) -> Iterator[tuple[str, etree._Element]]:
        return etree.iterparse(self.open(part), events=events, resolve_entities=False,
                               no_network=True, huge_tree=self.huge_tree, remove_comments=True)

    # ------------------------------------------------------- content types
    def _load_content_types(self) -> tuple[dict[str, str], dict[str, str]]:
        defaults, overrides = {}, {}
        root = self.xml("[Content_Types].xml")
        if root is not None:
            for el in root:
                ln = etree.QName(el).localname
                if ln == "Default":
                    defaults[el.get("Extension", "").lower()] = el.get("ContentType", "")
                elif ln == "Override":
                    overrides[el.get("PartName", "").lstrip("/").lower()] = el.get("ContentType", "")
        return defaults, overrides

    def content_type(self, part: str) -> str:
        defaults, overrides = self.content_types
        ct = overrides.get(part.lstrip("/").lower())
        if ct:
            return ct
        ext = posixpath.splitext(part)[1].lstrip(".").lower()
        return defaults.get(ext) or _EXT_MIME.get(ext, "application/octet-stream")

    # ---------------------------------------------------------- relations
    def rels(self, part: str) -> dict[str, Relationship]:
        if part in self._rels_cache:
            return self._rels_cache[part]
        base_dir, fname = posixpath.split(part.lstrip("/"))
        rels_part = posixpath.join(base_dir, "_rels", fname + ".rels")
        out: dict[str, Relationship] = {}
        root = self.xml(rels_part)
        if root is not None:
            for el in root:
                rid, target = el.get("Id"), el.get("Target", "")
                external = el.get("TargetMode") == "External"
                if not external:
                    target = (target.lstrip("/") if target.startswith("/")
                              else posixpath.normpath(posixpath.join(base_dir, target)))
                out[rid] = Relationship(rid, el.get("Type", ""), target, external)
        self._rels_cache[part] = out
        return out

    def rel_target_by_type(self, part: str, suffix: str) -> str | None:
        for rel in self.rels(part).values():
            if rel.type.endswith(suffix) and not rel.external:
                return rel.target
        return None

    def _find_main_part(self) -> str:
        for rel in self.rels("").values():
            if rel.type in (REL_OFFICE_DOC, REL_STRICT_OFFICE_DOC):
                return rel.target
        if self.has("word/document.xml"):
            return "word/document.xml"
        raise PackageError("Partie principale word/document.xml introuvable")

    # -------------------------------------------------------- core props
    def core_properties(self) -> dict[str, str]:
        root = self.xml("docProps/core.xml")
        if root is None:
            return {}
        out = {}
        for el in root:
            if el.text and el.text.strip():
                out[etree.QName(el).localname] = el.text.strip()
        return out

    def close(self) -> None:
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_EXT_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
    "bmp": "image/bmp", "tif": "image/tiff", "tiff": "image/tiff", "emf": "image/x-emf",
    "wmf": "image/x-wmf", "svg": "image/svg+xml", "webp": "image/webp", "wdp": "image/vnd.ms-photo",
}
