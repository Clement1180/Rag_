"""Modèle de document en mémoire, calqué sur DoclingDocument (tableaux plats + références $ref).

Les objets sont légers (slots) ; la sérialisation JSON n'est faite qu'à la demande.
Le chunker travaille directement sur ce modèle : le format Docling n'est qu'une vue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .blocks import CellData


@dataclass(slots=True, kw_only=True)
class Item:
    self_ref: str
    parent: str | None
    label: str
    children: list[str] = field(default_factory=list)
    content_layer: str = "body"
    meta: dict = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class TextItem(Item):
    text: str
    orig: str
    level: int | None = None          # section_header
    enumerated: bool | None = None    # list_item
    marker: str | None = None
    hyperlink: str | None = None


@dataclass(slots=True, kw_only=True)
class TableItem(Item):
    cells: list[CellData]
    num_rows: int
    num_cols: int
    captions: list[str] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class PictureItem(Item):
    image: dict | None = None         # mimetype, size, uri, sha256…
    captions: list[str] = field(default_factory=list)
    description: dict | None = None   # rempli par l'enrichissement multimodal (optionnel)


@dataclass(slots=True, kw_only=True)
class GroupItem(Item):
    name: str = "group"


class DocModel:
    def __init__(self, name: str, origin: dict):
        self.name, self.origin = name, origin
        self.body = GroupItem(self_ref="#/body", parent=None, label="unspecified", name="_root_")
        self.furniture = GroupItem(self_ref="#/furniture", parent=None, label="unspecified",
                                   name="_root_", content_layer="furniture")
        self.texts: list[TextItem] = []
        self.tables: list[TableItem] = []
        self.pictures: list[PictureItem] = []
        self.groups: list[GroupItem] = []
        self.meta: dict = {}

    # ------------------------------------------------------------ ajout
    def _attach(self, item: Item, parent: Item) -> None:
        parent.children.append(item.self_ref)

    def add_text(self, label: str, text: str, parent: Item, *, layer: str | None = None,
                 orig: str | None = None, **kw) -> TextItem:
        it = TextItem(self_ref=f"#/texts/{len(self.texts)}", parent=parent.self_ref, label=label,
                      text=text, orig=orig if orig is not None else text,
                      content_layer=layer or parent.content_layer, **kw)
        self.texts.append(it)
        self._attach(it, parent)
        return it

    def add_table(self, parent: Item, cells: list[CellData], num_rows: int, num_cols: int,
                  *, layer: str | None = None, meta: dict | None = None) -> TableItem:
        it = TableItem(self_ref=f"#/tables/{len(self.tables)}", parent=parent.self_ref, label="table",
                       cells=cells, num_rows=num_rows, num_cols=num_cols,
                       content_layer=layer or parent.content_layer, meta=meta or {})
        self.tables.append(it)
        self._attach(it, parent)
        return it

    def add_picture(self, parent: Item, *, label: str = "picture", image: dict | None = None,
                    layer: str | None = None, meta: dict | None = None) -> PictureItem:
        it = PictureItem(self_ref=f"#/pictures/{len(self.pictures)}", parent=parent.self_ref,
                         label=label, image=image, content_layer=layer or parent.content_layer,
                         meta=meta or {})
        self.pictures.append(it)
        self._attach(it, parent)
        return it

    def add_group(self, parent: Item, label: str, name: str, *, layer: str | None = None,
                  meta: dict | None = None) -> GroupItem:
        it = GroupItem(self_ref=f"#/groups/{len(self.groups)}", parent=parent.self_ref, label=label,
                       name=name, content_layer=layer or parent.content_layer, meta=meta or {})
        self.groups.append(it)
        self._attach(it, parent)
        return it

    # ---------------------------------------------------------- lecture
    def resolve(self, ref: str) -> Item:
        if ref == "#/body":
            return self.body
        if ref == "#/furniture":
            return self.furniture
        _, kind, idx = ref.split("/")
        return getattr(self, kind)[int(idx)]

    def iterate(self, root: Item | None = None, layers: set[str] | None = frozenset({"body"}),
                depth: int = 0) -> Iterator[tuple[Item, int]]:
        """Parcours en profondeur = ordre de lecture. `layers=None` : toutes les couches."""
        root = root or self.body
        for ref in root.children:
            it = self.resolve(ref)
            if layers is not None and it.content_layer not in layers:
                continue
            yield it, depth
            yield from self.iterate(it, layers, depth + 1)
