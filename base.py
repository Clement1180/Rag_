"""Contrat d'analyse d'image, indépendant de tout fournisseur.

Le moteur de parsing n'importe JAMAIS ce paquet : l'enrichissement est une passe
post-parsing optionnelle qui ne voit que le DocModel et les octets des images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

PROMPT_VERSION = "v1"
DEFAULT_PROMPT = (
    "Tu décris une image extraite d'un document professionnel afin de l'indexer dans un moteur "
    "de recherche. Décris factuellement son contenu : type (schéma, graphique, capture, photo, "
    "logo…), éléments et libellés lisibles, chiffres clés, relations. Retranscris le texte visible. "
    "N'invente rien. 120 mots maximum, en {language}.\n"
    "Contexte : section « {heading_path} ». Légende : « {caption} ». Texte alternatif : « {alt_text} »."
)


@dataclass(frozen=True, slots=True)
class ImageContext:
    caption: str = ""
    alt_text: str = ""
    heading_path: tuple[str, ...] = ()
    neighbor_text: str = ""
    language: str = "français"


@dataclass(frozen=True, slots=True)
class ImageInput:
    data: bytes
    mimetype: str
    sha256: str
    context: ImageContext = field(default_factory=ImageContext)

    def prompt(self, template: str = DEFAULT_PROMPT) -> str:
        c = self.context
        return template.format(language=c.language, heading_path=" > ".join(c.heading_path) or "-",
                               caption=c.caption or "-", alt_text=c.alt_text or "-")


@dataclass(slots=True)
class ImageDescription:
    text: str
    created_by: str                   # « fournisseur:modèle »
    prompt_version: str = PROMPT_VERSION
    extra: dict = field(default_factory=dict)


@runtime_checkable
class ImageDescriber(Protocol):
    """Tout objet exposant `id` et `describe()` convient (duck typing)."""
    id: str
    supported_mimetypes: frozenset[str]

    def describe(self, image: ImageInput) -> ImageDescription: ...
