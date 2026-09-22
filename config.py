"""Configuration du parser. Tous les défauts sont orientés RAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class HeadingConfig:
    detect_from_style_names: bool = True      # « Level 1 », « Titre 2 », « Niveau 3 »…
    detect_from_numbering: bool = True        # numérotation multi-niveaux Word
    detect_from_text_numbering: bool = True   # « 2.3.1 Contexte » saisi à la main
    detect_from_formatting: bool = True       # gras / taille / majuscules
    use_toc: bool = True                      # promotion via le sommaire (ancres _Toc, texte)
    max_chars: int = 200
    max_words: int = 25
    size_ratio: float = 1.15                  # taille >= corps * ratio
    max_signature_share: float = 0.30         # garde-fou : signature trop fréquente = corps
    min_paragraphs_for_share_guard: int = 15
    allow_in_tables: bool = False
    allow_in_textboxes: bool = False


@dataclass(slots=True)
class ParserConfig:
    include_furniture: bool = False           # en-têtes / pieds de page (exclus par défaut)
    include_footnotes: bool = True
    include_endnotes: bool = True
    include_textboxes: bool = True
    streaming: bool = True                    # iterparse : mémoire bornée sur gros documents
    huge_tree: bool = False
    max_uncompressed_bytes: int = 1_000_000_000
    unwrap_layout_tables: bool = True         # tableau 1x1 = cadre de mise en page
    detect_manual_lists: bool = True          # « • », « - », « a) » saisis à la main (DOC)
    detect_repeated_furniture: bool = True    # « Page 3 sur 12 » injecté dans le corps
    repeated_min_count: int = 3
    detect_text_toc: bool = True              # sommaire « texte ....... 12 » sans champ
    image_mode: Literal["reference", "embed", "export"] = "reference"
    image_export_dir: str | None = None
    hash_images: bool = True
    read_pixel_size: bool = True              # lit l'en-tête image via Pillow si dispo
    include_table_grid: bool = False          # grille 2D (coûteux, redondant avec table_cells)
    headings: HeadingConfig = field(default_factory=HeadingConfig)
