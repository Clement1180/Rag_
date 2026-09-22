"""Configuration minimale pour faire tourner chunk.py et ses tests.

Si vous avez déjà votre propre config.py, gardez-la : chunk.py lit les champs
ajoutés en v2 (parent_tokens, dedoublonner,
min_section_tokens, titre_dans_parent) avec une valeur par défaut, et
n'exige donc aucune modification de votre Config.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

DATA = Path("data")
PAGES_JSONL = DATA / "pages.jsonl"
CHUNKS_JSONL = DATA / "chunks.jsonl"
PARENTS_JSONL = DATA / "parents.jsonl"


@dataclass
class Config:
    strategie: str = "structurel"            # structurel | recursif | fixe
    chunk_tokens: int = 256
    chunk_overlap: int = 0
    parent_tokens: int | None = None         # None -> 4 × chunk_tokens
    ingest_version: str = "v1"
    prefixe_contexte: bool = True
    tableaux_par_ligne: bool = True
    dedoublonner: bool = True
    dedoublonner_min_tokens: int = 25        # plus court : dépend de sa section, jamais dédoublonné
    min_section_tokens: int = 40             # section plus courte fusionnée avec sa sœur/fille (0 = jamais)
    titre_dans_parent: bool = True           # le parent commence par son fil d'Ariane
    encodeur: str = "intfloat/multilingual-e5-base"
    max_seq_length: int = 512


CFG = Config()


def lire_jsonl(chemin: Path) -> Iterator[dict]:
    with open(chemin, encoding="utf-8") as fh:
        for ligne in fh:
            if ligne.strip():
                yield json.loads(ligne)


def ecrire_jsonl(chemin: Path, lignes: Iterable[dict]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fh:
        for x in lignes:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
