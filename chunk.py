"""Chunking : pages.jsonl | JSON Docling | DOCX -> chunks.jsonl + parents.jsonl

Trois choses que ce module fait et qu'on saute presque toujours :
  - il compte en TOKENS du modèle cible, pas en caractères ;
  - il découple l'unité de recherche (l'enfant) de l'unité de lecture
    (le parent), ce qui dissout le compromis petit/grand ;
  - il préfixe chaque chunk de « document — section — page », qui est le
    remède le plus rentable du cours.

Trois entrées possibles :
  - pages.jsonl, produit par notre ingestion maison (une ligne = une page) ;
  - la sortie JSON de Docling (DoclingDocument v2 : `docling --to json ...`
    ou `doc.save_as_json(...)`), un fichier ou un dossier de fichiers ;
  - des fichiers .docx, lus par docxrag, qui produit un DoclingDocument
    enrichi d'un sidecar `x_docx` (pages approximatives, textes alternatifs…).

Avec Docling, on n'a plus à retrouver les titres dans le texte par `find`, ni
à deviner le titre d'un tableau par sa bbox : l'arbre du document donne
l'ordre de lecture, la hiérarchie des titres, les légendes et la structure
cellule par cellule des tableaux. Conséquence : une section peut enjamber
plusieurs pages, et chaque enfant garde la page de son premier paragraphe.

Ce que la v2 garantit en plus :
  - identifiants stables : dérivés du contenu et de la section, pas de la
    position ; insérer un paragraphe ne renomme que les chunks de SA section ;
  - aucun texte indexé ne dépasse max_seq_length, préfixe compris ;
  - un parent de tableau reste borné (fenêtres de lignes, en-tête répété) ;
  - une page inconnue n'est jamais inventée ; une page estimée est marquée « ~ » ;
  - les figures sans légende restent cherchables (texte alternatif, description) ;
  - les lignes intercalaires d'un tableau (« Lot 2 ») donnent leur contexte
    aux lignes suivantes au lieu d'être jetées ;
  - les passages répétés à l'identique (mentions légales…) ne sont indexés
    qu'une fois par document ;
  - une section trop courte est fusionnée avec sa sœur ou sa fille ;
  - aucun parent ne dépasse parent_tokens, et il commence par son fil d'Ariane.

Champs de Config lus avec une valeur par défaut (une Config v1 reste valide) :
parent_tokens (4 × chunk_tokens), dedoublonner (True), dedoublonner_min_tokens
(25), min_section_tokens (40),
titre_dans_parent (True).

Usage :
    python chunk.py
    python chunk.py --strategie recursif --chunk-tokens 256
    python chunk.py --docling sortie_docling/              # dossier de *.json
    python chunk.py --docling rapport.json --parent-tokens 1024
    python chunk.py --docx dossier_word/                   # via docxrag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from config import (CFG, CHUNKS_JSONL, PAGES_JSONL, PARENTS_JSONL, Config,
                    ecrire_jsonl, lire_jsonl)


def _opt(cfg: Config, nom: str, defaut):
    """Champs ajoutés en v2, lus avec une valeur par défaut : une Config v1 reste valide."""
    v = getattr(cfg, nom, None)
    return defaut if v is None else v


# ─────────────────────────────────────────────────────────────────────────────
# Comptage des tokens
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _tokenizer(nom_modele: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(nom_modele)


def compteur_tokens(nom_modele: str):
    """Un tokenizer par nom de modèle (la v1 gardait le premier chargé, quel
    que soit le nom demandé ensuite). Les comptes sont mis en cache : le
    découpage recompte souvent les mêmes morceaux."""
    tok = _tokenizer(nom_modele)

    @lru_cache(maxsize=1 << 16)
    def n_tokens(texte: str) -> int:
        # add_special_tokens=False : on compte le contenu, pas les [CLS]/[SEP].
        return len(tok.encode(texte, add_special_tokens=False))

    return n_tokens


def compteur_approche():
    """~1,3 token par mot en français pour un tokenizer BPE/WordPiece.
    Suffisant pour un diagnostic, jamais pour la production."""
    return lambda texte: int(len(texte.split()) * 1.3)


# Réserve pour les tokens spéciaux ([CLS]/[SEP] ou <s></s>) et le saut de ligne
# entre préfixe et texte : ce qui est ENCODÉ doit tenir dans max_seq_length.
RESERVE_SPECIAUX = 3


def cible_enfant(cfg: Config, doc_id: str, section: str, n_tokens) -> int:
    """Budget réel du texte d'un enfant, une fois le préfixe déduit.

    La v1 bornait le texte à chunk_tokens puis y ajoutait le préfixe : avec une
    section longue, text_indexed dépassait max_seq_length et l'encodeur
    tronquait silencieusement la FIN du chunk.
    """
    cible = cfg.chunk_tokens
    if cfg.prefixe_contexte:
        # pire cas pour la page (« page ~999 »), la section étant connue
        tete = prefixe(doc_id, section, 998, approx=True)
        cible = min(cible, cfg.max_seq_length - n_tokens(tete) - RESERVE_SPECIAUX)
    return max(16, cible)


# ─────────────────────────────────────────────────────────────────────────────
# Découpage structurel (entrée pages.jsonl)
# ─────────────────────────────────────────────────────────────────────────────
def sections(page: dict) -> list[tuple[str, str]]:
    """Découpe le texte de la page aux titres détectés à l'ingestion.

    Retourne une liste (titre_de_section, texte). Le premier élément peut
    avoir un titre vide : c'est la matière avant le premier titre.
    """
    texte = page.get("text", "")
    titres = [h["text"].strip() for h in page.get("headings", []) if h.get("text")]
    if not texte.strip():
        return []
    if not titres:
        return [("", texte)]

    positions = []
    curseur = 0
    for t in titres:
        i = texte.find(t, curseur)
        if i >= 0:
            positions.append((i, t))
            curseur = i + len(t)
    if not positions:
        return [("", texte)]

    out = []
    if positions[0][0] > 0:
        out.append(("", texte[: positions[0][0]]))
    for (i, t), suivant in zip(positions, positions[1:] + [(len(texte), "")]):
        out.append((t, texte[i: suivant[0]]))
    return [(t, c) for t, c in out if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Découpage récursif si jamais la section structurelle est trop longue
# ─────────────────────────────────────────────────────────────────────────────
SEPARATEURS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]


def decouper_recursif(texte: str, cible: int, n_tokens, seps=None) -> list[str]:
    """Coupe sur le séparateur le plus fort disponible, et descend dans la
    hiérarchie tant qu'un morceau reste trop grand. Recolle ensuite les
    morceaux consécutifs tant que la somme tient dans la cible.

    v2 :
      - le séparateur reste collé au morceau de gauche (la v1 perdait le point
        final d'un chunk coupé après « . ») ;
      - le recollage additionne des comptes précalculés au lieu de retokeniser
        le texte qui grossit (coût quadratique sur un long paragraphe) ; si
        l'estimation se trompe (fusions BPE aux jointures, compteur approché),
        le groupe fautif est re-rempli en comptage exact, au même niveau de
        séparateur, au lieu de redescendre jusqu'au caractère ;
      - les espaces ne sont retirés qu'à la toute fin, jamais entre deux
        morceaux recollés.
    """
    return [m.strip() for m in _decouper(texte, cible, n_tokens,
                                         SEPARATEURS if seps is None else seps) if m.strip()]


def _decouper(texte: str, cible: int, n_tokens, seps: list[str]) -> list[str]:
    if not texte or n_tokens(texte) <= cible or not seps:
        return [texte] if texte else []
    sep, reste = seps[0], seps[1:]
    if sep:
        bruts = texte.split(sep)
        morceaux = [m + sep for m in bruts[:-1]] + [bruts[-1]]
    else:
        morceaux = list(texte)
    if len(morceaux) == 1:
        return _decouper(texte, cible, n_tokens, reste)

    fins: list[tuple[str, int]] = []
    for m in morceaux:
        if not m:
            continue
        n = n_tokens(m)
        if n > cible:
            fins.extend((f, n_tokens(f)) for f in _decouper(m, cible, n_tokens, reste))
        else:
            fins.append((m, n))       # un morceau blanc est gardé : il porte l'espace

    groupes, courant, total = [], [], 0
    for m, n in fins:
        if courant and total + n > cible:
            groupes.append(courant)
            courant, total = [], 0
        courant.append(m)
        total += n
    if courant:
        groupes.append(courant)

    sortie: list[str] = []
    for g in groupes:
        s = "".join(g)
        if len(g) == 1 or n_tokens(s) <= cible:
            sortie.append(s)
            continue
        cur = ""
        for m in g:                       # re-remplissage exact du groupe fautif
            if cur.strip() and n_tokens(cur + m) > cible:
                sortie.append(cur)
                cur = m
            else:
                cur += m
        if cur:
            sortie.append(cur)
    return sortie


def appliquer_chevauchement(morceaux: list[str], overlap_tokens: int,
                            n_tokens, cible: int | None = None) -> list[str]:
    """Fait déborder chaque morceau sur le précédent. Le chevauchement est un
    pansement pour le découpage fixe ; avec du structurel on peut le mettre à 0.

    v2 : la queue est mesurée en vrais tokens (la v1 prenait N MOTS pour N
    tokens), et elle est raccourcie si elle ferait dépasser la cible.
    """
    if overlap_tokens <= 0 or len(morceaux) < 2:
        return morceaux
    sortie = [morceaux[0]]
    for prec, m in zip(morceaux, morceaux[1:]):
        mots = prec.split()
        k, queue = 0, ""
        while k < len(mots) and n_tokens(queue) < overlap_tokens:
            k += 1
            queue = " ".join(mots[-k:])
        if cible is not None:
            while k > 0 and n_tokens(queue + " " + m) > cible:
                k -= 1
                queue = " ".join(mots[-k:]) if k else ""
        sortie.append((queue + " " + m).strip() if queue else m)
    return sortie


# ─────────────────────────────────────────────────────────────────────────────
# Tableaux
# ─────────────────────────────────────────────────────────────────────────────
_VIDE = re.compile(r"^[\s\-—|]*$")


def _rempli(c: str | None) -> bool:
    return bool(c) and not _VIDE.match(c)


def chunks_tableau(table: dict) -> list[tuple[int, str]]:
    """Sérialise un tableau en une ligne = un chunk, avec les en-têtes répétés.

    Elle préserve le lien paramètre <-> valeur, qui est ce qu'une extraction
    linéaire détruit. Retourne des couples (indice de ligne, texte).

    v2 :
      - une ligne intercalaire (« Lot 2 », « Charges de personnel ») n'est plus
        jetée : elle préfixe les lignes suivantes, qui en dépendent ;
      - une ligne de données avec une seule cellule remplie est conservée ;
      - un tableau d'une seule colonne renvoie [] (il rejoint le flux de texte).
    """
    entetes = list(table.get("header") or [])
    rows = table.get("rows", [])
    types = table.get("types") or ["donnee"] * len(rows)
    n_col = max([len(entetes)] + [len(r) for r in rows] + [0])
    if n_col < 2:
        return []
    sortie, contexte = [], ""
    for i, (ligne, genre) in enumerate(zip(rows, types)):
        cellules = [c for c in ligne if _rempli(c)]
        if not cellules:
            continue
        if genre == "intercalaire":
            contexte = cellules[0].strip()
            continue
        paires = []
        for j, cellule in enumerate(ligne):
            if not _rempli(cellule):
                continue
            col = entetes[j] if j < len(entetes) and entetes[j] else f"col{j}"
            paires.append(f"{col} : {cellule.strip()}")
        texte = " ; ".join(paires)
        sortie.append((i, f"[{contexte}] {texte}" if contexte else texte))
    return sortie


def _cellule_md(c: str | None) -> str:
    # Un « | » ou un saut de ligne dans une cellule casse la table Markdown.
    return (c or "").replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _entete_md(table: dict) -> str:
    rows = table.get("rows", [])
    n_col = max([len(table.get("header") or [])] + [len(r) for r in rows] + [1])
    entetes = list(table.get("header") or []) + [""] * n_col
    entetes = [_cellule_md(e) or f"col{i}" for i, e in enumerate(entetes[:n_col])]
    return "| " + " | ".join(entetes) + " |\n|" + "|".join("---" for _ in entetes) + "|"


def _ligne_md(r: list[str]) -> str:
    return "| " + " | ".join(_cellule_md(c) for c in r) + " |"


def _markdown(table: dict) -> str:
    """Forme tabulaire complète, stockée pour la GÉNÉRATION."""
    return "\n".join([_entete_md(table)] + [_ligne_md(r) for r in table.get("rows", [])])


def fenetres_tableau(table: dict, legende: str, limite: int, n_tokens) -> list[dict]:
    """Découpe un tableau en parents bornés à `limite` tokens.

    La v1 faisait du tableau entier UN parent : une seule ligne retrouvée dans
    un tableau de 400 lignes injectait tout le tableau dans le contexte du
    LLM. Chaque fenêtre répète l'en-tête et la légende, indique « lignes a–b /
    N », et reporte la dernière ligne intercalaire si elle commence au milieu
    d'un groupe. Retourne [{"text", "debut", "fin"}] (fin exclue).
    """
    rows = table.get("rows", [])
    if not rows:
        return []
    types = table.get("types") or ["donnee"] * len(rows)
    tete = _entete_md(table)
    lignes = [_ligne_md(r) for r in rows]
    comptes = [n_tokens(l) for l in lignes]
    fixe = n_tokens(legende) + n_tokens(tete) + 12      # 12 ≈ « (lignes 120–180 / 400) »

    bornes: list[tuple[int, int, int | None]] = []
    debut, total, report, dernier_inter = 0, fixe, None, None
    for i, n in enumerate(comptes):
        if i > debut and total + n > limite:
            bornes.append((debut, i, report))
            debut = i
            report = dernier_inter if types[i] != "intercalaire" else None
            total = fixe + (comptes[report] if report is not None else 0)
        if types[i] == "intercalaire":
            dernier_inter = i
        total += n
    bornes.append((debut, len(rows), report))

    sortie = []
    for a, b, rep in bornes:
        corps = ([lignes[rep]] if rep is not None else []) + lignes[a:b]
        mention = f"(lignes {a + 1}–{b} / {len(rows)})" if len(bornes) > 1 else ""
        titre = " ".join(x for x in (legende, mention) if x)
        texte = "\n\n".join(x for x in (titre, tete + "\n" + "\n".join(corps)) if x)
        sortie.append({"text": texte, "debut": a, "fin": b})
    return sortie


# ─────────────────────────────────────────────────────────────────────────────
# Identifiants, préfixe, fabrication des chunks
# ─────────────────────────────────────────────────────────────────────────────
class Identifiants:
    """Identifiants dérivés du contenu et de la section, pas de la position.

    La v1 mettait l'ordinal et la page dans la graine : insérer un paragraphe
    en tête de document décalait tous les ordinaux, donc renommait TOUS les
    chunks, et la mise à jour incrémentale réindexait tout. Ici, un chunk
    garde son identifiant tant que son texte et sa section ne changent pas.
    `occurrence` (rang du même texte dans la même section) ne sert qu'à
    distinguer deux passages strictement identiques.

    Limite assumée : dans la section modifiée elle-même, le regroupement
    glouton peut déplacer des frontières, donc renommer quelques voisins.
    Les autres sections ne bougent pas.
    """

    def __init__(self, doc_id: str, ingest_version: str):
        self.doc_id, self.version = doc_id, ingest_version
        self._vus: Counter = Counter()

    def __call__(self, genre: str, section: str, texte: str) -> str:
        h = hashlib.sha256(texte.encode("utf-8")).hexdigest()
        cle = (genre, section, h)
        occurrence = self._vus[cle]
        self._vus[cle] += 1
        graine = f"{self.doc_id}:{genre}:{section}:{h}:{occurrence}:{self.version}"
        return hashlib.sha256(graine.encode("utf-8")).hexdigest()[:16]


def prefixe(doc_id: str, section: str, page: int | None, approx: bool = False) -> str:
    """« document — section — page », préfixé au texte indexé.

    Une page inconnue est omise : un DOCX n'a pas de pagination calculée, et
    « page 1 » partout serait une information fausse. Une page estimée
    (docxrag : sauts de page explicites et derniers sauts rendus par Word)
    est marquée « ~ ».
    """
    bouts = [doc_id.upper()]
    s = (section or "").strip()
    if s:
        bouts.append(s if len(s) <= 80 else "…" + s[-79:])
    if page is not None:
        bouts.append(f"page ~{page + 1}" if approx else f"page {page + 1}")
    return " — ".join(bouts)


def _normaliser(texte: str) -> str:
    return re.sub(r"\s+", " ", texte).strip().lower()


class Sortie:
    """Accumule chunks et parents d'un document ; porte identifiants et dédoublonnage."""

    def __init__(self, cfg: Config, doc_id: str, n_tokens):
        self.cfg, self.doc_id, self.n_tokens = cfg, doc_id, n_tokens
        self.ids = Identifiants(doc_id, cfg.ingest_version)
        self.chunks: list[dict] = []
        self.parents: list[dict] = []
        self.doublons = 0
        self._vus: dict[str, dict] = {}          # chunks (figures, pages.jsonl)
        self._vus_para: dict[str, dict] = {}     # paragraphes (entrée Docling)
        self._dedup = bool(_opt(cfg, "dedoublonner", True))
        self._dedup_min = int(_opt(cfg, "dedoublonner_min_tokens", 25))

    def parent(self, genre: str, section: str, texte: str, page, approx: bool,
               pages: list | None = None, **extra) -> str:
        pid = f"{genre}_{self.ids(genre, section, texte)}"
        self.parents.append({
            "parent_id": pid, "doc_id": self.doc_id, "page": page, "page_approx": approx,
            "pages": pages if pages is not None else ([] if page is None else [page]),
            "section": section, "text": texte, "n_tokens": self.n_tokens(texte), **extra,
        })
        return pid

    def paragraphe_repete(self, para: dict) -> bool:
        """Dédoublonnage AVANT regroupement. Au niveau du chunk, il ne jouerait
        presque jamais : une mention légale est recollée au texte voisin, et
        deux chunks qui la contiennent diffèrent. Un paragraphe répeté (assez
        long pour être autonome) est donc écarté dès la lecture ; les chunks
        qui contiennent sa première occurrence reçoivent `autres_sections`."""
        if not self._dedup or para["n"] < self._dedup_min:
            return False
        cle = _normaliser(para["text"])
        premier = self._vus_para.get(cle)
        if premier is None:
            para["_chunks"] = []
            self._vus_para[cle] = para
            return False
        self.doublons += 1
        sec, autres = para["section"], premier.setdefault("autres_sections", [])
        if sec != premier["section"] and sec not in autres and len(autres) < 20:
            autres.append(sec)
            for c in premier["_chunks"]:
                c.setdefault("autres_sections", [])
                if sec not in c["autres_sections"]:
                    c["autres_sections"].append(sec)
        return True

    def enfant(self, parent_id: str, section: str, block_type: str, texte: str,
               page, approx: bool, source: str = "native", est_figure: bool = False,
               paras: list[dict] | None = None) -> dict | None:
        texte = texte.strip()
        if not texte:
            return None
        # Passage répété à l'identique (mentions légales, avertissements) : indexé
        # une fois par document. Deux exceptions, parce que leur sens dépend de
        # la section : les lignes de tableau, et les textes courts (« Sans
        # objet. » sous Pénalités ≠ « Sans objet. » sous Garanties). Le chunk
        # conservé garde la liste des autres sections où le passage figure.
        cle = None
        if self._dedup and block_type != "table" and self.n_tokens(texte) >= self._dedup_min:
            cle = _normaliser(texte)
            premier = self._vus.get(cle)
            if premier is not None:
                self.doublons += 1
                autres = premier.setdefault("autres_sections", [])
                if section != premier["section"] and section not in autres and len(autres) < 20:
                    autres.append(section)
                return None
        tete = prefixe(self.doc_id, section, page, approx) if self.cfg.prefixe_contexte else ""
        indexe = (tete + "\n" + texte) if tete else texte
        self.chunks.append({
            "chunk_id": self.ids("c", section, texte),
            "parent_id": parent_id,
            "doc_id": self.doc_id, "page": page, "page_approx": approx,
            "section": section, "block_type": block_type, "source": source,
            "text": texte,
            # text_indexed est ce qu'on ENCODE ; text est ce qu'on AFFICHE.
            "text_indexed": indexe,
            "n_tokens": self.n_tokens(texte),
            "n_tokens_indexed": self.n_tokens(indexe),
            "est_figure": est_figure,
            "ingest_version": self.cfg.ingest_version,
        })
        chunk = self.chunks[-1]
        if cle is not None:
            self._vus[cle] = chunk
        for p in paras or ():                     # paragraphes dédoublonnables qu'il contient
            if "_chunks" in p:
                p["_chunks"].append(chunk)
                for sec in p.get("autres_sections", []):
                    chunk.setdefault("autres_sections", [])
                    if sec not in chunk["autres_sections"]:
                        chunk["autres_sections"].append(sec)
        return chunk


    def finaliser(self) -> None:
        """Retire les parents dont tous les enfants étaient des doublons : ils ne
        seraient jamais atteints par la recherche."""
        utilises = {c["parent_id"] for c in self.chunks}
        self.parents = [p for p in self.parents if p["parent_id"] in utilises]


def _emettre_table(sortie: Sortie, table: dict, legende: str, section: str,
                   page, approx: bool, limite: int, source: str) -> bool:
    """Tableau -> parents fenêtrés + une ligne = un enfant. Retourne False si le
    tableau n'a pas de lignes exploitables (l'appelant le verse alors au texte)."""
    cfg, n_tokens = sortie.cfg, sortie.n_tokens
    lignes = chunks_tableau(table) if cfg.tableaux_par_ligne else []
    if not lignes:
        return False
    titre = " > ".join(x for x in (section, legende) if x)
    cible = cible_enfant(cfg, sortie.doc_id, titre, n_tokens)
    for f in fenetres_tableau(table, legende, limite, n_tokens):
        pid = sortie.parent("t", titre, f["text"], page, approx,
                            lignes_tableau=[f["debut"], f["fin"]])
        for i, texte in lignes:
            if not f["debut"] <= i < f["fin"]:
                continue
            # Cellule géante (texte libre dans un tableau) : on la redécoupe.
            morceaux = [texte] if n_tokens(texte) <= cible else decouper_recursif(texte, cible, n_tokens)
            for m in morceaux:
                sortie.enfant(pid, titre, "table", m, page, approx, source=source)
    return True


def _limite_parent(cfg: Config) -> int:
    return _opt(cfg, "parent_tokens", 4 * cfg.chunk_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Entrée pages.jsonl (ingestion maison)
# ─────────────────────────────────────────────────────────────────────────────
def construire(pages, cfg: Config, n_tokens) -> tuple[list[dict], list[dict]]:
    chunks: list[dict] = []
    parents: list[dict] = []
    sorties: dict[str, Sortie] = {}
    limite = _limite_parent(cfg)

    for page in pages:
        if page.get("etat") == "error":
            continue
        doc_id, num = page.get("doc_id", "unknown"), page.get("page", 0)
        s = sorties.setdefault(doc_id, Sortie(cfg, doc_id, n_tokens))
        source = page.get("source", "native")

        unites = (sections(page) if cfg.strategie == "structurel"
                  else [("", page.get("text", ""))])

        for titre, corps in unites:
            if not corps.strip():
                continue
            cible = cible_enfant(cfg, doc_id, titre, n_tokens)
            # v2 : une section de page plus longue que parent_tokens donne
            # plusieurs parents au lieu d'un seul parent démesuré.
            for bloc in decouper_recursif(corps, limite, n_tokens):
                parent = s.parent("p", titre, bloc, num, False)
                if cfg.strategie == "fixe":
                    mots = bloc.split()
                    pas = max(1, min(cfg.chunk_tokens, cible))
                    enfants = [" ".join(mots[i:i + pas]) for i in range(0, len(mots), pas)]
                else:
                    enfants = decouper_recursif(bloc, cible, n_tokens)
                enfants = appliquer_chevauchement(enfants, cfg.chunk_overlap, n_tokens, cible)
                for e in enfants:
                    s.enfant(parent, titre, "text", e, num, False, source=source,
                             est_figure=page.get("est_figure", False))

        for table in page.get("tables", []):
            titre_t = _titre_au_dessus(page, table.get("bbox"))
            _emettre_table(s, table, "", titre_t, num, False, limite, source)

    for s in sorties.values():
        s.finaliser()
        chunks += s.chunks
        parents += s.parents
    return chunks, parents


def _titre_au_dessus(page: dict, bbox) -> str:
    """Titre dont la boîte est la plus basse parmi celles au-dessus du tableau."""
    if not bbox:
        return ""
    y_table = bbox[1]
    candidats = [h for h in page.get("headings", [])
                 if h.get("bbox") and h["bbox"][1] < y_table]
    if not candidats:
        return ""
    return max(candidats, key=lambda h: h["bbox"][1])["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Entrée Docling (DoclingDocument v2, JSON)
# ─────────────────────────────────────────────────────────────────────────────
# On lit le JSON brut plutôt que d'importer docling_core : le module reste
# utilisable sans Docling installé, et un DoclingDocument en mémoire passe
# aussi via doc.export_to_dict().

LABELS_IGNORES = {"page_header", "page_footer"}   # en-têtes/pieds répétés


def charger_docling(chemin: Path) -> list[dict]:
    fichiers = sorted(chemin.glob("*.json")) if chemin.is_dir() else [chemin]
    docs = []
    for f in fichiers:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("schema_name") != "DoclingDocument":
            print(f"  ignoré (pas un DoclingDocument v2) : {f.name}")
            continue
        d["_fichier"] = f.stem
        docs.append(d)
    return docs


def charger_docx(chemin: Path) -> list[dict]:
    """.docx -> DoclingDocument (dict) via docxrag, métadonnées en sidecar x_docx."""
    from docxrag import parse_docx, to_docling_dict
    fichiers = sorted(chemin.glob("*.docx")) if chemin.is_dir() else [chemin]
    docs = []
    for f in fichiers:
        if f.name.startswith("~$"):                 # fichier verrou de Word
            continue
        try:
            res = parse_docx(str(f))
        except Exception as exc:                    # .doc renommé, zip corrompu…
            print(f"  ignoré ({type(exc).__name__}: {exc}) : {f.name}")
            continue
        for w in res.warnings:
            print(f"  {f.name} : {w}")
        d = to_docling_dict(res.model)
        d["_fichier"] = f.stem
        docs.append(d)
    return docs


def _doc_id(doc: dict) -> str:
    nom = ((doc.get("origin") or {}).get("filename")
           or doc.get("name") or doc.get("_fichier") or "unknown")
    return Path(nom).stem


def _resoudre(doc: dict, ref: dict) -> dict:
    """{"$ref": "#/texts/12"} -> doc["texts"][12]"""
    obj = doc
    for p in ref["$ref"].split("/")[1:]:
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    return obj


def _x(doc: dict, item: dict) -> dict:
    """Métadonnées docxrag d'un item : sidecar `x_docx` (défaut) ou `meta.docx__props`
    (mode inline). Dictionnaire vide pour un JSON Docling ordinaire."""
    ref = item.get("self_ref")
    side = ((doc.get("x_docx") or {}).get("items") or {}).get(ref)
    if side:
        return side
    return (item.get("meta") or {}).get("docx__props") or {}


def _page(doc: dict, item: dict, defaut: tuple) -> tuple[int | None, bool]:
    """(page à partir de 0, approximative ?). Docling numérote à partir de 1.

    Ordre : `prov` (PDF, page exacte) -> `page_hint` docxrag (estimée) ->
    page du bloc précédent. Au tout début d'un DOCX sans indice : None.
    """
    prov = item.get("prov") or []
    if prov:
        return prov[0]["page_no"] - 1, False
    hint = _x(doc, item).get("page_hint")
    if hint:
        return int(hint) - 1, True
    return defaut


def _texte(item: dict) -> str:
    t = (item.get("text") or "").strip()
    if t and item.get("label") == "list_item" and item.get("marker"):
        t = f"{item['marker']} {t}"
    return t


def _legende(doc: dict, item: dict) -> str:
    return " ".join(_texte(_resoudre(doc, r)) for r in item.get("captions", [])).strip()


# Texte alternatif généré automatiquement par Office (« Une image contenant
# texte, capture d'écran… Description générée automatiquement ») : du bruit.
_ALT_AUTO = re.compile(r"(g[ée]n[ée]r[ée]e? automatiquement|generated automatically|"
                       r"automatisch generiert)\s*\.?\s*$", re.I)
_NOM_GENERIQUE = re.compile(r"^(image|picture|graphique|chart|diagramme|objet|object|"
                            r"zone de texte|text box)\s*\d*$", re.I)


def texte_figure(doc: dict, item: dict) -> str:
    """Tout ce qui rend une figure cherchable : légende, texte alternatif,
    titre, description (Docling récent : meta.description ; ancien :
    annotations ; docxrag : sidecar) et valeurs d'un graphique.

    La v1 ne gardait que légende + annotations : une image de DOCX sans
    légende mais avec texte alternatif était perdue.
    """
    x = _x(doc, item)
    if x.get("decorative"):
        return ""
    vus, bouts = set(), []

    def ajouter(t: str | None, etiquette: str = "") -> None:
        t = (t or "").strip()
        if not t or _NOM_GENERIQUE.match(t) or _normaliser(t) in vus:
            return
        vus.add(_normaliser(t))
        bouts.append(f"{etiquette}{t}")

    ajouter(_legende(doc, item))
    alt = x.get("alt_text") or ""
    if not _ALT_AUTO.search(alt):
        ajouter(alt)
    ajouter(x.get("title"))
    desc = ((item.get("meta") or {}).get("description") or {}).get("text")
    desc = desc or (x.get("description") or {}).get("text")
    desc = desc or " ".join((a.get("text") or "") for a in item.get("annotations", [])
                            if a.get("kind") == "description")
    ajouter(desc, "Description : ")
    ajouter(x.get("chart_text"), "Données : ")
    return "\n".join(bouts)


def blocs_docling(doc: dict) -> Iterator[dict]:
    """Parcourt l'arbre `body` dans l'ordre de lecture et produit des blocs
    {kind: heading|text|table|picture, page, page_approx, label, ...}."""
    etat = {"page": (None, False)}

    def visiter(noeud: dict) -> Iterator[dict]:
        for ref in noeud.get("children", []):
            item = _resoudre(doc, ref)
            if item.get("content_layer", "body") != "body":   # furniture
                continue
            label = item.get("label", "")
            if label in LABELS_IGNORES:
                continue
            etat["page"] = _page(doc, item, etat["page"])
            page, approx = etat["page"]
            base = {"page": page, "page_approx": approx, "label": label}

            if label == "inline":
                # Groupe « en ligne » : fragments d'un même paragraphe
                # (texte + formule + lien...) -> on les recolle.
                bouts = [_texte(_resoudre(doc, c)) for c in item.get("children", [])]
                texte = " ".join(b for b in bouts if b)
                if texte:
                    yield {**base, "kind": "text", "label": "text", "text": texte}
                continue
            if label == "table":
                # Pas de descente : les légendes sont lues via item["captions"].
                yield {**base, "kind": "table", "item": item}
                continue
            if label in ("picture", "chart"):
                yield {**base, "kind": "picture", "item": item}
                continue

            if label in ("title", "section_header"):
                yield {**base, "kind": "heading", "text": (item.get("text") or "").strip(),
                       "level": 0 if label == "title" else item.get("level", 1)}
            elif "text" in item:                   # text, list_item, code, formula, footnote...
                texte = _texte(item)
                if texte:
                    yield {**base, "kind": "text", "text": texte}
            # Groupes (listes, chapitres...) et éléments ayant des enfants.
            yield from visiter(item)

    yield from visiter(doc["body"])


def _grille(data: dict) -> list[list[dict]]:
    """Grille cellule par cellule. Les cellules fusionnées y sont répétées à
    chaque position couverte, ce qui est exactement ce qu'on veut pour
    sérialiser ligne par ligne."""
    if data.get("grid"):
        return data["grid"]
    n_l, n_c = data.get("num_rows", 0), data.get("num_cols", 0)
    grille = [[{"text": ""} for _ in range(n_c)] for _ in range(n_l)]
    for c in data.get("table_cells", []):
        for i in range(c["start_row_offset_idx"], min(c["end_row_offset_idx"], n_l)):
            for j in range(c["start_col_offset_idx"], min(c["end_col_offset_idx"], n_c)):
                grille[i][j] = c
    return grille


def _origine(c: dict) -> tuple | None:
    """Identité d'une cellule fusionnée (sa cellule d'origine) ; None si inconnue."""
    r, k = c.get("start_row_offset_idx"), c.get("start_col_offset_idx")
    return None if r is None or k is None else (r, k)


def table_docling(item: dict) -> dict:
    """TableItem Docling -> {"header", "rows", "types"}, le format attendu par
    chunks_tableau, fenetres_tableau et _markdown.

    v2 :
      - une fusion HORIZONTALE n'est écrite qu'une fois (la v1 produisait
        « Col2 : Lot 2 ; Col3 : Lot 2 ; Col4 : Lot 2 ») ; une fusion VERTICALE
        reste répétée, pour que chaque ligne se suffise à elle-même ;
      - une ligne dont l'unique valeur couvre plusieurs colonnes, ou marquée
        row_section, est typée « intercalaire ».
    """
    grille = _grille(item.get("data") or {})
    if not grille:
        return {"header": [], "rows": [], "types": []}

    # Lignes d'en-tête = lignes de tête dont toutes les cellules remplies
    # sont marquées column_header (TableFormer, ou w:tblHeader pour docxrag).
    n_entete = 0
    for ligne in grille:
        remplies = [c for c in ligne if (c.get("text") or "").strip()]
        if remplies and all(c.get("column_header") for c in remplies):
            n_entete += 1
        else:
            break
    # Heuristique : aucun en-tête détecté -> la première ligne en tient lieu.
    if n_entete == 0 and len(grille) > 1:
        n_entete = 1
    if n_entete >= len(grille):
        n_entete = len(grille) - 1

    # En-têtes sur plusieurs lignes : « Tension / Min », « Tension / Max ».
    n_col = max(len(l) for l in grille)
    entetes = []
    for j in range(n_col):
        bouts: list[str] = []
        for ligne in grille[:n_entete]:
            t = (ligne[j].get("text") or "").strip() if j < len(ligne) else ""
            if t and (not bouts or bouts[-1] != t):
                bouts.append(t)
        entetes.append(" / ".join(bouts))

    rows, types = [], []
    for ligne in grille[n_entete:]:
        r, prec = [], None
        for c in ligne:
            orig = _origine(c)
            doublon = orig is not None and orig == prec
            r.append("" if doublon else (c.get("text") or "").strip())
            prec = orig
        remplies = [t for t in r if _rempli(t)]
        large = any((c.get("col_span") or 1) > 1 for c in ligne if (c.get("text") or "").strip())
        section = any(c.get("row_section") for c in ligne)
        intercalaire = n_col >= 2 and len(remplies) == 1 and (large or section)
        rows.append(r)
        types.append("intercalaire" if intercalaire else "donnee")
    return {"header": entetes, "rows": rows, "types": types}


def _para(texte: str, bloc: dict, section: str, n_tokens) -> dict:
    return {"text": texte, "label": bloc["label"], "page": bloc["page"],
            "page_approx": bloc.get("page_approx", False),
            "section": section, "n": n_tokens(texte)}


def _grouper(paras: list[dict], limite: int) -> list[list[dict]]:
    """Regroupement glouton de paragraphes entiers tant que ça tient. On somme
    les comptes précalculés au lieu de retokeniser le texte recollé.

    v2 : un titre ne termine jamais un groupe ; il passe au groupe suivant,
    avec le contenu qu'il annonce.
    """
    groupes, courant, total = [], [], 0
    for p in paras:
        if courant and total + p["n"] > limite:
            report = []
            if len(courant) > 1 and courant[-1]["label"] in LABELS_TITRES:
                report = [courant.pop()]
            groupes.append(courant)
            courant, total = report, sum(x["n"] for x in report)
        courant.append(p)
        total += p["n"]
    if courant:
        groupes.append(courant)
    return groupes


LABELS_TITRES = {"title", "section_header"}


def _section_commune(groupe: list[dict]) -> str:
    """Plus long préfixe commun des fils d'Ariane d'un groupe. Après fusion de
    « 1 > 1.1 » et « 1 > 1.2 », le groupe appartient à « 1 », pas à « 1 > 1.1 »."""
    chemins = [p["section"].split(" > ") if p["section"] else [] for p in groupe]
    commun: list[str] = []
    for bouts in zip(*chemins):
        if any(b != bouts[0] for b in bouts):
            break
        commun.append(bouts[0])
    return " > ".join(commun)


def _borner(paras: list[dict], limite: int, n_tokens) -> list[dict]:
    """Un paragraphe plus long que la limite d'un parent (clause de 3 pages,
    tableau d'une colonne versé au texte) est redécoupé : sans cela, le
    parent qui le contient dépasserait parent_tokens."""
    out = []
    for p in paras:
        if p["n"] <= limite:
            out.append(p)
            continue
        for m in decouper_recursif(p["text"], limite, n_tokens):
            out.append({**p, "text": m, "n": n_tokens(m)})
    return out


def _joindre(groupe: list[dict]) -> str:
    out = groupe[0]["text"]
    for prec, p in zip(groupe, groupe[1:]):
        liste = prec["label"] == "list_item" and p["label"] == "list_item"
        out += ("\n" if liste else "\n\n") + p["text"]
    return out


def _emettre_texte(sortie: Sortie, paras: list[dict], parent_tokens: int) -> None:
    """Paragraphes d'une section -> parents (≤ parent_tokens) -> enfants
    (≤ cible, préfixe déduit). Un parent borné évite qu'une section de 20 pages
    devienne une « unité de lecture » inutilisable par le LLM.

    v2 : le parent commence par son fil d'Ariane (option `titre_dans_parent`),
    car le LLM lit le parent sans le préfixe de l'enfant ; la place de ce titre
    est réservée dans le budget du parent.
    """
    cfg, n_tokens, doc_id = sortie.cfg, sortie.n_tokens, sortie.doc_id
    avec_titre = bool(_opt(cfg, "titre_dans_parent", True))
    reserve = (max(n_tokens(p["section"]) for p in paras) + 2) if avec_titre else 0
    limite = max(32, parent_tokens - reserve)
    for groupe in _grouper(_borner(paras, limite, n_tokens), limite):
        tete = groupe[0]
        section = _section_commune(groupe)
        corps = _joindre(groupe)
        titre_deja_la = tete["label"] in LABELS_TITRES and section.endswith(tete["text"])
        if avec_titre and section and not titre_deja_la:
            corps = f"{section}\n\n{corps}"
        pages = sorted({p["page"] for p in groupe if p["page"] is not None})
        parent_id = sortie.parent("p", section, corps, tete["page"],
                                  tete["page_approx"], pages=pages)
        cible = cible_enfant(cfg, doc_id, section, n_tokens)

        if cfg.strategie == "fixe":
            mots, pas = _joindre(groupe).split(), max(1, cible)
            enfants = [(" ".join(mots[i:i + pas]), {**tete, "section": section}, groupe)
                       for i in range(0, len(mots), pas)]
        else:
            # On coupe AUX FRONTIÈRES DE PARAGRAPHES données par Docling ; seul
            # un paragraphe à lui seul trop long passe par le récursif.
            enfants = []
            for g in _grouper(groupe, cible):
                t = _joindre(g)
                origine = {**g[0], "section": _section_commune(g)}
                if len(g) == 1 and g[0]["n"] > cible:
                    enfants += [(m, origine, g) for m in decouper_recursif(t, cible, n_tokens)]
                else:
                    enfants.append((t, origine, g))

        textes = appliquer_chevauchement([t for t, _, _ in enfants], cfg.chunk_overlap,
                                         n_tokens, cible)
        for (_, origine, g), texte in zip(enfants, textes):
            sortie.enfant(parent_id, origine["section"], "text", texte,
                          origine["page"], origine["page_approx"], source=_source(sortie),
                          paras=g)


def _source(sortie: Sortie) -> str:
    return getattr(sortie, "source", "docling")


def construire_docling(docs: list[dict], cfg: Config, n_tokens,
                       parent_tokens: int) -> tuple[list[dict], list[dict]]:
    """Entrée Docling. `docs` : dicts DoclingDocument (JSON chargé, sortie de
    docxrag, ou doc.export_to_dict() si on part d'un objet Docling en mémoire)."""
    chunks: list[dict] = []
    parents: list[dict] = []
    doublons = 0

    for doc in docs:
        doc_id = _doc_id(doc)
        s = Sortie(cfg, doc_id, n_tokens)
        s.source = "docxrag" if "x_docx" in doc else "docling"
        chemin: list[tuple[int, str]] = []    # pile (niveau, titre)
        paras: list[dict] = []
        min_section = _opt(cfg, "min_section_tokens", 40)
        titre_dans_paras = cfg.strategie != "structurel" or not cfg.prefixe_contexte
        tampon = {"niveau": None, "titre": None, "insere": False}

        def vider():
            if paras:
                _emettre_texte(s, paras, parent_tokens)
                paras.clear()

        for b in blocs_docling(doc):
            if b["kind"] == "heading":
                if not b["text"]:
                    continue
                # structurel : un titre ferme la section. Sinon le titre n'est
                # qu'un paragraphe de plus, mais il met quand même à jour le
                # fil d'Ariane utilisé dans le préfixe.
                #
                # v2 : une section trop courte (< min_section_tokens) n'est pas
                # fermée si la suivante est sa sœur ou sa fille : « 1 Objet »
                # (deux lignes) et « 1.1 Périmètre » forment une seule unité, au
                # lieu d'un chunk de dix mots sans contenu utile. On ne fusionne
                # jamais vers un niveau supérieur (autre chapitre).
                fusion = False
                if cfg.strategie == "structurel":
                    total = sum(p["n"] for p in paras)
                    fusion = bool(paras) and total < min_section and \
                        tampon["niveau"] is not None and b["level"] >= tampon["niveau"]
                    if not fusion:
                        vider()
                        tampon.update(niveau=b["level"], titre=None, insere=False)
                while chemin and chemin[-1][0] >= b["level"]:
                    chemin.pop()
                chemin.append((b["level"], b["text"]))
                para_titre = _para(b["text"], b, " > ".join(t for _, t in chemin), n_tokens)
                if titre_dans_paras:
                    # Sans préfixe, le titre n'apparaîtrait nulle part : on le
                    # garde comme premier paragraphe de sa section.
                    paras.append(para_titre)
                elif fusion:
                    # Sections fusionnées : leurs titres entrent dans le texte,
                    # sinon on ne saurait plus où commence chacune.
                    if not tampon["insere"] and tampon["titre"]:
                        paras.insert(0, tampon["titre"])
                        tampon["insere"] = True
                    paras.append(para_titre)
                else:
                    tampon["titre"] = para_titre
                continue

            section = " > ".join(t for _, t in chemin)

            if b["kind"] == "text":
                para = _para(b["text"], b, section, n_tokens)
                if not s.paragraphe_repete(para):
                    paras.append(para)

            elif b["kind"] == "table":
                table = table_docling(b["item"])
                if not table["rows"]:
                    continue
                legende = _legende(doc, b["item"])
                if not _emettre_table(s, table, legende, section, b["page"], b["page_approx"],
                                      parent_tokens, s.source):
                    # Option désactivée ou tableau à une colonne : le tableau
                    # entier rejoint le flux de texte plutôt que d'être perdu.
                    paras.append(_para((legende + "\n\n" + _markdown(table)).strip(),
                                       b, section, n_tokens))

            elif b["kind"] == "picture":
                texte = texte_figure(doc, b["item"])
                if not texte:
                    continue
                pid = s.parent("f", section, texte, b["page"], b["page_approx"])
                cible = cible_enfant(cfg, doc_id, section, n_tokens)
                for m in decouper_recursif(texte, cible, n_tokens):
                    s.enfant(pid, section, "figure", m, b["page"], b["page_approx"],
                             source=s.source, est_figure=True)

        vider()
        s.finaliser()
        chunks += s.chunks
        parents += s.parents
        doublons += s.doublons

    construire_docling.doublons = doublons
    return chunks, parents


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic
# ─────────────────────────────────────────────────────────────────────────────
def diagnostic(chunks: list[dict], parents: list[dict], cfg: Config) -> list[str]:
    """Statistiques et alertes. On contrôle text_indexed, ce qui est ENCODÉ."""
    lignes = [f"{len(chunks)} chunks, {len(parents)} parents"]
    if not chunks:
        return lignes

    def quantiles(valeurs: list[int]) -> str:
        v = sorted(valeurs)
        def q(p): return v[min(len(v) - 1, int(len(v) * p))]
        return f"med={q(.5)}  p75={q(.75)}  p95={q(.95)}  p99={q(.99)}  max={v[-1]}"

    lignes.append("tokens texte   " + quantiles([c["n_tokens"] for c in chunks]))
    indexes = [c.get("n_tokens_indexed", c["n_tokens"]) for c in chunks]
    lignes.append("tokens indexés " + quantiles(indexes))
    depassent = sum(1 for n in indexes if n + RESERVE_SPECIAUX - 1 > cfg.max_seq_length)
    if depassent:
        lignes.append(f"  ATTENTION : {depassent} chunk(s) dépassent max_seq_length="
                      f"{cfg.max_seq_length} une fois le préfixe ajouté -> troncature silencieuse")
    petits = sum(1 for c in chunks if c["block_type"] == "text" and c["n_tokens"] < 20)
    if petits > 0.2 * len(chunks):
        lignes.append(f"  NOTE : {petits} chunks de texte < 20 tokens (sections très courtes ?)")
    lignes.append("types : " + str({t: sum(1 for c in chunks if c["block_type"] == t)
                                    for t in ("text", "table", "figure")}))
    sans_page = sum(1 for c in chunks if c["page"] is None)
    approx = sum(1 for c in chunks if c.get("page_approx"))
    if sans_page or approx:
        lignes.append(f"pages : {approx} estimées, {sans_page} inconnues")
    gros = max(p["n_tokens"] for p in parents)
    lignes.append(f"parent le plus long : {gros} tokens")
    return lignes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pages.jsonl | JSON Docling | DOCX -> chunks.jsonl")
    ap.add_argument("--pages", default=str(PAGES_JSONL))
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--docling", help="JSON Docling (fichier ou dossier de *.json) "
                                       "à utiliser à la place de --pages")
    src.add_argument("--docx", help="fichier .docx ou dossier, lu par docxrag")
    ap.add_argument("--out", default=str(CHUNKS_JSONL))
    ap.add_argument("--parents", default=str(PARENTS_JSONL))
    ap.add_argument("--strategie", choices=["structurel", "recursif", "fixe"])
    ap.add_argument("--chunk-tokens", type=int)
    ap.add_argument("--parent-tokens", type=int,
                    help="taille max d'un parent (défaut 4 × chunk-tokens)")
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--sans-dedoublonnage", action="store_true")
    ap.add_argument("--sans-tokenizer", action="store_true",
                    help="comptage approché, pour un diagnostic rapide")
    args = ap.parse_args(argv)

    cfg = CFG
    if args.strategie:
        cfg.strategie = args.strategie
    if args.chunk_tokens:
        cfg.chunk_tokens = args.chunk_tokens
    if args.overlap is not None:
        cfg.chunk_overlap = args.overlap
    if args.parent_tokens:
        cfg.parent_tokens = args.parent_tokens
    if args.sans_dedoublonnage:
        cfg.dedoublonner = False

    n_tokens = compteur_approche() if args.sans_tokenizer else compteur_tokens(cfg.encodeur)

    if args.docling or args.docx:
        docs = (charger_docling(Path(args.docling)) if args.docling
                else charger_docx(Path(args.docx)))
        print(f"{len(docs)} document(s)")
        chunks, parents = construire_docling(docs, cfg, n_tokens, _limite_parent(cfg))
        if getattr(construire_docling, "doublons", 0):
            print(f"{construire_docling.doublons} passage(s) répété(s) non réindexé(s)")
    else:
        pages = list(lire_jsonl(Path(args.pages)))
        chunks, parents = construire(pages, cfg, n_tokens)

    ecrire_jsonl(Path(args.out), chunks)
    ecrire_jsonl(Path(args.parents), parents)
    for ligne in diagnostic(chunks, parents, cfg):
        print(ligne)
    return 0 if chunks else 1


if __name__ == "__main__":
    raise SystemExit(main())
