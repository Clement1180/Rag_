"""Tests de chunk.py v2 : de DOCX (via docxrag) et de JSON Docling jusqu'aux chunks."""
import pytest
from docxbuilder import build_docx, drawing, p, png_bytes, run, tbl, tc, tr

import chunk as ck
from config import Config

N = ck.compteur_approche()
PHRASE = "Le prestataire assure la maintenance corrective et évolutive du système d'information."


def docs_de(tmp_path, body, nom="rapport", **kw):
    f = tmp_path / f"{nom}.docx"
    f.write_bytes(build_docx(body, **kw))
    return ck.charger_docx(f)


def construire(docs, **cfg):
    c = Config(**{"chunk_tokens": 64, "max_seq_length": 128, **cfg})
    return ck.construire_docling(docs, c, N, ck._limite_parent(c))


def section(titre, n, marque=""):
    return p(titre, style="Heading1") + "".join(p(f"{marque}Paragraphe {i}. {PHRASE}") for i in range(n))


def test_identifiants_stables_apres_insertion(tmp_path):
    avant = section("Objet", 6) + section("Périmètre", 6) + section("Annexes", 6)
    apres = p("Paragraphe inséré en tête. " + PHRASE) + avant
    c1, _ = construire(docs_de(tmp_path, avant, "a"))
    c2, _ = construire(docs_de(tmp_path, apres, "a"))
    ids = lambda cs, sec: [c["chunk_id"] for c in cs if c["section"] == sec]
    for sec in ("Objet", "Périmètre", "Annexes"):
        assert ids(c1, sec) == ids(c2, sec)          # aucune section existante renommée
    assert len(c2) == len(c1) + 1


def test_texte_indexe_tient_dans_max_seq_length(tmp_path):
    titre = "Conditions particulières d'exécution des prestations de tierce maintenance applicative"
    chunks, _ = construire(docs_de(tmp_path, section(titre, 20)), chunk_tokens=120, max_seq_length=100)
    assert chunks and all(c["n_tokens_indexed"] + 2 <= 100 for c in chunks)
    # titre de plus de 80 caractères : le préfixe en garde la fin, la plus spécifique
    assert all(c["text_indexed"].startswith("RAPPORT — …") and "applicative — page ~1" in c["text_indexed"]
               for c in chunks)


def test_page_docx_estimee_jamais_inventee(tmp_path):
    saut = p(runs='<w:r><w:br w:type="page"/></w:r>')
    chunks, _ = construire(docs_de(tmp_path, section("A", 2) + saut + section("B", 2, "b")))
    assert {c["page"] for c in chunks} == {0, 1}
    assert all(c["page_approx"] for c in chunks)
    assert "page ~2" in [c for c in chunks if c["section"] == "B"][0]["text_indexed"]


def test_page_inconnue_omise():
    doc = {"schema_name": "DoclingDocument", "name": "sans_page",
           "body": {"children": [{"$ref": "#/texts/0"}]},
           "texts": [{"self_ref": "#/texts/0", "label": "text", "text": PHRASE, "prov": []}]}
    chunks, _ = construire([doc])
    assert chunks[0]["page"] is None and "page" not in chunks[0]["text_indexed"]


def test_parent_de_tableau_borne_et_lignes_completes(tmp_path):
    lignes = [tr(tc("Code", bold=True), tc("Libellé", bold=True), header=True)]
    lignes += [tr(tc(f"C{i:03d}"), tc(f"Libellé détaillé numéro {i} de la nomenclature")) for i in range(200)]
    chunks, parents = construire(docs_de(tmp_path, p("Nomenclature", style="Heading1") + tbl(*lignes)),
                                 parent_tokens=300)
    tparents = [x for x in parents if x["parent_id"].startswith("t_")]
    assert len(tparents) > 5 and all(x["n_tokens"] <= 300 for x in tparents)
    assert all("| Code | Libellé |" in x["text"] and "(lignes " in x["text"] for x in tparents)
    rows = [c for c in chunks if c["block_type"] == "table"]
    assert len(rows) == 200 and rows[0]["text"].startswith("Code : C000 ; Libellé :")
    par_id = {x["parent_id"]: x for x in tparents}
    assert all(c["text"].split(" ; ")[0].split(" : ")[1] in par_id[c["parent_id"]]["text"] for c in rows)


def test_ligne_intercalaire_propagee_et_fusion_horizontale_unique(tmp_path):
    t = tbl(tr(tc("Poste", bold=True), tc("Montant", bold=True), header=True),
            tr(tc("Lot 2 : infrastructure", span=2)),
            tr(tc("Serveurs"), tc("12 000")),
            tr(tc("Stockage"), tc("8 000")))
    chunks, _ = construire(docs_de(tmp_path, t))
    rows = [c["text"] for c in chunks if c["block_type"] == "table"]
    assert rows == ["[Lot 2 : infrastructure] Poste : Serveurs ; Montant : 12 000",
                    "[Lot 2 : infrastructure] Poste : Stockage ; Montant : 8 000"]


def test_intercalaire_reporte_dans_la_fenetre_suivante(tmp_path):
    lignes = [tr(tc("Poste", bold=True), tc("Montant", bold=True), header=True),
              tr(tc("Lot 1 : logiciel", span=2))]
    lignes += [tr(tc(f"Licence {i}"), tc(f"{i * 100}")) for i in range(80)]
    _, parents = construire(docs_de(tmp_path, tbl(*lignes)), parent_tokens=150)
    tparents = [x for x in parents if x["parent_id"].startswith("t_")]
    assert len(tparents) > 2 and all("Lot 1 : logiciel" in x["text"] for x in tparents)


def test_markdown_echappe_les_barres():
    table = {"header": ["A", "B"], "rows": [["x | y", "ligne 1\nligne 2"]]}
    assert ck._markdown(table).splitlines()[-1] == "| x \\| y | ligne 1<br>ligne 2 |"


def test_figure_sans_legende_retrouvee_par_texte_alternatif(tmp_path):
    body = (p("Architecture", style="Heading1") + p(PHRASE)
            + p(runs=run(extra=drawing("rIdA", descr="Schéma des flux entre le SI RH et la paie"))))
    chunks, _ = construire(docs_de(tmp_path, body, media={"rIdA": ("a.png", png_bytes())}))
    fig = [c for c in chunks if c["block_type"] == "figure"]
    assert len(fig) == 1 and "flux entre le SI RH" in fig[0]["text"] and fig[0]["est_figure"]


def test_texte_alternatif_automatique_office_ignore(tmp_path):
    alt = "Une image contenant texte, capture d'écran\n\nDescription générée automatiquement"
    body = p(PHRASE) + p(runs=run(extra=drawing("rIdA", descr=alt)))
    chunks, _ = construire(docs_de(tmp_path, body, media={"rIdA": ("a.png", png_bytes())}))
    assert not [c for c in chunks if c["block_type"] == "figure"]


def test_description_docling_recente_et_ancienne():
    base = {"schema_name": "DoclingDocument", "name": "d", "body": {"children": [{"$ref": "#/pictures/0"}]},
            "texts": []}
    recent = {**base, "pictures": [{"self_ref": "#/pictures/0", "label": "picture", "captions": [],
                                    "meta": {"description": {"text": "Courbe de charge"}}}]}
    ancien = {**base, "pictures": [{"self_ref": "#/pictures/0", "label": "picture", "captions": [],
                                    "annotations": [{"kind": "description", "text": "Courbe de charge"}]}]}
    for doc in (recent, ancien):
        chunks, _ = construire([doc])
        assert chunks[0]["text"] == "Description : Courbe de charge"


def test_passages_repetes_indexes_une_fois(tmp_path):
    mention = ("Document confidentiel : diffusion restreinte aux destinataires de la présente note, "
               "toute reproduction même partielle est interdite sans accord écrit préalable.")
    body = section("A", 2) + p(mention) + section("B", 2, "b") + p(mention)
    chunks, _ = construire(docs_de(tmp_path, body))
    assert sum(mention in c["text"] for c in chunks) == 1
    chunks, _ = construire(docs_de(tmp_path, body), dedoublonner=False)
    assert sum(mention in c["text"] for c in chunks) == 2


def test_titre_conserve_sans_prefixe(tmp_path):
    chunks, _ = construire(docs_de(tmp_path, section("Garanties", 1)), prefixe_contexte=False)
    assert chunks[0]["text"].startswith("Garanties")


def test_recursif_respecte_la_cible():
    texte = " ".join(f"Phrase numéro {i} du paragraphe très long." for i in range(300))
    morceaux = ck.decouper_recursif(texte, 50, N)
    assert all(N(m) <= 50 for m in morceaux)
    assert " ".join(morceaux).split() == texte.split()     # rien de perdu, ponctuation comprise
    assert all(m.endswith(".") for m in morceaux)


def test_chevauchement_en_tokens_sans_depasser():
    morceaux = ["un deux trois quatre cinq six sept huit neuf dix", "onze douze treize"]
    out = ck.appliquer_chevauchement(morceaux, 4, N, cible=10)
    assert out[1].endswith("onze douze treize") and N(out[1]) <= 10 and out[1] != morceaux[1]


def test_config_v1_sans_nouveaux_champs(tmp_path):
    class ConfigV1:                                   # l'ancienne Config, sans parent_tokens ni dedoublonner
        strategie, chunk_tokens, chunk_overlap = "structurel", 64, 0
        ingest_version, prefixe_contexte, tableaux_par_ligne = "v1", True, True
        max_seq_length = 128
    cfg = ConfigV1()
    chunks, _ = ck.construire_docling(docs_de(tmp_path, section("A", 3)), cfg, N, ck._limite_parent(cfg))
    assert chunks


def test_entree_pages_jsonl():
    pages = [{"doc_id": "guide", "page": 0, "text": "Intro\n" + PHRASE * 3,
              "headings": [{"text": "Intro"}],
              "tables": [{"header": ["Param", "Valeur"], "rows": [["Tension", "230 V"], ["Fréquence", "50 Hz"]]}]}]
    c = Config(chunk_tokens=64, max_seq_length=128)
    chunks, parents = ck.construire(pages, c, N)
    assert "GUIDE — Intro — page 1" in chunks[0]["text_indexed"]
    assert [x["text"] for x in chunks if x["block_type"] == "table"] == ["Param : Tension ; Valeur : 230 V",
                                                                      "Param : Fréquence ; Valeur : 50 Hz"]


def test_cli_docx(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "note.docx").write_bytes(build_docx(section("Objet", 4)))
    (tmp_path / "w" / "~$note.docx").write_bytes(b"verrou")
    rc = ck.main(["--docx", str(tmp_path / "w"), "--sans-tokenizer",
                  "--out", str(tmp_path / "c.jsonl"), "--parents", str(tmp_path / "p.jsonl")])
    assert rc == 0 and (tmp_path / "c.jsonl").read_text(encoding="utf-8").count("\n") >= 1


# ───────────────────────── compléments v2 ─────────────────────────
def test_petites_sections_soeurs_fusionnees_avec_leurs_titres(tmp_path):
    body = (p("Marché", style="Heading1") + p("Objet", style="Heading2") + p("Court.")
            + p("Durée", style="Heading2") + p("Deux ans.") + p("Prix", style="Heading2") + section("Détail", 0)
            + "".join(p(f"Ligne {i}. {PHRASE}") for i in range(3)))
    chunks, _ = construire(docs_de(tmp_path, body))
    fusion = [c for c in chunks if "Court." in c["text"]]
    assert len(fusion) == 1 and "Deux ans." in fusion[0]["text"]
    assert "Objet" in fusion[0]["text"] and "Durée" in fusion[0]["text"]   # titres gardés dans le texte
    assert fusion[0]["section"] == "Marché"                                  # section commune


def test_pas_de_fusion_vers_un_autre_chapitre(tmp_path):
    body = (p("Chapitre A", style="Heading1") + p("Sous A", style="Heading2") + p("Court.")
            + p("Chapitre B", style="Heading1") + p("Autre texte court."))
    chunks, _ = construire(docs_de(tmp_path, body))
    assert not any("Court." in c["text"] and "Autre texte" in c["text"] for c in chunks)


def test_fusion_desactivable(tmp_path):
    body = section("A", 1) + section("B", 1, "b")
    chunks, _ = construire(docs_de(tmp_path, body), min_section_tokens=0)
    assert {c["section"] for c in chunks} == {"A", "B"}


def test_parents_bornes_meme_avec_un_paragraphe_geant(tmp_path):
    geant = " ".join(f"Clause {i}. {PHRASE}" for i in range(60))
    chunks, parents = construire(docs_de(tmp_path, p("Clauses", style="Heading1") + p(geant)),
                                 parent_tokens=200)
    assert len(parents) > 1 and all(x["n_tokens"] <= 200 for x in parents)
    assert all(x["text"].startswith("Clauses\n\n") for x in parents)        # fil d'Ariane en tête
    assert {c["parent_id"] for c in chunks} == {x["parent_id"] for x in parents}


def test_titre_dans_parent_desactivable(tmp_path):
    _, parents = construire(docs_de(tmp_path, section("Objet", 2)), titre_dans_parent=False)
    assert not parents[0]["text"].startswith("Objet")


def test_aucun_parent_orphelin_apres_dedoublonnage(tmp_path):
    mention = ("Document confidentiel : diffusion restreinte aux destinataires de la présente note, "
               "toute reproduction même partielle est interdite sans accord écrit préalable.")
    body = section("A", 3) + p("B", style="Heading1") + p(mention) + p("C", style="Heading1") + p(mention)
    chunks, parents = construire(docs_de(tmp_path, body), min_section_tokens=0)
    assert {x["parent_id"] for x in parents} == {c["parent_id"] for c in chunks}


def test_titre_jamais_en_fin_de_groupe():
    paras = [{"text": "t", "label": "text", "n": 40}, {"text": "T", "label": "section_header", "n": 5},
             {"text": "u", "label": "text", "n": 40}]
    groupes = ck._grouper(paras, 50)
    assert [[x["text"] for x in g] for g in groupes] == [["t"], ["T", "u"]]


def test_pages_jsonl_parent_borne():
    pages = [{"doc_id": "g", "page": 0, "text": " ".join(f"Article {i}. {PHRASE}" for i in range(80))}]
    c = Config(chunk_tokens=64, max_seq_length=128, parent_tokens=256)
    _, parents = ck.construire(pages, c, N)
    assert len(parents) > 1 and all(x["n_tokens"] <= 256 for x in parents)


def test_recursif_ne_colle_jamais_les_mots():
    texte = " ".join(f"Clause {i}. {PHRASE}" for i in range(60))
    for cible in (190, 40, 7):
        morceaux = ck.decouper_recursif(texte, cible, N)
        assert " ".join(morceaux).split() == texte.split()
        assert all(N(m) <= cible for m in morceaux)
    # mot unique plus long que la cible : seul cas de coupe au caractère
    morceaux = ck.decouper_recursif("x" * 50 + " fin", 1, lambda t: len(t) // 10)
    assert "".join(morceaux).replace(" ", "") == "x" * 50 + "fin"
    assert all(len(m) // 10 <= 1 for m in morceaux)


def test_texte_court_repete_garde_dans_chaque_section(tmp_path):
    body = (section("Pénalités", 0) + p("Sans objet.") + section("Garanties", 0) + p("Sans objet."))
    chunks, _ = construire(docs_de(tmp_path, body), min_section_tokens=0)
    assert sorted(c["section"] for c in chunks if c["text"] == "Sans objet.") == ["Garanties", "Pénalités"]


def test_doublon_long_trace_ses_autres_sections(tmp_path):
    mention = ("Document confidentiel : toute diffusion en dehors des destinataires désignés est interdite "
               "sans l'accord écrit préalable de la direction juridique du groupe.")
    body = section("A", 1) + p(mention) + section("B", 1, "b") + p(mention)
    chunks, _ = construire(docs_de(tmp_path, body), min_section_tokens=0)
    garde = [c for c in chunks if mention in c["text"]]
    assert len(garde) == 1 and garde[0]["autres_sections"] == ["B"]
