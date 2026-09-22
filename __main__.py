"""CLI : python -m docxrag fichier.docx -o out.json [--chunks chunks.jsonl] [--describe anthropic:MODELE]"""
from __future__ import annotations

import argparse
import json
import sys

from . import ChunkerConfig, ParserConfig, chunk_document, parse_docx, to_docling_dict, to_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="docxrag")
    ap.add_argument("source")
    ap.add_argument("-o", "--output", help="JSON DoclingDocument (défaut : stdout)")
    ap.add_argument("--markdown")
    ap.add_argument("--chunks", help="JSONL de chunks")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--include-furniture", action="store_true")
    ap.add_argument("--images", choices=["reference", "embed", "export"], default="reference")
    ap.add_argument("--image-dir")
    ap.add_argument("--no-streaming", action="store_true")
    ap.add_argument("--describe", help="fournisseur:modèle — active l'analyse multimodale")
    ap.add_argument("--vision-cache")
    a = ap.parse_args(argv)
    cfg = ParserConfig(include_furniture=a.include_furniture, image_mode=a.images,
                       image_export_dir=a.image_dir, streaming=not a.no_streaming)
    res = parse_docx(a.source, cfg)
    if a.describe:
        from .vision import DescriptionCache, VisionConfig, enrich_pictures, get_describer
        stats = enrich_pictures(res.model, a.source, get_describer(a.describe), VisionConfig(enabled=True),
                                DescriptionCache(a.vision_cache))
        print(f"vision: {stats}", file=sys.stderr)
    out = json.dumps(to_docling_dict(res.model), ensure_ascii=False, indent=1)
    if a.output:
        open(a.output, "w", encoding="utf-8").write(out)
    elif not a.chunks and not a.markdown:
        print(out)
    if a.markdown:
        open(a.markdown, "w", encoding="utf-8").write(to_markdown(res.model))
    if a.chunks:
        with open(a.chunks, "w", encoding="utf-8") as fh:
            for c in chunk_document(res.model, ChunkerConfig(max_tokens=a.max_tokens)):
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"timings(ms): {res.timings_ms}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
