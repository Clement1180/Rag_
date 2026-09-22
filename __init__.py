"""docxrag — parser DOCX orienté RAG (lxml, sortie compatible DoclingDocument)."""
from .chunking import Chunk, ChunkerConfig, HierarchicalChunker, chunk_document
from .config import HeadingConfig, ParserConfig
from .model import DocModel
from .parser import DocxParser, ParseResult, parse_docx
from .serialize import to_docling_dict, to_markdown

__all__ = ["DocxParser", "ParseResult", "parse_docx", "ParserConfig", "HeadingConfig", "DocModel",
           "to_docling_dict", "to_markdown", "Chunk", "ChunkerConfig", "HierarchicalChunker",
           "chunk_document"]
__version__ = "0.1.0"
