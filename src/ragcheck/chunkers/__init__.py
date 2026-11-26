"""Chunking strategies. All of them satisfy ``chunk.text == doc.text[chunk.start:chunk.end]``."""

from ragcheck.chunkers.fixed import FixedSizeChunker
from ragcheck.chunkers.markdown import MarkdownHeaderChunker
from ragcheck.chunkers.recursive import DEFAULT_SEPARATORS, RecursiveChunker
from ragcheck.chunkers.sentence import SentenceWindowChunker, split_sentences

__all__ = [
    "DEFAULT_SEPARATORS",
    "FixedSizeChunker",
    "MarkdownHeaderChunker",
    "RecursiveChunker",
    "SentenceWindowChunker",
    "split_sentences",
]
