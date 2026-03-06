"""SEC filing chunker. Single-strategy token chunking with sentence-aware boundaries."""
import re
from typing import List, Dict
import tiktoken

from src.backend.content_detector import FilingSection
from src.backend.chunking_config import get_chunking_config


class SECChunker:
    """Chunks SEC filing sections into ~512-token pieces with sentence snapping."""

    def __init__(self):
        config = get_chunking_config()
        self.chunk_size = config['chunk_size']
        self.chunk_overlap = config['chunk_overlap']
        self.min_chunk_size = config['min_chunk_size']
        self.use_sentence_boundaries = config['enable_sentence_boundaries']
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.sentence_end = re.compile(r'[.!?]+[\s\n]+')

    def chunk_document(self, sections: List[FilingSection]) -> List[Dict]:
        """Chunk all sections, tagging each chunk with section metadata."""
        all_chunks = []

        for section in sections:
            section_chunks = self._chunk_section(section.text)
            total = len(section_chunks)

            for i, chunk in enumerate(section_chunks):
                chunk['section_name'] = section.name
                chunk['section_chunk_index'] = i
                chunk['total_section_chunks'] = total

            all_chunks.extend(section_chunks)

        return all_chunks

    def _chunk_section(self, text: str) -> List[Dict]:
        """Token-based sliding window with sentence-boundary snapping."""
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.chunk_size:
            stripped = text.strip()
            if stripped:
                return [{'text': stripped, 'chunk_type': 'narrative',
                         'token_count': len(tokens)}]
            return []

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.tokenizer.decode(tokens[start:end])

            if self.use_sentence_boundaries and end < len(tokens):
                chunk_text = self._snap_to_sentence(chunk_text)

            chunk_text = chunk_text.strip()
            token_count = len(self.tokenizer.encode(chunk_text))

            if chunk_text and token_count >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'chunk_type': 'narrative',
                    'token_count': token_count,
                })

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _snap_to_sentence(self, text: str) -> str:
        """Truncate text at the last complete sentence boundary."""
        matches = list(self.sentence_end.finditer(text))
        if matches:
            return text[:matches[-1].end()].strip()
        return text
