"""Adaptive SEC chunker: token sliding window + sentence snapping for narrative; structure-aware for tables/lists."""
import re
from typing import List, Dict

import tiktoken

from src.backend.content_detector import FilingSection, ContentBlock, SECFilingParser
from src.backend.chunking_config import get_chunking_config


class SECChunker:
    """Chunks SEC sections: narrative by token/sentence; tables by row groups; lists kept whole."""

    def __init__(self):
        config = get_chunking_config()
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.min_chunk_size = config["min_chunk_size"]
        self.use_sentence_boundaries = config["enable_sentence_boundaries"]
        self.table_row_chunk_size = config["table_row_chunk_size"]
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.sentence_end = re.compile(r"[.!?]+[\s\n]+")

    def chunk_document(self, sections: List[FilingSection]) -> List[Dict]:
        """Chunk all sections; each chunk has section_name, chunk_type, token_count, text."""
        all_chunks = []

        for section in sections:
            blocks = SECFilingParser.split_into_blocks(section.text)
            section_chunks: List[Dict] = []

            for block in blocks:
                if block.block_type == "table":
                    section_chunks.extend(self._chunk_table(block))
                elif block.block_type == "list":
                    section_chunks.extend(self._chunk_list(block))
                else:
                    section_chunks.extend(self._chunk_narrative(block.text))

            total = len(section_chunks)
            for i, chunk in enumerate(section_chunks):
                chunk["section_name"] = section.name
                chunk["section_chunk_index"] = i
                chunk["total_section_chunks"] = total
            all_chunks.extend(section_chunks)

        return all_chunks

    def _chunk_table(self, block: ContentBlock) -> List[Dict]:
        """Split table into row-group chunks; keep header on first chunk, repeat if splitting."""
        rows = block.rows or block.text.split("\n")
        if not rows:
            return []

        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        chunk_texts: List[str] = []

        if not data_rows:
            chunk_texts = [block.text]
        else:
            step = self.table_row_chunk_size
            for start in range(0, len(data_rows), step):
                end = min(start + step, len(data_rows))
                group = [header] + data_rows[start:end]
                chunk_texts.append("\n".join(group))

        chunks = []
        for part in chunk_texts:
            part = part.strip()
            if not part:
                continue
            token_count = len(self.tokenizer.encode(part))
            chunks.append({
                "text": part,
                "chunk_type": "table",
                "token_count": token_count,
            })
        return chunks

    def _chunk_list(self, block: ContentBlock) -> List[Dict]:
        """Keep list as one chunk (or split by token size if over chunk_size)."""
        text = block.text.strip()
        if not text:
            return []
        token_count = len(self.tokenizer.encode(text))
        if token_count <= self.chunk_size:
            return [{"text": text, "chunk_type": "list", "token_count": token_count}]
        return self._chunk_narrative(text, chunk_type="list")

    def _chunk_narrative(self, text: str, chunk_type: str = "narrative") -> List[Dict]:
        """Token-based sliding window with optional sentence-boundary snapping."""
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.chunk_size:
            stripped = text.strip()
            if stripped:
                return [{
                    "text": stripped,
                    "chunk_type": chunk_type,
                    "token_count": len(tokens),
                }]
            return []

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.tokenizer.decode(tokens[start:end])

            if self.use_sentence_boundaries and end < len(tokens):
                chunk_text = self._snap_to_sentence(chunk_text)

            chunk_text = chunk_text.strip()
            # Use slice length as token count (upper bound; snapping only shortens the text).
            # Avoids an extra encode() call per chunk.
            token_count = end - start

            if chunk_text and token_count >= self.min_chunk_size:
                chunks.append({
                    "text": chunk_text,
                    "chunk_type": chunk_type,
                    "token_count": token_count,
                })

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _snap_to_sentence(self, text: str) -> str:
        """Truncate at last complete sentence boundary."""
        matches = list(self.sentence_end.finditer(text))
        return text if not matches else text[:matches[-1].end()].strip()
