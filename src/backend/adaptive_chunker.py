"""
Adaptive chunking strategies for different content types.
Handles tables, lists, financial statements, and narrative text with appropriate strategies.
"""
import re
from typing import List, Dict, Tuple
import tiktoken

from src.backend.content_detector import ContentSection
from src.backend.chunking_config import (
    get_chunking_config,
    get_table_config,
    get_list_config,
    get_narrative_config
)

class AdaptiveChunker:
    """Adaptive chunking with content-aware strategies."""
    
    def __init__(self):
        self.config = get_chunking_config()
        self.table_config = get_table_config()
        self.list_config = get_list_config()
        self.narrative_config = get_narrative_config()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Sentence boundary pattern (simple regex)
        self.sentence_pattern = re.compile(r'[.!?]+[\s\n]+')
    
    def chunk_section(self, section: ContentSection) -> List[Dict]:
        """
        Chunk a content section using appropriate strategy.
        
        Args:
            section: ContentSection to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if section.type == 'table':
            return self._chunk_table(section)
        elif section.type == 'list':
            return self._chunk_list(section)
        elif section.type == 'financial_statement':
            return self._chunk_financial_statement(section)
        else:  # narrative
            return self._chunk_narrative(section)
    
    def _chunk_table(self, section: ContentSection) -> List[Dict]:
        """
        Chunk table by row groups, preserving headers.
        
        Strategy:
        - Split into row groups of configurable size
        - Each chunk includes the table header
        - Metadata tracks row range and table ID
        """
        chunks = []
        lines = section.text.split('\n')
        
        # Extract header (usually first row)
        header = section.metadata.get('header', '')
        header_line = lines[0] if lines else ''
        
        # Get row chunk size from config
        rows_per_chunk = self.table_config['row_chunk_size']
        preserve_header = self.table_config['preserve_header']
        
        # Start from row 1 (skip header)
        row_start = 1
        table_id = f"table_{section.start_pos}"
        
        while row_start < len(lines):
            row_end = min(row_start + rows_per_chunk, len(lines))
            
            # Build chunk with optional header
            chunk_lines = []
            if preserve_header and header_line:
                chunk_lines.append(header_line)
            chunk_lines.extend(lines[row_start:row_end])
            
            chunk_text = '\n'.join(chunk_lines)
            
            # Check token count
            token_count = len(self.tokenizer.encode(chunk_text))
            
            chunks.append({
                'text': chunk_text,
                'chunk_type': 'table',
                'table_id': table_id,
                'row_range': f"{row_start}-{row_end-1} of {len(lines)-1}",
                'token_count': token_count,
                'metadata': section.metadata
            })
            
            row_start = row_end
        
        return chunks
    
    def _chunk_list(self, section: ContentSection) -> List[Dict]:
        """
        Chunk lists keeping items together when possible.
        
        Strategy:
        - Keep list items together under token limit
        - Split at natural boundaries if too large
        - Preserve parent context
        """
        chunks = []
        lines = section.text.split('\n')
        
        # Find context (text before first list item)
        context = ""
        first_item_idx = 0
        for i, line in enumerate(lines):
            if re.match(r'^\s*[-•●○\d+\.)(\(]', line):
                if i > 0:
                    context = '\n'.join(lines[:i]).strip()
                first_item_idx = i
                break
        
        # Chunk list items
        chunk_size = self.config['chunk_size']
        current_chunk = []
        current_tokens = 0
        
        # Add context tokens if present
        if context:
            current_chunk.append(context)
            current_tokens = len(self.tokenizer.encode(context))
        
        for line in lines[first_item_idx:]:
            line_tokens = len(self.tokenizer.encode(line))
            
            if current_tokens + line_tokens > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_type': 'list',
                    'list_type': section.metadata.get('list_type', 'unknown'),
                    'token_count': current_tokens,
                    'metadata': section.metadata
                })
                
                # Start new chunk with context
                current_chunk = [context] if context else []
                current_tokens = len(self.tokenizer.encode(context)) if context else 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_type': 'list',
                'list_type': section.metadata.get('list_type', 'unknown'),
                'token_count': current_tokens,
                'metadata': section.metadata
            })
        
        return chunks
    
    def _chunk_financial_statement(self, section: ContentSection) -> List[Dict]:
        """
        Chunk financial statements as semantic units.
        
        Strategy:
        - Try to keep entire statement together if under limit
        - Split by logical sections if too large
        - Preserve statement type metadata
        """
        text = section.text
        tokens = self.tokenizer.encode(text)
        chunk_size = self.config['chunk_size']
        
        # If under limit, keep as single chunk
        if len(tokens) <= chunk_size:
            return [{
                'text': text,
                'chunk_type': 'financial_statement',
                'statement_type': section.metadata.get('statement_type', 'unknown'),
                'token_count': len(tokens),
                'metadata': section.metadata
            }]
        
        # Otherwise, split by paragraphs or table sections
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))
            
            if current_tokens + para_tokens > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_type': 'financial_statement',
                    'statement_type': section.metadata.get('statement_type', 'unknown'),
                    'token_count': current_tokens,
                    'metadata': section.metadata
                })
                
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_type': 'financial_statement',
                'statement_type': section.metadata.get('statement_type', 'unknown'),
                'token_count': current_tokens,
                'metadata': section.metadata
            })
        
        return chunks
    
    def _chunk_narrative(self, section: ContentSection) -> List[Dict]:
        """
        Chunk narrative text with sentence-aware boundaries.
        
        Strategy:
        - Fixed-size windows with overlap
        - Snap to sentence boundaries if enabled
        - Respect paragraph breaks
        """
        text = section.text
        tokens = self.tokenizer.encode(text)
        
        chunk_size = self.narrative_config['chunk_size']
        overlap = self.narrative_config['overlap']
        use_sentences = self.narrative_config['sentence_boundaries']
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Snap to sentence boundary if enabled and not at end
            if use_sentences and end < len(tokens):
                chunk_text = self._snap_to_sentence(chunk_text)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'chunk_type': 'narrative',
                    'token_count': len(self.tokenizer.encode(chunk_text)),
                    'metadata': section.metadata
                })
            
            start += chunk_size - overlap
        
        return chunks
    
    def _snap_to_sentence(self, text: str) -> str:
        """
        Snap text to the last complete sentence.
        
        Args:
            text: Text to snap
            
        Returns:
            Text ending at sentence boundary
        """
        # Find last sentence boundary
        sentences = list(self.sentence_pattern.finditer(text))
        
        if sentences:
            # Keep up to last sentence boundary
            last_boundary = sentences[-1].end()
            return text[:last_boundary].strip()
        
        # No sentence boundary found, return as-is
        return text

def chunk_document_adaptive(text: str, chunker: AdaptiveChunker, detector) -> List[Dict]:
    """
    Main entry point for adaptive document chunking.
    
    Args:
        text: Document text
        chunker: AdaptiveChunker instance
        detector: ContentDetector instance
        
    Returns:
        List of chunk dictionaries
    """
    # Detect content sections
    sections = detector.detect_content_sections(text)
    
    # Chunk each section
    all_chunks = []
    for section in sections:
        section_chunks = chunker.chunk_section(section)
        all_chunks.extend(section_chunks)
    
    return all_chunks
