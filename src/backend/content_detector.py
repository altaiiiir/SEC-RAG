"""
Content type detection for SEC filings.
Detects tables, lists, financial statements, and narrative text using regex patterns.
"""
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class ContentSection:
    """Represents a section of content with a specific type."""
    type: str  # table, list, financial_statement, narrative
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ContentDetector:
    """Detects different content types in SEC filings."""
    
    # Table detection patterns
    TABLE_PIPE_PATTERN = re.compile(r'^\s*\|[^\n]+\|[^\n]+\|\s*$', re.MULTILINE)
    TABLE_ALIGNED_PATTERN = re.compile(r'^[^\S\n]*\S+[^\S\n]+\S+[^\S\n]+\S+.*$', re.MULTILINE)
    
    # List detection patterns
    LIST_BULLET_PATTERN = re.compile(r'^\s*[-•●○]\s+.+$', re.MULTILINE)
    LIST_NUMBERED_PATTERN = re.compile(r'^\s*\d+[\.)]\s+.+$', re.MULTILINE)
    LIST_LETTERED_PATTERN = re.compile(r'^\s*\([a-z0-9]+\)\s+.+$', re.MULTILINE)
    
    # Financial statement keywords
    FINANCIAL_KEYWORDS = [
        'total assets', 'total liabilities', 'stockholders equity',
        'net income', 'net loss', 'operating income', 'gross profit',
        'total revenue', 'total operating expenses',
        'cash flows from operating', 'cash flows from investing', 'cash flows from financing',
        'balance sheet', 'income statement', 'statement of operations',
        'statement of cash flows', 'comprehensive income'
    ]
    
    # Section header patterns
    SECTION_PATTERN = re.compile(r'^(Item\s+\d+[A-Z]?\..*|ITEM\s+\d+[A-Z]?\..*)', re.MULTILINE)
    
    def __init__(self, min_table_rows: int = 3, min_list_items: int = 2):
        """
        Initialize content detector.
        
        Args:
            min_table_rows: Minimum consecutive rows to detect a table
            min_list_items: Minimum items to detect a list
        """
        self.min_table_rows = min_table_rows
        self.min_list_items = min_list_items
    
    def detect_tables(self, text: str) -> List[Tuple[int, int, Dict]]:
        """
        Detect table regions in text.
        
        Returns:
            List of (start_pos, end_pos, metadata) tuples
        """
        tables = []
        
        # Find pipe-delimited tables
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            if self.TABLE_PIPE_PATTERN.match(lines[i]):
                # Found potential table start
                start_line = i
                consecutive_rows = 1
                
                # Count consecutive table rows
                i += 1
                while i < len(lines) and self.TABLE_PIPE_PATTERN.match(lines[i]):
                    consecutive_rows += 1
                    i += 1
                
                # If enough rows, mark as table
                if consecutive_rows >= self.min_table_rows:
                    start_pos = sum(len(line) + 1 for line in lines[:start_line])
                    end_pos = sum(len(line) + 1 for line in lines[:i])
                    
                    # Extract header if present (usually first row)
                    header = lines[start_line] if start_line < len(lines) else ""
                    
                    tables.append((start_pos, end_pos, {
                        'row_count': consecutive_rows,
                        'header': header.strip(),
                        'format': 'pipe'
                    }))
            else:
                i += 1
        
        return tables
    
    def detect_lists(self, text: str) -> List[Tuple[int, int, Dict]]:
        """
        Detect list regions in text.
        
        Returns:
            List of (start_pos, end_pos, metadata) tuples
        """
        lists = []
        
        # Combine all list patterns
        list_patterns = [
            (self.LIST_BULLET_PATTERN, 'bullet'),
            (self.LIST_NUMBERED_PATTERN, 'numbered'),
            (self.LIST_LETTERED_PATTERN, 'lettered')
        ]
        
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            matched_pattern = None
            list_type = None
            
            # Check which pattern matches
            for pattern, ptype in list_patterns:
                if pattern.match(lines[i]):
                    matched_pattern = pattern
                    list_type = ptype
                    break
            
            if matched_pattern:
                # Found list start
                start_line = i
                consecutive_items = 1
                
                # Count consecutive list items
                i += 1
                while i < len(lines):
                    if matched_pattern.match(lines[i]):
                        consecutive_items += 1
                        i += 1
                    elif lines[i].strip() == '':
                        # Allow blank lines within list
                        i += 1
                    else:
                        break
                
                # If enough items, mark as list
                if consecutive_items >= self.min_list_items:
                    start_pos = sum(len(line) + 1 for line in lines[:start_line])
                    end_pos = sum(len(line) + 1 for line in lines[:i])
                    
                    lists.append((start_pos, end_pos, {
                        'item_count': consecutive_items,
                        'list_type': list_type
                    }))
            else:
                i += 1
        
        return lists
    
    def detect_financial_statements(self, text: str) -> List[Tuple[int, int, Dict]]:
        """
        Detect financial statement regions.
        
        Returns:
            List of (start_pos, end_pos, metadata) tuples
        """
        statements = []
        text_lower = text.lower()
        
        # Look for financial keywords
        for keyword in self.FINANCIAL_KEYWORDS:
            if keyword in text_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(keyword, start)
                    if pos == -1:
                        break
                    
                    # Extract surrounding context (paragraph or table)
                    # Look backwards for paragraph/section start
                    para_start = max(0, text.rfind('\n\n', 0, pos))
                    # Look forwards for paragraph/section end
                    para_end = text.find('\n\n', pos + len(keyword))
                    if para_end == -1:
                        para_end = len(text)
                    
                    # Check if this region has tabular structure
                    region = text[para_start:para_end]
                    has_numbers = bool(re.search(r'\d{1,3}(,\d{3})*(\.\d+)?', region))
                    has_structure = bool(re.search(r'\n.*\d+.*\n.*\d+', region))
                    
                    if has_numbers and has_structure:
                        statements.append((para_start, para_end, {
                            'keyword': keyword,
                            'statement_type': self._classify_statement(keyword)
                        }))
                    
                    start = pos + len(keyword)
        
        # Remove overlaps (keep longest)
        statements = self._remove_overlaps(statements)
        return statements
    
    def _classify_statement(self, keyword: str) -> str:
        """Classify financial statement type from keyword."""
        keyword_lower = keyword.lower()
        if 'balance sheet' in keyword_lower or 'assets' in keyword_lower or 'liabilities' in keyword_lower:
            return 'balance_sheet'
        elif 'income' in keyword_lower or 'operations' in keyword_lower or 'revenue' in keyword_lower:
            return 'income_statement'
        elif 'cash flow' in keyword_lower:
            return 'cash_flow_statement'
        else:
            return 'financial_data'
    
    def _remove_overlaps(self, regions: List[Tuple[int, int, Dict]]) -> List[Tuple[int, int, Dict]]:
        """Remove overlapping regions, keeping the longest ones."""
        if not regions:
            return []
        
        # Sort by start position
        sorted_regions = sorted(regions, key=lambda x: x[0])
        
        result = []
        current = sorted_regions[0]
        
        for region in sorted_regions[1:]:
            # Check for overlap
            if region[0] < current[1]:
                # Overlaps - keep the longer one
                if (region[1] - region[0]) > (current[1] - current[0]):
                    current = region
            else:
                # No overlap
                result.append(current)
                current = region
        
        result.append(current)
        return result
    
    def extract_section_name(self, text: str, position: int) -> str:
        """
        Extract the section name (e.g., 'Item 1A. Risk Factors') for a given position.
        
        Args:
            text: Full document text
            position: Character position
            
        Returns:
            Section name or empty string
        """
        # Find all section headers
        matches = list(self.SECTION_PATTERN.finditer(text))
        
        # Find the section that contains this position
        for i, match in enumerate(matches):
            section_start = match.start()
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            if section_start <= position < section_end:
                return match.group(1).strip()
        
        return ""
    
    def detect_content_sections(self, text: str) -> List[ContentSection]:
        """
        Detect all content sections in text and classify them.
        
        Returns:
            List of ContentSection objects in document order
        """
        # Detect all content types
        tables = self.detect_tables(text)
        lists = self.detect_lists(text)
        financial_statements = self.detect_financial_statements(text)
        
        # Combine all detected regions
        all_regions = []
        
        for start, end, meta in tables:
            all_regions.append((start, end, 'table', meta))
        
        for start, end, meta in lists:
            all_regions.append((start, end, 'list', meta))
        
        for start, end, meta in financial_statements:
            all_regions.append((start, end, 'financial_statement', meta))
        
        # Sort by start position
        all_regions.sort(key=lambda x: x[0])
        
        # Fill gaps with narrative text
        sections = []
        current_pos = 0
        
        for start, end, content_type, meta in all_regions:
            # Add narrative section before this region if there's a gap
            if start > current_pos:
                narrative_text = text[current_pos:start].strip()
                if narrative_text:
                    sections.append(ContentSection(
                        type='narrative',
                        text=narrative_text,
                        start_pos=current_pos,
                        end_pos=start,
                        metadata={}
                    ))
            
            # Add the detected region
            sections.append(ContentSection(
                type=content_type,
                text=text[start:end],
                start_pos=start,
                end_pos=end,
                metadata=meta
            ))
            
            current_pos = end
        
        # Add final narrative section if needed
        if current_pos < len(text):
            narrative_text = text[current_pos:].strip()
            if narrative_text:
                sections.append(ContentSection(
                    type='narrative',
                    text=narrative_text,
                    start_pos=current_pos,
                    end_pos=len(text),
                    metadata={}
                ))
        
        return sections
