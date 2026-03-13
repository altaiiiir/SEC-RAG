"""SEC filing parser. Strips XBRL data, splits by Item headers, detects tables/lists."""
import re
from typing import List, Literal
from dataclasses import dataclass


@dataclass
class ContentBlock:
    """A contiguous block of content: table, list, or narrative."""
    block_type: Literal["table", "list", "narrative"]
    text: str
    rows: List[str] | None = None  # for tables: line-by-line for row-group splitting


@dataclass
class FilingSection:
    """A named section of an SEC filing (e.g., 'Item 1A. Risk Factors')."""
    name: str
    text: str
    start_pos: int
    end_pos: int


class SECFilingParser:
    """Parses SEC EDGAR filings: strips metadata/XBRL, splits by Item headers."""

    SECTION_PATTERN = re.compile(
        r'^(Item\s+\d+[A-Z]?\..*|ITEM\s+\d+[A-Z]?\..*)', re.MULTILINE
    )

    # hardcoded markers for the end of the XBRL data blob specicific to this dataset
    XBRL_END_MARKERS = [
        'UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION',
        'UNITED STATESSECURITIES AND EXCHANGE COMMISSION',
    ]

    def parse(self, raw_text: str) -> List[FilingSection]:
        """Full pipeline: strip header/XBRL, split into Item sections."""
        text = self._strip_metadata_header(raw_text)
        text = self._strip_xbrl(text)
        return self._split_into_sections(text)

    def _strip_metadata_header(self, text: str) -> str:
        """Remove the metadata header above the ====== separator."""
        sep_pos = text.find('=' * 10)
        if sep_pos != -1:
            newline_after = text.find('\n', sep_pos)
            if newline_after != -1:
                return text[newline_after + 1:]
        return text

    def _strip_xbrl(self, text: str) -> str:
        """Remove the XBRL data blob that precedes the readable filing content."""
        for marker in self.XBRL_END_MARKERS:
            pos = text.find(marker)
            if pos != -1:
                return text[pos:]
        return text

    def _split_into_sections(self, text: str) -> List[FilingSection]:
        """Split filing text at 'Item N.' headers."""
        matches = list(self.SECTION_PATTERN.finditer(text))

        if not matches:
            stripped = text.strip()
            if stripped:
                return [FilingSection("Full Document", stripped, 0, len(text))]
            return []

        sections = []

        # Preamble (cover page, table of contents) before first Item
        if matches[0].start() > 100:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.append(FilingSection(
                    "Preamble", preamble, 0, matches[0].start()
                ))

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append(FilingSection(
                    match.group(1).strip(), section_text, start, end
                ))

        return sections

    # --- Block detection (tables, lists) within section text ---
    _COLUMN_SPLIT = re.compile(r"[ \t]{2,}")
    _LIST_LINE = re.compile(r"^\s*(?:[\-\*•]\s+|\d+[.)]\s+)", re.MULTILINE)

    @classmethod
    def split_into_blocks(cls, text: str) -> List[ContentBlock]:
        """Split section text into table, list, and narrative blocks."""
        blocks: List[ContentBlock] = []
        # Split by double newline to get paragraphs
        raw_paras = re.split(r"\n\s*\n", text)
        for para in raw_paras:
            para = para.strip()
            if not para:
                continue
            lines = para.split("\n")
            if cls._looks_like_table(lines):
                blocks.append(ContentBlock("table", para, rows=lines))
            elif cls._looks_like_list(lines):
                blocks.append(ContentBlock("list", para, rows=lines))
            else:
                blocks.append(ContentBlock("narrative", para, rows=None))
        return blocks

    @classmethod
    def _looks_like_table(cls, lines: List[str]) -> bool:
        """True if most lines have multiple columns (2+ spaces/tabs)."""
        if len(lines) < 2:
            return False
        column_count = 0
        for line in lines:
            parts = [p for p in cls._COLUMN_SPLIT.split(line) if p.strip()]
            if len(parts) >= 2:
                column_count += 1
        return column_count >= max(2, len(lines) * 0.5)

    @classmethod
    def _looks_like_list(cls, lines: List[str]) -> bool:
        """True if most non-empty lines look like list items."""
        if not lines:
            return False
        list_lines = sum(1 for line in lines if line.strip() and cls._LIST_LINE.match(line))
        return list_lines >= max(2, len([l for l in lines if l.strip()]) * 0.5)
