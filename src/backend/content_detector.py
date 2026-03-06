"""SEC filing parser. Strips XBRL data and splits filings into sections by Item headers."""
import re
from typing import List
from dataclasses import dataclass


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
