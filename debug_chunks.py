from src.backend.content_detector import SECFilingParser
from src.backend.adaptive_chunker import SECChunker
from pathlib import Path

text = Path('edgar_corpus/AAPL_10K_2022Q3_2022-10-28_full.txt').read_text()
parser = SECFilingParser()
chunker = SECChunker()

sections = parser.parse(text)
print(f'Sections: {len(sections)}')
for s in sections:
    print(f'  {s.name}: {len(s.text)} chars')

chunks = chunker.chunk_document(sections)
print(f'\nChunks: {len(chunks)}')
for c in chunks:
    print(f'  [{c["section_name"]}] chunk {c["section_chunk_index"]+1}/{c["total_section_chunks"]}: {c["token_count"]} tokens')
