import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.chunker import chunk_text, process_file


def test_chunk_text_basic():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_text_small():
    text = "Hello world"
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


def test_chunk_text_empty():
    chunks = chunk_text("", chunk_size=400, chunk_overlap=50)
    assert chunks == []


def test_process_file_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("This is a test document with some content about bakery products.")
    result = process_file(f)
    assert len(result) >= 1
    assert result[0].source_name == "test.txt"
    assert result[0].source_id.startswith("DOC-")
