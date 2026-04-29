"""
Milestone 3 test: exporter.py

Checks:
- export_by_document creates one file per post (respecting max_docs_per_category)
- export_by_category creates one file per category
- slugify handles dots and caps correctly
- export_summary creates valid JSON
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.exporter import export_by_document, export_by_category, export_summary, slugify


SAMPLE_DOCS = [
    {"text": "Astronauts orbit Earth in the space station.", "category": "sci.space", "category_index": 0},
    {"text": "The Hubble telescope captures distant galaxies.", "category": "sci.space", "category_index": 0},
    {"text": "Symptoms of influenza include fever and fatigue.", "category": "sci.med", "category_index": 1},
]


def test_slugify():
    print("Test 1: slugify converts category names...")
    assert slugify("sci.space") == "sci_space"
    assert slugify("comp.sys.ibm.pc.hardware") == "comp_sys_ibm_pc_hardware"
    print("  OK")


def test_export_by_document():
    print("Test 2: export_by_document writes one file per post...")
    with tempfile.TemporaryDirectory() as tmpdir:
        written = export_by_document(SAMPLE_DOCS, tmpdir)
        assert len(written) == 3
        for path in written:
            assert Path(path).exists()
            assert Path(path).read_text(encoding="utf-8").strip() != ""
    print(f"  OK — {len(written)} files written.")


def test_export_by_document_max_per_cat():
    print("Test 3: export_by_document respects max_docs_per_category...")
    with tempfile.TemporaryDirectory() as tmpdir:
        written = export_by_document(SAMPLE_DOCS, tmpdir, max_docs_per_category=1)
        # 1 sci.space + 1 sci.med = 2 files
        assert len(written) == 2
    print(f"  OK -- max_docs_per_category=1 -> {len(written)} files.")


def test_export_by_category():
    print("Test 4: export_by_category writes one file per category...")
    with tempfile.TemporaryDirectory() as tmpdir:
        written = export_by_category(SAMPLE_DOCS, tmpdir)
        assert len(written) == 2  # sci.space and sci.med
        for path in written:
            assert Path(path).exists()
        # sci_space.txt should contain both space posts
        space_file = Path(tmpdir) / "sci_space.txt"
        content = space_file.read_text(encoding="utf-8")
        assert "Hubble" in content
        assert "Astronauts" in content
    print(f"  OK — {len(written)} category files.")


def test_export_summary():
    print("Test 5: export_summary writes valid JSON...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_summary(SAMPLE_DOCS, tmpdir)
        assert Path(path).exists()
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["total_documents"] == 3
        assert "sci.space" in summary["categories"]
        assert summary["categories"]["sci.space"] == 2
    print("  OK")


if __name__ == "__main__":
    print("=" * 50)
    print("Milestone 3: Exporter Tests")
    print("=" * 50)
    test_slugify()
    test_export_by_document()
    test_export_by_document_max_per_cat()
    test_export_by_category()
    test_export_summary()
    print()
    print("All Milestone 3 tests passed.")
