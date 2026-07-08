from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_public_corpus_overlap import _scan_file, _shingles  # noqa: E402


def test_overlap_checker_finds_eight_word_shingles_only(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "A real source contains this exact run of eight important words today.",
        encoding="utf-8",
    )
    targets = _shingles(
        "Different opening, then this exact run of eight important words today.", 8
    )
    hits = _scan_file(corpus, targets, 8)

    assert hits == {
        ("this", "exact", "run", "of", "eight", "important", "words", "today")
    }
    assert not _scan_file(corpus, _shingles("only seven unrelated words live here now", 8), 8)
