from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
import re
from typing import Iterable

from llm_workload_benchmark.dataset import DatasetItem, load_suite


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".text",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
WORD_PATTERN = re.compile(r"[\w]+(?:['’.-][\w]+)*", re.UNICODE)


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_PATTERN.finditer(text)]


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    words = _words(text)
    return {
        tuple(words[index : index + size])
        for index in range(len(words) - size + 1)
    }


def _source_text(item: DatasetItem) -> str:
    if item.benchmark == "email_to_action":
        return item.prompt.split("--- RAW EMAIL ---\n", 1)[1].split(
            "\n--- END EMAIL ---", 1
        )[0]
    if item.benchmark == "messy_text_to_schema":
        return item.prompt.rsplit("Text: ", 1)[1]
    if item.benchmark == "grounded_compression":
        return item.prompt.split("\n\nSource: ", 1)[1]
    if item.benchmark == "tool_use":
        return item.prompt
    raise ValueError(f"no public-source extractor for {item.benchmark}")


def _corpus_key(item: DatasetItem) -> str | None:
    source = item.provenance.source
    if source is None or "fully rewritten" not in source.dataset.casefold():
        return None
    dataset = source.dataset.casefold()
    aliases = {
        "enron corpus": "enron",
        "sroie": "sroie",
        "funsd": "funsd",
        "cord": "cord",
        "home-assistant-requests-v2": "home_assistant_v2",
        "home-assistant-requests": "home_assistant_v1",
        "dialogsum": "dialogsum",
        "qmsum": "qmsum",
        "meetingbank": "meetingbank",
    }
    for marker, key in aliases.items():
        if marker in dataset:
            return key
    return None


def _text_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.casefold() in TEXT_SUFFIXES:
            yield candidate


def _scan_file(
    path: Path,
    targets: set[tuple[str, ...]],
    size: int,
) -> set[tuple[str, ...]]:
    hits: set[tuple[str, ...]] = set()
    window: deque[str] = deque(maxlen=size)
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            for word in _words(line):
                window.append(word)
                if len(window) == size:
                    shingle = tuple(window)
                    if shingle in targets:
                        hits.add(shingle)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject benchmark sources sharing long shingles with style corpora"
    )
    parser.add_argument("--suite", type=Path, action="append", required=True)
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Optional benchmark family filter; repeat to select several families",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Corpus key and local file/directory; repeat for every provenance corpus",
    )
    parser.add_argument("--shingle-size", type=int, default=8)
    args = parser.parse_args()
    if args.shingle_size < 8:
        parser.error("shingle size must be at least 8")

    corpora: dict[str, Path] = {}
    for declaration in args.corpus:
        name, separator, raw_path = declaration.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --corpus value: {declaration!r}")
        path = Path(raw_path)
        if not path.exists():
            parser.error(f"corpus path does not exist: {path}")
        corpora[name] = path

    items_by_corpus: dict[str, list[DatasetItem]] = defaultdict(list)
    seen_ids: set[str] = set()
    selected_families = set(args.family)
    for suite_path in args.suite:
        for family, items in load_suite(suite_path).items.items():
            if selected_families and family not in selected_families:
                continue
            for item in items:
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)
                key = _corpus_key(item)
                if key is not None:
                    items_by_corpus[key].append(item)

    missing = sorted(set(items_by_corpus) - set(corpora))
    if missing:
        parser.error("missing local corpus paths for: " + ", ".join(missing))

    failures: list[tuple[str, str, str, str]] = []
    checked_items = 0
    checked_files = 0
    for corpus_key, items in sorted(items_by_corpus.items()):
        item_shingles = {
            item.id: _shingles(_source_text(item), args.shingle_size) for item in items
        }
        reverse: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for item_id, shingles in item_shingles.items():
            checked_items += 1
            for shingle in shingles:
                reverse[shingle].add(item_id)
        targets = set(reverse)
        for corpus_file in _text_files(corpora[corpus_key]):
            checked_files += 1
            for shingle in _scan_file(corpus_file, targets, args.shingle_size):
                phrase = " ".join(shingle)
                for item_id in reverse[shingle]:
                    failures.append(
                        (corpus_key, item_id, str(corpus_file), phrase)
                    )

    print(
        f"checked {checked_items} rewritten items against {checked_files} corpus files "
        f"using {args.shingle_size}-word shingles"
    )
    if failures:
        for corpus, item_id, path, phrase in failures[:100]:
            print(f"OVERLAP {corpus} {item_id} {path}: {phrase}")
        if len(failures) > 100:
            print(f"... {len(failures) - 100} additional overlaps")
        raise SystemExit(1)
    print("zero overlaps")


if __name__ == "__main__":
    main()
