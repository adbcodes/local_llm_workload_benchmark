from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llm_workload_benchmark.dataset import (
    DIFFICULTY_ORDER,
    BenchmarkDefinition,
    DatasetError,
    DatasetItem,
    SuiteManifest,
    _validate_variant_lineage,
    load_dataset,
    load_suite,
    score_answer,
)


@dataclass(frozen=True)
class BuildResult:
    suite_path: Path
    written: tuple[Path, ...]
    unchanged: tuple[Path, ...]


def build_authoring_suite(suite_path: Path, *, check: bool = False) -> BuildResult:
    """Compile readable YAML authoring shards into runtime JSONL files."""

    resolved_suite_path = suite_path.resolve()
    manifest = _load_model(resolved_suite_path, SuiteManifest)
    outputs: list[tuple[Path, str]] = []
    suite_items: dict[str, DatasetItem] = {}

    for relative_definition_path in manifest.benchmark_files:
        definition_path = resolved_suite_path.parent / relative_definition_path
        definition = _load_model(definition_path, BenchmarkDefinition)
        output_path = definition_path.parent / definition.items_path
        if definition.authoring_paths:
            items = _load_authoring_items(definition_path, definition)
        else:
            items = load_dataset(output_path)
        for item in items:
            if item.id in suite_items:
                raise DatasetError(f"duplicate item id across suite: {item.id!r}")
            suite_items[item.id] = item
        if definition.authoring_paths:
            content = "".join(
                json.dumps(_runtime_item(item), separators=(",", ":")) + "\n"
                for item in items
            )
            outputs.append((output_path, content))

    _validate_variant_lineage(suite_items)

    changed = [
        path
        for path, content in outputs
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if check and changed:
        raise DatasetError(
            "generated JSONL is out of date: "
            + ", ".join(str(path) for path in changed)
        )

    written: list[Path] = []
    unchanged: list[Path] = []
    for output_path, content in outputs:
        if output_path in changed:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary_path.write_text(content, encoding="utf-8")
            temporary_path.replace(output_path)
            written.append(output_path)
        else:
            unchanged.append(output_path)

    load_suite(resolved_suite_path)
    return BuildResult(
        suite_path=resolved_suite_path,
        written=tuple(written),
        unchanged=tuple(unchanged),
    )


def _load_authoring_items(
    definition_path: Path,
    definition: BenchmarkDefinition,
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    seen_ids: set[str] = set()

    for relative_authoring_path in definition.authoring_paths:
        authoring_path = definition_path.parent / relative_authoring_path
        raw_document = _load_yaml(authoring_path)
        if not isinstance(raw_document, dict):
            raise DatasetError(f"authoring file must contain a YAML object: {authoring_path}")
        unknown_keys = set(raw_document) - {"schema_version", "benchmark", "items"}
        if unknown_keys:
            raise DatasetError(
                f"unknown authoring fields in {authoring_path}: "
                + ", ".join(sorted(unknown_keys))
            )
        if raw_document.get("schema_version") != 1:
            raise DatasetError(f"authoring file must use schema_version 1: {authoring_path}")
        if raw_document.get("benchmark") != definition.id:
            raise DatasetError(
                f"authoring file {authoring_path} must declare benchmark "
                f"{definition.id!r}"
            )
        raw_items = raw_document.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise DatasetError(
                f"authoring file must contain a non-empty items list: {authoring_path}"
            )

        for item_number, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                raise DatasetError(
                    f"item {item_number} in {authoring_path} must be a YAML object"
                )
            complete_item: dict[str, Any] = {"benchmark": definition.id, **raw_item}
            try:
                item = DatasetItem.model_validate(complete_item)
            except ValidationError as error:
                raise DatasetError(
                    f"invalid authoring item in {authoring_path} at item "
                    f"{item_number}:\n{error}"
                ) from error
            if item.id in seen_ids:
                raise DatasetError(
                    f"duplicate authoring item id {item.id!r} in {definition.id}"
                )
            if item.scoring.method not in definition.scoring_methods:
                raise DatasetError(
                    f"item {item.id!r} uses undeclared scoring method "
                    f"{item.scoring.method!r}"
                )
            _validate_gold(item)
            seen_ids.add(item.id)
            items.append(item)

    items.sort(key=lambda item: DIFFICULTY_ORDER[item.difficulty])
    if len(items) != definition.current_question_count:
        raise DatasetError(
            f"{definition.id} declares {definition.current_question_count} current "
            f"questions but its authoring files contain {len(items)}"
        )
    counts = Counter(item.difficulty for item in items)
    actual_distribution = {
        difficulty: counts[difficulty] for difficulty in DIFFICULTY_ORDER
    }
    if actual_distribution != definition.current_difficulty_distribution:
        raise DatasetError(
            f"{definition.id} authoring difficulty counts do not match "
            "current_difficulty_distribution"
        )
    return items


def _runtime_item(item: DatasetItem) -> dict[str, Any]:
    value = item.model_dump(mode="json")
    if value["variant_of"] is None:
        value.pop("variant_of")
    return value


def _validate_gold(item: DatasetItem) -> None:
    if item.scoring.method in {"llm_judge", "executable_python"}:
        return
    value = item.expected["value"]
    answer = (
        json.dumps(value, separators=(",", ":"))
        if item.response_contract.type == "json"
        else str(value)
    )
    if not score_answer(item, answer).passed:
        raise DatasetError(
            f"expected answer for authoring item {item.id!r} does not satisfy its scorer"
        )


def _load_model(path: Path, model_type: type[Any]) -> Any:
    raw_value = _load_yaml(path)
    try:
        return model_type.model_validate(raw_value)
    except ValidationError as error:
        raise DatasetError(f"invalid dataset metadata in {path}:\n{error}") from error


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"dataset file does not exist: {path}") from error
    except OSError as error:
        raise DatasetError(f"could not read dataset file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise DatasetError(f"dataset file is not valid YAML: {path}: {error}") from error
