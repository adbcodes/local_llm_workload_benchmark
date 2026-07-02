from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_workload_benchmark.dataset import BenchmarkDefinition, load_dataset


class CatalogError(ValueError):
    """Raised when the benchmark catalog and its files disagree."""


class CatalogSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Literal["A", "B", "C", "D", "E"]
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    scoring: str = Field(min_length=1)


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    suite: Literal["A", "B", "C", "D", "E"]
    status: Literal["planned", "started", "complete"]
    kind: Literal["question_set"]
    definition_path: str = Field(min_length=1)
    active: bool = True


class CatalogDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2]
    name: str = Field(min_length=1)
    suites: list[CatalogSuite] = Field(min_length=5, max_length=5)
    benchmarks: list[CatalogEntry] = Field(min_length=1)
    evaluation_files: dict[str, str]
    probe_sets: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CatalogValidation:
    benchmark_count: int
    question_set_count: int
    current_question_count: int
    planned_question_set_count: int


def validate_catalog(path: Path) -> CatalogValidation:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        catalog = CatalogDocument.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise CatalogError(f"invalid benchmark catalog {path}: {error}") from error

    suite_ids = [suite.id for suite in catalog.suites]
    if suite_ids != list("ABCDE"):
        raise CatalogError("catalog suites must be ordered A through E")
    entry_ids = [entry.id for entry in catalog.benchmarks]
    if len(entry_ids) != len(set(entry_ids)):
        raise CatalogError("catalog benchmark ids must be unique")

    current_question_count = 0
    planned = 0
    for entry in catalog.benchmarks:
        definition_path = path.parent / entry.definition_path
        try:
            definition_raw = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise CatalogError(f"could not read {definition_path}: {error}") from error
        if not isinstance(definition_raw, dict) or definition_raw.get("id") != entry.id:
            raise CatalogError(f"catalog entry {entry.id!r} does not match its definition")
        if entry.kind == "question_set":
            try:
                definition = BenchmarkDefinition.model_validate(definition_raw)
            except ValidationError as error:
                raise CatalogError(f"invalid benchmark definition {definition_path}: {error}") from error
            if definition.suite != entry.suite or definition.status != entry.status:
                raise CatalogError(f"catalog metadata differs for {entry.id!r}")
            if entry.active:
                items_path = definition_path.parent / definition.items_path
                items = load_dataset(items_path, allow_empty=entry.status == "planned")
                if len(items) != definition.current_question_count:
                    raise CatalogError(
                        f"catalog count differs for {entry.id!r}: "
                        f"definition={definition.current_question_count}, items={len(items)}"
                    )
                current_question_count += len(items)
            planned += entry.status == "planned"

    for relative_path in [*catalog.evaluation_files.values(), *catalog.probe_sets]:
        if not (path.parent / relative_path).is_file():
            raise CatalogError(f"catalog resource does not exist: {relative_path}")

    return CatalogValidation(
        benchmark_count=len(catalog.benchmarks),
        question_set_count=sum(entry.active for entry in catalog.benchmarks),
        current_question_count=current_question_count,
        planned_question_set_count=planned,
    )
