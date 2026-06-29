from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_workload_benchmark.dataset import BenchmarkDefinition, DatasetError, load_suite


class CatalogError(ValueError):
    """Raised when the benchmark catalog and its files disagree."""


class CatalogSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Literal["A", "B", "C", "D", "E", "F"]
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    scoring: str = Field(min_length=1)


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    suite: Literal["A", "B", "C", "D", "E", "F"]
    status: Literal["planned", "started", "complete"]
    kind: Literal["question_set", "evaluation_track", "experiment_group"]
    definition_path: str = Field(min_length=1)
    active: bool = True


class CatalogDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2]
    name: str = Field(min_length=1)
    suites: list[CatalogSuite] = Field(min_length=6, max_length=6)
    benchmarks: list[CatalogEntry] = Field(min_length=1)
    evaluation_files: dict[str, str]
    probe_sets: list[str] = Field(min_length=1)


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
    if suite_ids != list("ABCDEF"):
        raise CatalogError("catalog suites must be ordered A through F")
    entry_ids = [entry.id for entry in catalog.benchmarks]
    if len(entry_ids) != len(set(entry_ids)):
        raise CatalogError("catalog benchmark ids must be unique")

    question_set_ids: set[str] = set()
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
                question_set_ids.add(entry.id)
            planned += entry.status == "planned"

    for relative_path in [*catalog.evaluation_files.values(), *catalog.probe_sets]:
        if not (path.parent / relative_path).is_file():
            raise CatalogError(f"catalog resource does not exist: {relative_path}")

    all_suite_path = path.parent / "suites" / "all.yaml"
    try:
        suite = load_suite(all_suite_path)
    except DatasetError as error:
        raise CatalogError(f"all-benchmarks suite is invalid: {error}") from error
    if set(suite.definitions) != question_set_ids:
        missing = question_set_ids - set(suite.definitions)
        extra = set(suite.definitions) - question_set_ids
        raise CatalogError(f"all suite mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return CatalogValidation(
        benchmark_count=len(catalog.benchmarks),
        question_set_count=len(question_set_ids),
        current_question_count=sum(len(items) for items in suite.items.values()),
        planned_question_set_count=planned,
    )
