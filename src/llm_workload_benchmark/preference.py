from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_workload_benchmark.dataset import DatasetError, load_suite
from llm_workload_benchmark.evaluation import EvaluationResult

CandidateLabel = Literal["a", "b", "c"]
HumanSelection = tuple[CandidateLabel, ...] | Literal["none"]
_LABELS = ("a", "b", "c")


class PreferenceError(ValueError):
    """Raised when a saved experiment cannot produce a human ballot."""


@dataclass(frozen=True)
class BlindCandidate:
    label: str
    response: str


@dataclass(frozen=True)
class BlindComparison:
    """One anonymous multi-model comparison shown to the human voter."""

    number: int
    total: int
    benchmark: str
    item_id: str
    repetition: int
    prompt: str
    candidates: tuple[BlindCandidate, ...]


@dataclass(frozen=True)
class ArenaBallotItem:
    comparison: BlindComparison
    candidate_model_ids: dict[str, str]


@dataclass(frozen=True)
class PreferenceBallot:
    experiment_id: str | None
    experiment_directory: Path
    model_ids: tuple[str, ...]
    seed: int
    items: tuple[ArenaBallotItem, ...]


PreferenceChooser = Callable[[BlindComparison], HumanSelection]


def completed_model_ids(experiment_directory: Path) -> tuple[str, ...]:
    """Return completed model IDs in experiment order."""

    index = _read_json_object(experiment_directory.resolve() / "experiment.json")
    entries = index.get("models")
    if not isinstance(entries, list):
        raise PreferenceError("experiment index has no model list")
    return tuple(
        entry["model_id"]
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("status") == "completed"
        and isinstance(entry.get("model_id"), str)
    )


def prepare_preference_ballot(
    experiment_directory: Path,
    *,
    model_ids: tuple[str, ...] | None = None,
    seed: int = 42,
) -> PreferenceBallot:
    """Load and anonymously order answers from two or three completed models."""

    root = experiment_directory.resolve()
    index = _read_json_object(root / "experiment.json")
    selected = _select_models(index, model_ids)
    selected_ids = tuple(entry["model_id"] for entry in selected)
    prompts = _load_prompts(root, index)
    records_by_model = {
        entry["model_id"]: _load_model_records(root, entry) for entry in selected
    }
    key_sets = [set(records) for records in records_by_model.values()]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise PreferenceError("the model runs do not contain the same completed items")
    if not key_sets[0]:
        raise PreferenceError("the selected model runs contain no comparable answers")

    keys = sorted(key_sets[0])
    items: list[ArenaBallotItem] = []
    for number, key in enumerate(keys, start=1):
        ordered_ids = _anonymous_order(seed, key, selected_ids)
        candidate_model_ids = dict(zip(_LABELS, ordered_ids, strict=False))
        candidates = tuple(
            BlindCandidate(
                label=label,
                response=records_by_model[model_id][key]["preference_response"],
            )
            for label, model_id in candidate_model_ids.items()
        )
        items.append(
            ArenaBallotItem(
                comparison=BlindComparison(
                    number=number,
                    total=len(keys),
                    benchmark=key[0],
                    item_id=key[1],
                    repetition=key[2],
                    prompt=prompts.get(
                        key[1], "Prompt unavailable in current dataset."
                    ),
                    candidates=candidates,
                ),
                candidate_model_ids=candidate_model_ids,
            )
        )
    experiment_id = index.get("experiment_id")
    return PreferenceBallot(
        experiment_id=experiment_id if isinstance(experiment_id, str) else None,
        experiment_directory=root,
        model_ids=selected_ids,
        seed=seed,
        items=tuple(items),
    )


def default_preference_path(ballot: PreferenceBallot) -> Path:
    """Return the experiment-local multi-model human vote artifact."""

    return ballot.experiment_directory / "human_preferences--arena.jsonl"


def append_human_preference(
    path: Path,
    *,
    ballot: PreferenceBallot,
    item: ArenaBallotItem,
    choice: HumanSelection,
) -> None:
    """Validate and durably append one multi-model human vote."""

    if choice == "none":
        selected_labels: tuple[CandidateLabel, ...] = ()
    else:
        selected_labels = choice
        if (
            not selected_labels
            or len(selected_labels) != len(set(selected_labels))
            or any(label not in item.candidate_model_ids for label in selected_labels)
        ):
            raise PreferenceError(f"invalid human preference selection: {choice!r}")
    selected_models = [
        item.candidate_model_ids[label] for label in selected_labels
    ]
    evaluation = EvaluationResult(
        type="human",
        evaluator="blind_multiway_preference",
        version=2,
        passed=bool(selected_models),
        score=1.0 if selected_models else 0.0,
        details={
            "selected_labels": list(selected_labels),
            "selected_model_ids": selected_models,
            "none_of_above": choice == "none",
            "candidate_model_ids": item.candidate_model_ids,
            "order_seed": ballot.seed,
        },
    )
    responses = {
        candidate.label: candidate.response for candidate in item.comparison.candidates
    }
    record = {
        "schema_version": 3,
        "experiment_id": ballot.experiment_id,
        "benchmark": item.comparison.benchmark,
        "item_id": item.comparison.item_id,
        "repetition": item.comparison.repetition,
        "candidates": [
            {
                "label": label,
                "model_id": model_id,
                "response_sha256": _response_hash(responses[label]),
            }
            for label, model_id in item.candidate_model_ids.items()
        ],
        "evaluation": evaluation.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()


def completed_preference_count(path: Path, ballot: PreferenceBallot) -> int:
    """Validate saved arena votes and return the next ballot position."""

    if not path.exists():
        return 0
    records = _read_jsonl(path)
    if len(records) > len(ballot.items):
        raise PreferenceError("preference file contains more votes than this ballot")
    for index, record in enumerate(records):
        item = ballot.items[index]
        comparison = item.comparison
        candidates = record.get("candidates")
        evaluation = record.get("evaluation")
        details = evaluation.get("details") if isinstance(evaluation, dict) else None
        expected_candidates = [
            {
                "label": candidate.label,
                "model_id": item.candidate_model_ids[candidate.label],
                "response_sha256": _response_hash(candidate.response),
            }
            for candidate in comparison.candidates
        ]
        if (
            record.get("schema_version") != 3
            or record.get("experiment_id") != ballot.experiment_id
            or record.get("benchmark") != comparison.benchmark
            or record.get("item_id") != comparison.item_id
            or record.get("repetition") != comparison.repetition
            or candidates != expected_candidates
            or not isinstance(details, dict)
            or details.get("candidate_model_ids") != item.candidate_model_ids
            or details.get("order_seed") != ballot.seed
        ):
            raise PreferenceError(
                f"saved vote {index + 1} does not match the current arena ballot"
            )
    return len(records)


def collect_human_preferences(
    experiment_directory: Path,
    *,
    chooser: PreferenceChooser,
    model_ids: tuple[str, ...] | None = None,
    seed: int = 42,
    output_path: Path | None = None,
) -> Path:
    """Collect blind multi-model votes for a saved matrix experiment."""

    ballot = prepare_preference_ballot(
        experiment_directory,
        model_ids=model_ids,
        seed=seed,
    )
    destination = (output_path or default_preference_path(ballot)).resolve()
    if destination.exists():
        raise PreferenceError(f"preference output already exists: {destination}")
    try:
        for item in ballot.items:
            append_human_preference(
                destination,
                ballot=ballot,
                item=item,
                choice=chooser(item.comparison),
            )
    except BaseException:
        if destination.exists() and destination.stat().st_size == 0:
            destination.unlink()
        raise
    return destination


def summarize_human_preferences(path: Path) -> dict[str, Any]:
    """Return multi-model choice totals from a human vote artifact."""

    records = _read_jsonl(path)
    wins: dict[str, int] = {}
    none_of_above = 0
    for record in records:
        details = record["evaluation"]["details"]
        for model_id in details["candidate_model_ids"].values():
            wins.setdefault(model_id, 0)
        selected = details["selected_model_ids"]
        if details["none_of_above"]:
            none_of_above += 1
        else:
            for model_id in selected:
                wins[model_id] += 1
    return {
        "votes": len(records),
        "wins": wins,
        "none_of_above": none_of_above,
    }


def _load_prompts(root: Path, index: dict[str, Any]) -> dict[str, str]:
    dataset_reference = index.get("dataset")
    if not isinstance(dataset_reference, str):
        raise PreferenceError("experiment index is missing its dataset path")
    dataset_path = Path(dataset_reference)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    try:
        suite = load_suite(dataset_path.resolve())
    except DatasetError as error:
        raise PreferenceError(f"could not load experiment dataset: {error}") from error
    return {
        item.id: item.prompt
        for items in suite.items.values()
        for item in items
    }


def _select_models(
    index: dict[str, Any], model_ids: tuple[str, ...] | None
) -> list[dict[str, Any]]:
    entries = index.get("models")
    if not isinstance(entries, list):
        raise PreferenceError("experiment index has no model list")
    completed = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("status") == "completed"
        and isinstance(entry.get("model_id"), str)
    ]
    if model_ids is not None:
        if len(model_ids) != len(set(model_ids)):
            raise PreferenceError("human preference requires different models")
        by_id = {entry["model_id"]: entry for entry in completed}
        missing = [model_id for model_id in model_ids if model_id not in by_id]
        if missing:
            raise PreferenceError(f"completed model runs not found: {missing}")
        completed = [by_id[model_id] for model_id in model_ids]
    if not 2 <= len(completed) <= 3:
        raise PreferenceError("human preference requires two or three completed models")
    return completed


def _load_model_records(
    experiment_root: Path, entry: dict[str, Any]
) -> dict[tuple[str, str, int], dict[str, Any]]:
    run_reference = entry.get("run_directory")
    if not isinstance(run_reference, str):
        raise PreferenceError(f"model {entry['model_id']!r} has no run directory")
    results_path = (experiment_root / run_reference / "results.jsonl").resolve()
    if not results_path.is_relative_to(experiment_root):
        raise PreferenceError(f"model results path escapes experiment: {run_reference}")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in _read_jsonl(results_path):
        response = record.get("raw_response")
        if not isinstance(response, str):
            response = record.get("evaluated_response")
        if record.get("status") != "completed" or not isinstance(response, str):
            continue
        key = (record.get("benchmark"), record.get("item_id"), record.get("repetition"))
        if (
            not isinstance(key[0], str)
            or not isinstance(key[1], str)
            or not isinstance(key[2], int)
        ):
            raise PreferenceError(f"invalid result identity in {results_path}")
        if key in records:
            raise PreferenceError(f"duplicate result identity in {results_path}: {key}")
        records[key] = {**record, "preference_response": response}
    return records


def _anonymous_order(
    seed: int,
    key: tuple[str, str, int],
    model_ids: tuple[str, ...],
) -> tuple[str, ...]:
    def order_key(model_id: str) -> bytes:
        material = (
            f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}\0{model_id}"
        ).encode()
        return hashlib.sha256(material).digest()

    return tuple(sorted(model_ids, key=order_key))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PreferenceError(
            f"required preference artifact does not exist: {path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise PreferenceError(
            f"could not read preference artifact {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise PreferenceError(f"preference artifact must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError) as error:
        raise PreferenceError(f"could not read preference artifact {path}: {error}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PreferenceError(
                f"invalid JSON in {path} on line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise PreferenceError(f"JSONL record must be an object in {path}")
        records.append(value)
    return records


def _response_hash(response: str) -> str:
    return hashlib.sha256(response.encode("utf-8")).hexdigest()
