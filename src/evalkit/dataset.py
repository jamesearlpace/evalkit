"""Dataset loading and deterministic train/test/validation splitting."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataRow:
    """A single eval item."""

    id: str
    input: str
    expected_output: str
    context: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetError(ValueError):
    """Raised when dataset input cannot be loaded or validated."""


_INPUT_KEYS = ("input", "question", "query", "prompt")
_EXPECTED_KEYS = ("expected_output", "expected", "answer", "ground_truth")
_RESERVED_KEYS = {
    "id",
    "input",
    "question",
    "query",
    "prompt",
    "expected_output",
    "expected",
    "answer",
    "ground_truth",
    "context",
}


def load_dataset(path: str | Path, fmt: str | None = None) -> list[DataRow]:
    """Load a CSV or JSONL dataset.

    Required logical fields are ``input`` and ``expected_output``. Common aliases
    such as ``question`` and ``answer`` are accepted to make starter datasets
    less fussy.
    """

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetError(f"Dataset not found: {dataset_path}")

    resolved_format = (fmt or dataset_path.suffix.lstrip(".")).lower()
    if resolved_format == "json":
        resolved_format = "jsonl"

    if resolved_format == "csv":
        return _load_csv(dataset_path)
    if resolved_format == "jsonl":
        return _load_jsonl(dataset_path)

    raise DatasetError(f"Unsupported dataset format '{resolved_format}'. Use csv or jsonl.")


def split_dataset(
    rows: list[DataRow],
    ratios: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, list[DataRow]]:
    """Split rows deterministically by ratio.

    The default is 80/20 train/test. Counts are allocated by floor plus largest
    remainder, so every row lands in exactly one split.
    """

    if ratios is None:
        ratios = {"train": 0.8, "test": 0.2}
    if not rows:
        return {name: [] for name in ratios}

    cleaned = {name: float(value) for name, value in ratios.items() if float(value) > 0}
    if not cleaned:
        raise DatasetError("At least one split ratio must be greater than zero.")

    names = list(cleaned)
    total_ratio = sum(cleaned.values())
    exact_counts = {name: len(rows) * value / total_ratio for name, value in cleaned.items()}
    counts = {name: int(exact_counts[name]) for name in names}
    remaining = len(rows) - sum(counts.values())

    remainders = sorted(
        names,
        key=lambda name: (exact_counts[name] - counts[name], -names.index(name)),
        reverse=True,
    )
    for name in remainders[:remaining]:
        counts[name] += 1

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    result: dict[str, list[DataRow]] = {}
    cursor = 0
    for name in names:
        count = counts[name]
        result[name] = shuffled[cursor : cursor + count]
        cursor += count
    return result


def _load_csv(path: Path) -> list[DataRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"CSV has no header row: {path}")
        return [_normalize_record(record, index + 1) for index, record in enumerate(reader)]


def _load_jsonl(path: Path) -> list[DataRow]:
    rows: list[DataRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON on line {index} of {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise DatasetError(f"JSONL line {index} must be an object.")
            rows.append(_normalize_record(record, index))
    return rows


def _normalize_record(record: dict[str, Any], index: int) -> DataRow:
    input_value = _first_present(record, _INPUT_KEYS)
    expected_value = _first_present(record, _EXPECTED_KEYS)

    if input_value is None or str(input_value).strip() == "":
        raise DatasetError(f"Row {index} is missing an input/question value.")
    if expected_value is None:
        raise DatasetError(f"Row {index} is missing an expected_output/answer value.")

    row_id = str(record.get("id") or index)
    metadata = {key: value for key, value in record.items() if key not in _RESERVED_KEYS}
    return DataRow(
        id=row_id,
        input=str(input_value),
        expected_output=str(expected_value),
        context=record.get("context"),
        metadata=metadata,
    )


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None
