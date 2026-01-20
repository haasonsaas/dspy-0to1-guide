"""Dataset integrity tests for sample data assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets" / "sample_data"


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


@pytest.mark.parametrize("path", sorted(DATASETS_DIR.glob("*.json")))
def test_json_datasets_have_metadata_and_counts(path: Path):
    payload = _load_json(path)

    assert "metadata" in payload, f"missing metadata in {path.name}"
    assert "data" in payload, f"missing data section in {path.name}"

    metadata = payload["metadata"]
    assert (
        "description" in metadata and metadata["description"]
    ), "description should not be empty"

    data_section = payload["data"]
    if isinstance(data_section, list):
        assert len(data_section) == metadata["count"], f"count mismatch in {path.name}"
        assert all(isinstance(item, dict) for item in data_section)


EXPECTED_FIELDS = {
    "qa": {"question", "answer"},
    "math": {"problem", "answer"},
    "sentiment": {"text", "sentiment"},
    "classification": {"text", "category"},
    "summarization": {"document", "summary"},
}


@pytest.mark.parametrize("path", sorted(DATASETS_DIR.glob("*.jsonl")))
def test_jsonl_records_match_expected_schema(path: Path):
    stem = path.stem.split("_")[0]
    required = EXPECTED_FIELDS.get(stem)
    if required is None:
        pytest.skip(f"No schema assertion for {path.name}")

    for row in _iter_jsonl(path):
        assert required.issubset(row.keys()), f"missing keys in {path.name}: {row}"


@pytest.mark.parametrize(
    "dataset",
    ["qa", "math", "sentiment", "classification", "summarization"],
)
def test_train_test_split_counts_match_metadata(dataset: str):
    train_path = DATASETS_DIR / f"{dataset}_train.json"
    test_path = DATASETS_DIR / f"{dataset}_test.json"
    combined_path = DATASETS_DIR / f"{dataset}.json"

    if not (train_path.exists() and test_path.exists() and combined_path.exists()):
        pytest.skip(f"Missing complete split for {dataset}")

    train_data = _load_json(train_path)["data"]
    test_data = _load_json(test_path)["data"]
    combined = _load_json(combined_path)

    assert len(train_data) + len(test_data) == combined["metadata"]["count"]
