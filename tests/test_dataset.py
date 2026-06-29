from __future__ import annotations

from evalkit.dataset import DataRow, load_dataset, split_dataset


def test_load_jsonl_accepts_common_aliases(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"id":"1","question":"Q","answer":"A","topic":"demo"}\n', encoding="utf-8")

    rows = load_dataset(path)

    assert rows == [DataRow(id="1", input="Q", expected_output="A", context=None, metadata={"topic": "demo"})]


def test_split_dataset_is_deterministic_and_complete():
    rows = [DataRow(id=str(index), input=f"q{index}", expected_output=f"a{index}") for index in range(10)]

    first = split_dataset(rows, {"train": 0.6, "test": 0.2, "validation": 0.2}, seed=42)
    second = split_dataset(rows, {"train": 0.6, "test": 0.2, "validation": 0.2}, seed=42)

    assert {key: [row.id for row in value] for key, value in first.items()} == {
        key: [row.id for row in value] for key, value in second.items()
    }
    assert {key: len(value) for key, value in first.items()} == {
        "train": 6,
        "test": 2,
        "validation": 2,
    }
    assert sorted(row.id for split in first.values() for row in split) == [str(index) for index in range(10)]
