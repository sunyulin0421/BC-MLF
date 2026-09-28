#!/usr/bin/env python3
"""
Convert MMSA/M-SENA pkl feature files into the HF Dataset layout expected by
BC-MLF configs:

    <data_root>/<dataset>/Processed/mms2s/hf_unaligned_<max_len>_<split>.arrow

The generated datasets contain only the fields used by the BC-MLF training
path (bienc/msalm): raw_text, audio, vision, lengths, labels, id, and index.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset


DATASETS = {
    "mosei": {
        "pkl": "Processed/unaligned_50.pkl",
        "max_token_len": 50,
    },
    "sims": {
        "pkl": "Processed/unaligned_39.pkl",
        "max_token_len": 39,
    },
}


def as_float32_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    array[array == -np.inf] = 0
    array[np.isnan(array)] = 0
    return array


def as_1d_float32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def build_labels(split_data: dict[str, Any], dataset_name: str) -> list[dict[str, list[float]]]:
    labels_m = np.asarray(split_data["regression_labels"], dtype=np.float32).reshape(-1)
    labels: list[dict[str, list[float]]] = []

    if dataset_name == "sims":
        labels_t = np.asarray(split_data["regression_labels_T"], dtype=np.float32).reshape(-1)
        labels_a = np.asarray(split_data["regression_labels_A"], dtype=np.float32).reshape(-1)
        labels_v = np.asarray(split_data["regression_labels_V"], dtype=np.float32).reshape(-1)
        for m, t, a, v in zip(labels_m, labels_t, labels_a, labels_v):
            labels.append(
                {
                    "M": [float(m)],
                    "T": [float(t)],
                    "A": [float(a)],
                    "V": [float(v)],
                }
            )
    else:
        for m in labels_m:
            labels.append({"M": [float(m)]})

    return labels


def split_to_dataset(split_data: dict[str, Any], dataset_name: str) -> Dataset:
    sample_count = len(split_data["raw_text"])
    columns: dict[str, Any] = {
        "raw_text": as_list(split_data["raw_text"]),
        "audio": as_float32_array(split_data["audio"]),
        "vision": as_float32_array(split_data["vision"]),
        "labels": build_labels(split_data, dataset_name),
        "index": np.arange(sample_count, dtype=np.int64),
    }

    if "id" in split_data:
        columns["id"] = as_list(split_data["id"])
    else:
        columns["id"] = [str(i) for i in range(sample_count)]

    if "audio_lengths" in split_data:
        columns["audio_lengths"] = np.asarray(split_data["audio_lengths"], dtype=np.int64)
    if "vision_lengths" in split_data:
        columns["vision_lengths"] = np.asarray(split_data["vision_lengths"], dtype=np.int64)

    dataset = Dataset.from_dict(columns)
    format_columns = ["audio", "vision", "labels", "index"]
    if "audio_lengths" in columns:
        format_columns.append("audio_lengths")
    if "vision_lengths" in columns:
        format_columns.append("vision_lengths")
    dataset.set_format(
        type="torch",
        columns=format_columns,
        output_all_columns=True,
    )
    return dataset


def convert_dataset(
    data_root: Path,
    dataset_name: str,
    output_root: Path | None,
    overwrite: bool,
) -> None:
    meta = DATASETS[dataset_name]
    pkl_path = data_root / dataset_name / meta["pkl"]
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Missing pkl file: {pkl_path}")

    out_dir = output_root if output_root else data_root / dataset_name / "Processed" / "mms2s"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_name}] loading {pkl_path}")
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    for split in ("train", "valid", "test"):
        if split not in data:
            raise KeyError(f"{pkl_path} does not contain split: {split}")

        split_out = out_dir / f"hf_unaligned_{meta['max_token_len']}_{split}.arrow"
        if split_out.exists():
            if not overwrite:
                print(f"[{dataset_name}] skip existing {split_out}")
                continue
            shutil.rmtree(split_out)

        print(f"[{dataset_name}] converting {split} -> {split_out}")
        dataset = split_to_dataset(data[split], dataset_name)
        dataset.save_to_disk(str(split_out))
        print(f"[{dataset_name}] saved {split}: {len(dataset)} samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert unaligned MMSA pkl files to BC-MLF mms2s HF datasets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("BCMLF_DATA_ROOT", "data/mmsa")),
        help="Root containing mosei/ and sims/ directories (defaults to BCMLF_DATA_ROOT).",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory. Only use with a single --dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing hf_unaligned_* directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root and args.dataset == "all":
        raise ValueError("--output-root can only be used when converting one dataset")

    dataset_names = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        convert_dataset(args.data_root, dataset_name, args.output_root, args.overwrite)


if __name__ == "__main__":
    main()

