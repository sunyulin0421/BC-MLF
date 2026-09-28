#!/usr/bin/env python3
"""
Download the language models used by BC-MLF into the Hugging Face cache.

The script is retry-friendly: files already present in the cache are reused on
the next run, so it is safe to rerun after a network timeout.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = [
    "gpt2",
    "gpt2-large",
    "HuggingFaceTB/SmolLM2-1.7B",
    "uer/gpt2-chinese-cluecorpussmall",
]

ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "model-*.safetensors",
    "pytorch_model-*.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
]

IGNORE_PATTERNS = [
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "*.tflite",
    "onnx/*",
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
]


def download_snapshot(model_name: str, cache_dir: str | None) -> None:
    print(f"\nDownloading snapshot: {model_name}", flush=True)
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )


def verify_transformers_load(model_name: str, cache_dir: str | None) -> None:
    print(f"Verifying tokenizer: {model_name}", flush=True)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    print(f"Verifying model: {model_name}", flush=True)
    AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)


def download_with_retries(
    models: Iterable[str],
    cache_dir: str | None,
    max_retries: int,
    sleep_seconds: int,
    verify: bool,
) -> None:
    for model_name in models:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"\n[{model_name}] attempt {attempt}/{max_retries}", flush=True)
                download_snapshot(model_name, cache_dir)
                if verify:
                    verify_transformers_load(model_name, cache_dir)
                print(f"[{model_name}] done", flush=True)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"[{model_name}] failed: {type(exc).__name__}: {exc}", flush=True)
                if attempt < max_retries:
                    print(f"[{model_name}] retrying in {sleep_seconds}s", flush=True)
                    time.sleep(sleep_seconds)
        if last_error is not None:
            raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BC-MLF language models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model repo ids to download.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Retry count per model.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=20,
        help="Seconds to sleep between retries.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After download, verify AutoTokenizer/AutoModel can load offline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # These environment variables are honored by recent huggingface_hub versions.
    # They are harmless on older versions and help slow mirrors avoid short reads.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    download_with_retries(
        models=args.models,
        cache_dir=args.cache_dir,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep,
        verify=args.verify,
    )
    print("\nAll requested models downloaded.", flush=True)


if __name__ == "__main__":
    main()

