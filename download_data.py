"""Download a Hugging Face text dataset and convert it into GPT-2 token ids.

Writes:
    data/raw/<dataset>/...parquet      the original file(s)
    data/tokens/<name>.npy             token ids as uint16
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

# ---------------------------------------------------------------- settings --
DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"      # leave "" for datasets without configs
SPLIT = "train"
TEXT_COLUMN = "text"
MAX_TOKENS = 2_000_000            # cap so a laptop can train on it
END_OF_TEXT = 50256

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw" / DATASET.replace("/", "__")
TOKEN_DIR = BASE_DIR / "data" / "tokens"


def token_file():
    name = f"{DATASET.replace('/', '__')}_{CONFIG or 'default'}_{SPLIT}.npy"
    return TOKEN_DIR / name


def find_parquet_files():
    """List the parquet files for the chosen config and split."""
    files = HfApi().list_repo_files(DATASET, repo_type="dataset")
    prefix = f"{CONFIG}/" if CONFIG else ""
    matches = [f for f in files
               if f.endswith(".parquet") and f.startswith(prefix)
               and f.split("/")[-1].startswith(SPLIT)]
    if not matches:
        sys.exit(f"No parquet files found for {DATASET} / {CONFIG} / {SPLIT}")
    return sorted(matches)


def main():
    out = token_file()
    if out.exists():
        print(f"{out.name} already exists, nothing to do.")
        return

    tokenizer = Tokenizer.from_pretrained("gpt2")
    ids = []
    for name in find_parquet_files():
        path = hf_hub_download(DATASET, name, repo_type="dataset", local_dir=RAW_DIR)
        texts = pq.read_table(path, columns=[TEXT_COLUMN]).column(0).to_pylist()
        print(f"read    : {name}  ({len(texts):,} rows)")

        for text in texts:
            if text and text.strip():
                ids.extend(tokenizer.encode(text).ids)
                ids.append(END_OF_TEXT)
            if len(ids) >= MAX_TOKENS:
                break
        if len(ids) >= MAX_TOKENS:
            break

    tokens = np.array(ids[:MAX_TOKENS], dtype=np.uint16)
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out, tokens)
    print(f"saved   : {out}  ({len(tokens):,} tokens)")


if __name__ == "__main__":
    main()
