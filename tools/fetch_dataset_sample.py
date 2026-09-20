"""Fetch a prefix of a MiniMind dataset file from ModelScope.

The pretrain corpus is 1.2 GB (7.9 GB for the full variant), but a short
training benchmark reads only the first few thousand lines. An HTTP range
request pulls just that prefix, which turns a multi-minute download on a
metered box into a few seconds.

The prefix is truncated at the last complete line, split into train and
holdout, and written with a provenance record so a later report can say
exactly which bytes produced it.

    python -m tools.fetch_dataset_sample --samples 20000 --holdout 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELSCOPE_DATASET = "gongjy/minimind_dataset"
# Average record in pretrain_t2t_mini.jsonl is ~775 bytes; the margin covers
# files with longer documents so one request is normally enough.
ESTIMATED_BYTES_PER_RECORD = 1200


def download_url(dataset: str, file_path: str, revision: str) -> str:
    return (
        f"https://www.modelscope.cn/api/v1/datasets/{dataset}/repo"
        f"?Revision={revision}&FilePath={file_path}"
    )


def fetch_prefix(url: str, byte_count: int, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{byte_count - 1}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status not in (200, 206):
            raise RuntimeError(f"unexpected status {response.status} from {url}")
        # Some proxies ignore Range and answer 200 with the full multi-GB file.
        # Never read beyond the requested prefix into memory or paid runtime.
        return response.read(byte_count)


def complete_records(payload: bytes, limit: int) -> tuple[list[dict], int]:
    """Parse whole JSON lines from a byte prefix, ignoring a truncated tail."""
    end = payload.rfind(b"\n")
    body = payload if end == -1 else payload[:end]
    records: list[dict] = []
    for line in body.split(b"\n"):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A mid-character cut can only affect the final line.
            break
        if len(records) >= limit:
            break
    return records, len(body)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=MODELSCOPE_DATASET)
    parser.add_argument("--file", default="pretrain_t2t_mini.jsonl")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--samples", type=int, default=20000, help="records kept for training")
    parser.add_argument("--holdout", type=int, default=2000, help="records split off for validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--prefix", default=None, help="output basename (defaults to the source stem)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.holdout < 0:
        raise SystemExit("samples must be positive and holdout non-negative")
    wanted = args.samples + args.holdout
    url = download_url(args.dataset, args.file, args.revision)
    byte_count = wanted * ESTIMATED_BYTES_PER_RECORD

    print(f"fetching ~{byte_count / 1024**2:.1f} MB prefix of {args.file} from {args.dataset}")
    try:
        payload = fetch_prefix(url, byte_count, args.timeout)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
        print(f"download failed: {error}", file=sys.stderr)
        print(
            "If ModelScope is unreachable, download the file manually from\n"
            f"  https://www.modelscope.cn/datasets/{args.dataset}/files\n"
            "and pass it to the suite with --train-data.",
            file=sys.stderr,
        )
        return 1

    records, consumed = complete_records(payload, wanted)
    if not records:
        print("no complete records in the downloaded prefix", file=sys.stderr)
        return 1
    if args.text_field not in records[0]:
        print(
            f"records have fields {sorted(records[0])}, not {args.text_field!r}; "
            "pass --text-field to match",
            file=sys.stderr,
        )
        return 1
    if len(records) < wanted:
        print(
            f"warning: asked for {wanted} records, the prefix yielded {len(records)}; "
            "raise --samples budget or fetch the file in full",
            file=sys.stderr,
        )

    holdout = min(args.holdout, max(0, len(records) - 1))
    train_records = records[: len(records) - holdout]
    holdout_records = records[len(records) - holdout :] if holdout else []

    stem = args.prefix or Path(args.file).stem
    train_path = args.out_dir / f"{stem}_train.jsonl"
    write_jsonl(train_path, train_records)
    outputs = {"train": str(train_path), "train_records": len(train_records)}
    if holdout_records:
        holdout_path = args.out_dir / f"{stem}_holdout.jsonl"
        write_jsonl(holdout_path, holdout_records)
        outputs |= {"validation": str(holdout_path), "validation_records": len(holdout_records)}

    provenance = {
        "source": "modelscope",
        "dataset": args.dataset,
        "file": args.file,
        "revision": args.revision,
        "url": url,
        "downloaded_bytes": len(payload),
        "parsed_bytes": consumed,
        "prefix_sha256": hashlib.sha256(payload[:consumed]).hexdigest(),
        "text_field": args.text_field,
        **outputs,
    }
    provenance_path = args.out_dir / f"{stem}_provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"train      : {outputs['train']} ({outputs['train_records']} records)")
    if holdout_records:
        print(f"validation : {outputs['validation']} ({outputs['validation_records']} records)")
    print(f"provenance : {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
