"""Download and verify PTB-XL v1.0.3 from PhysioNet's official open-data distribution.

Source:    https://physionet.org/content/ptb-xl/1.0.3/  (DOI 10.13026/kfzx-aw45)
Files:     s3://physionet-open/ptb-xl/1.0.3/ — PhysioNet's AWS open-data bucket, listed on the
           dataset page above (accessed anonymously over HTTPS).
Checksums: SHA256SUMS.txt fetched from physionet.org (not from the bucket).
License:   Creative Commons Attribution 4.0 International
Target:    ml/data/raw/ptb-xl-1.0.3/ (gitignored)

Every file listed in SHA256SUMS.txt is downloaded and its SHA-256 verified; the run fails if
any file is missing or mismatched. Re-running skips files that already verify.

The equivalent single-archive download (https://physionet.org/content/ptb-xl/get-zip/1.0.3/,
1,839,504,686 bytes, no published archive checksum) is throttled to ~150 KiB/s, which is why
per-file download is used.

Usage:  uv run python scripts/download_ptbxl.py [--workers 32]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

VERSION = "1.0.3"
PAGE_URL = f"https://physionet.org/content/ptb-xl/{VERSION}/"
SHA256SUMS_URL = f"https://physionet.org/files/ptb-xl/{VERSION}/SHA256SUMS.txt"
FILE_BASE_URL = f"https://physionet-open.s3.amazonaws.com/ptb-xl/{VERSION}/"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "ml" / "data" / "raw" / f"ptb-xl-{VERSION}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_verified(rel: str, digest: str, root: Path) -> tuple[str, str]:
    """Return (rel, status) with status in {'cached', 'downloaded', 'mismatch'}."""
    dest = root / rel
    if dest.is_file() and sha256_bytes(dest.read_bytes()) == digest:
        return rel, "cached"
    data = fetch(FILE_BASE_URL + rel)
    if sha256_bytes(data) != digest:
        return rel, "mismatch"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return rel, "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    root: Path = args.dataset_dir
    root.mkdir(parents=True, exist_ok=True)

    print(f"Fetching authoritative checksums: {SHA256SUMS_URL}")
    sums_bytes = fetch(SHA256SUMS_URL)
    entries = []
    for line in sums_bytes.decode().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            entries.append((rel.strip(), digest))
    print(f"{len(entries)} files listed")

    status: dict[str, list[str]] = {"cached": [], "downloaded": [], "mismatch": []}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_verified, rel, d, root): rel for rel, d in entries}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="files"):
            try:
                rel, s = fut.result()
                status[s].append(rel)
            except Exception as exc:  # noqa: BLE001 - report every failure, then fail the run
                failed.append(f"{futures[fut]}: {exc}")

    # SHA256SUMS.txt does not list itself; keep the physionet.org copy alongside the data.
    (root / "SHA256SUMS.txt").write_bytes(sums_bytes)

    listed = {rel for rel, _ in entries}
    extra = sorted(
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) not in listed | {"SHA256SUMS.txt"}
    )
    total_bytes = sum((root / rel).stat().st_size for rel in listed if (root / rel).is_file())
    manifest = {
        "dataset": "PTB-XL",
        "version": VERSION,
        "page_url": PAGE_URL,
        "file_base_url": FILE_BASE_URL,
        "sha256sums_url": SHA256SUMS_URL,
        "sha256sums_sha256": sha256_bytes(sums_bytes),
        "license": "CC BY 4.0",
        "doi": "10.13026/kfzx-aw45",
        "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_dir": str(root.relative_to(REPO_ROOT)),
        "files_listed": len(entries),
        "files_verified": len(status["cached"]) + len(status["downloaded"]),
        "files_downloaded_this_run": len(status["downloaded"]),
        "dataset_bytes": total_bytes,
        "sha256_mismatched": sorted(status["mismatch"]),
        "download_failed": sorted(failed),
        "unlisted_files_present": extra,
    }
    manifest_path = root.parent / f"ptb-xl-{VERSION}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if not isinstance(v, list)}, indent=2))
    print(f"mismatched={len(status['mismatch'])} failed={len(failed)} unlisted={len(extra)}")
    if status["mismatch"] or failed:
        print("VERIFICATION FAILED — see manifest", file=sys.stderr)
        return 1
    print(f"All {len(entries)} files verified against PhysioNet SHA256SUMS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
