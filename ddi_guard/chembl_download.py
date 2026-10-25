"""Resumable, range-validated downloader for the official ChEMBL 34 archive."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import Any

from .prepare_chembl_archive import EXPECTED_SHA256, sha256_file

OFFICIAL_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    "chembl_34/chembl_34_sqlite.tar.gz"
)
EXPECTED_SIZE = 4_859_330_198
OFFICIAL_HOST = "ftp.ebi.ac.uk"


def _request(url: str, method: str, headers: dict[str, str] | None = None,
             timeout: float = 60.0):
    request = Request(url, method=method, headers={"User-Agent": "DDI-GUARD/0.1"} | (headers or {}))
    return urlopen(request, timeout=timeout)


def _check_host(url: str, official_only: bool) -> None:
    parsed = urlparse(url)
    if not parsed.hostname or parsed.scheme not in ({"https"} if official_only else {"http", "https"}):
        raise ValueError("ChEMBL downloads must use HTTPS with a host name")
    if official_only and parsed.hostname.casefold() != OFFICIAL_HOST:
        raise ValueError(f"Refusing non-official ChEMBL host: {parsed.hostname}")


def _download_archive_once(url: str, destination: Path, *, expected_size: int = EXPECTED_SIZE,
                           expected_sha256: str = EXPECTED_SHA256, partial_path: Path | None = None,
                           timeout: float = 60.0, chunk_size: int = 1024 * 1024,
                           official_only: bool = True) -> dict[str, Any]:
    """Resume a partial file only after verifying HTTP byte-range semantics.

    Partial bytes are retained across network and process failures. A final file is
    atomically installed only after size and SHA-256 validation. An existing path
    can be used as the partial path to resume the historical archive in place.
    """
    _check_host(url, official_only)
    if expected_size < 1 or chunk_size < 1 or timeout <= 0:
        raise ValueError("expected_size, chunk_size, and timeout must be positive")
    destination = Path(destination)
    partial = Path(partial_path) if partial_path is not None else destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    lock = partial.with_name(partial.name + ".download.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Another downloader holds the lock: {lock}") from exc
    os.write(lock_fd, f"pid={os.getpid()}\nstarted={datetime.now(timezone.utc).isoformat()}\n".encode())
    os.close(lock_fd)
    started = time.monotonic()
    try:
        with _request(url, "HEAD", timeout=timeout) as response:
            final_url = response.geturl()
            _check_host(final_url, official_only)
            remote_size = int(response.headers.get("Content-Length", "0"))
            accept_ranges = response.headers.get("Accept-Ranges", "").casefold()
        if remote_size != expected_size:
            raise ValueError(f"Remote archive size mismatch: expected {expected_size}, observed {remote_size}")
        if "bytes" not in accept_ranges:
            raise ValueError("Official server did not advertise byte-range support")

        # If the final destination already exists, never overwrite it. It may be
        # the historical partial archive when destination and partial are identical.
        if destination.exists() and destination.resolve() != partial.resolve():
            raise FileExistsError(f"Destination already exists: {destination}")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError(f"Partial archive is larger than expected: {offset} > {expected_size}")
        if offset == expected_size:
            observed_hash = sha256_file(partial)
            if observed_hash.casefold() != expected_sha256.casefold():
                quarantine = partial.with_name(partial.name + f".bad-checksum-{int(time.time())}")
                partial.replace(quarantine)
                raise ValueError(f"Complete archive checksum mismatch; quarantined at {quarantine}; observed {observed_hash}")
            if destination.resolve() != partial.resolve():
                os.replace(partial, destination)
            return {"status": "completed", "destination": str(destination), "partial_path": str(partial),
                    "bytes": expected_size, "sha256": observed_hash, "resumed_from_bytes": expected_size,
                    "downloaded_this_run_bytes": 0, "duration_seconds": 0.0}

        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            response = _request(url, "GET", headers=headers, timeout=timeout)
        except HTTPError as exc:
            raise RuntimeError(f"Archive request failed with HTTP {exc.code}") from exc
        with response:
            final_url = response.geturl()
            _check_host(final_url, official_only)
            status = response.getcode()
            if offset:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if status != 206 or not match or int(match.group(1)) != offset or int(match.group(3)) != expected_size:
                    raise RuntimeError(f"Server did not honor resume offset {offset}; HTTP {status}, Content-Range={content_range!r}")
                expected_response_bytes = int(match.group(2)) - offset + 1
            else:
                if status != 200:
                    raise RuntimeError(f"Expected HTTP 200 for a fresh download, observed {status}")
                expected_response_bytes = expected_size
            response_length = response.headers.get("Content-Length")
            if response_length is not None and int(response_length) != expected_response_bytes:
                raise RuntimeError("Response Content-Length disagrees with the validated range")
            mode = "ab" if offset else "wb"
            written = 0
            with partial.open(mode) as output:
                while True:
                    block = response.read(chunk_size)
                    if not block:
                        break
                    output.write(block)
                    written += len(block)
                output.flush()
                os.fsync(output.fileno())
        observed_size = partial.stat().st_size
        if written != expected_response_bytes or observed_size != expected_size:
            return {"status": "partial", "destination": str(destination), "partial_path": str(partial),
                    "bytes": observed_size, "expected_bytes": expected_size,
                    "resumed_from_bytes": offset, "downloaded_this_run_bytes": written,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "reason": "Response ended before the advertised archive range completed; rerun to resume."}
        observed_hash = sha256_file(partial)
        if observed_hash.casefold() != expected_sha256.casefold():
            quarantine = partial.with_name(partial.name + f".bad-checksum-{int(time.time())}")
            partial.replace(quarantine)
            raise ValueError(f"Archive checksum mismatch; quarantined at {quarantine}; observed {observed_hash}")
        if destination.resolve() != partial.resolve():
            os.replace(partial, destination)
        return {"status": "completed", "destination": str(destination), "partial_path": str(partial),
                "bytes": observed_size, "sha256": observed_hash, "resumed_from_bytes": offset,
                "downloaded_this_run_bytes": written,
                "duration_seconds": round(time.monotonic() - started, 3)}
    except (URLError, TimeoutError, OSError) as exc:
        observed_size = partial.stat().st_size if partial.exists() else 0
        raise RuntimeError(f"Download interrupted; kept {observed_size} resumable bytes at {partial}: {exc}") from exc
    finally:
        lock.unlink(missing_ok=True)


def _is_transient_network_error(error: BaseException | None) -> bool:
    """Recognize connection failures that are safe to retry from the saved offset."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (TimeoutError, ConnectionError, URLError,
                              http.client.IncompleteRead, http.client.RemoteDisconnected)):
            return True
        error = getattr(error, "reason", None) or error.__cause__
    return False


def download_archive(url: str, destination: Path, *, expected_size: int = EXPECTED_SIZE,
                     expected_sha256: str = EXPECTED_SHA256, partial_path: Path | None = None,
                     timeout: float = 60.0, chunk_size: int = 1024 * 1024,
                     official_only: bool = True, max_retries: int = 5,
                     retry_backoff_seconds: float = 1.0) -> dict[str, Any]:
    """Download with bounded retries, resuming each retry from validated bytes on disk."""
    if max_retries < 0 or retry_backoff_seconds < 0:
        raise ValueError("max_retries and retry_backoff_seconds must be non-negative")
    for attempt in range(max_retries + 1):
        try:
            return _download_archive_once(
                url, destination, expected_size=expected_size, expected_sha256=expected_sha256,
                partial_path=partial_path, timeout=timeout, chunk_size=chunk_size,
                official_only=official_only)
        except RuntimeError as exc:
            if attempt >= max_retries or not _is_transient_network_error(exc.__cause__):
                raise
            time.sleep(min(retry_backoff_seconds * (2 ** attempt), 30.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=OFFICIAL_URL)
    parser.add_argument("--destination", type=Path, default=Path("/private/tmp/chembl_34_sqlite.tar.gz"))
    parser.add_argument("--partial-path", type=Path, help="existing resumable file; defaults to DESTINATION.part")
    parser.add_argument("--expected-size", type=int, default=EXPECTED_SIZE)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--audit-path", type=Path, default=Path("artifacts/chembl_34_source_audit.json"))
    args = parser.parse_args()
    partial_path = args.partial_path or args.destination.with_name(args.destination.name + ".part")

    def update_audit(status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        if not args.audit_path.is_file():
            return
        audit = json.loads(args.audit_path.read_text(encoding="utf-8"))
        observed_bytes = partial_path.stat().st_size if partial_path.exists() else 0
        audit.update({"download_status": status, "downloaded_bytes": observed_bytes,
                      "expected_bytes": args.expected_size,
                      "download_updated_at_utc": datetime.now(timezone.utc).isoformat()})
        if result is not None:
            audit["download_result"] = result
        if error is not None:
            audit["download_error"] = error
        else:
            audit.pop("download_error", None)
        temporary = args.audit_path.with_name(args.audit_path.name + ".tmp")
        temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.audit_path)
        coverage_path = args.audit_path.with_name("chembl_coverage_audit.json")
        if coverage_path.is_file():
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            coverage["status"] = ("blocked_archive_download_incomplete" if status != "completed"
                                  else "blocked_import_not_run")
            coverage["official_chEMBL_34_archive"] = {
                "path": str(partial_path), "status": status,
                "expected_sha256": args.expected_sha256,
                "downloaded_bytes": observed_bytes, "expected_bytes": args.expected_size,
                "downloaded_fraction": observed_bytes / args.expected_size if args.expected_size else None,
            }
            coverage["reason"] = ("The official ChEMBL 34 archive is still incomplete and resumable."
                                  if status != "completed" else
                                  "The verified archive is ready for safe extraction and ChEMBL 34 import.")
            coverage_tmp = coverage_path.with_name(coverage_path.name + ".tmp")
            coverage_tmp.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(coverage_tmp, coverage_path)

    update_audit("downloading_resumable")
    try:
        result = download_archive(args.url, args.destination, expected_size=args.expected_size,
                                  expected_sha256=args.expected_sha256, partial_path=args.partial_path,
                                  timeout=args.timeout)
    except Exception as exc:
        update_audit("partial_resumable_interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    update_audit(result["status"], result=result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
