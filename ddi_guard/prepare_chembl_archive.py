"""Verify the official ChEMBL 34 archive checksum and safely extract its SQLite data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

EXPECTED_SHA256 = "7d40409e4f31440674a79e26db628015e82b85b056a7d8234efe9c95b6cbd8e7"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(archive: Path, destination: Path, expected_sha256: str = EXPECTED_SHA256) -> dict[str, Any]:
    observed = sha256_file(archive)
    if observed.lower() != expected_sha256.lower():
        raise ValueError(f"Archive checksum mismatch: expected {expected_sha256}, observed {observed}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        total_bytes = 0
        db_members = []
        for member in members:
            member_path = PurePosixPath(member.name)
            windows_path = PureWindowsPath(member.name)
            if (member_path.is_absolute() or ".." in member_path.parts
                    or windows_path.is_absolute() or windows_path.drive or ".." in windows_path.parts):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported non-regular archive member: {member.name}")
            total_bytes += member.size
            if member.isfile() and (member.name.endswith(".db") or member.name.endswith(".sqlite") or member.name.endswith(".sqlite3")):
                db_members.append(member.name)
        if not db_members:
            raise ValueError("No SQLite database member found in the verified archive")
        expected_files = {str(PurePosixPath(m.name)): m.size for m in members if m.isfile()}
        existing_files = {str(p.relative_to(destination).as_posix()): p.stat().st_size
                          for p in destination.rglob("*") if p.is_file() and not p.is_symlink()}
        symlinks = [p for p in destination.rglob("*") if p.is_symlink()]
        if symlinks or any(p.is_file() and p.is_symlink() for p in destination.rglob("*")):
            raise ValueError(f"Extraction directory contains symlinks: {destination}")
        if existing_files:
            if existing_files != expected_files:
                raise FileExistsError(f"Extraction destination is non-empty and does not exactly match archive members: {destination}")
            # A previous verified extraction may have completed before a later
            # schema check failed. Reuse exact member paths/sizes; never overwrite.
            free_bytes = shutil.disk_usage(destination).free
        else:
            free_bytes = shutil.disk_usage(destination).free
            minimum_free_bytes = total_bytes + 512 * 1024 * 1024
            if free_bytes < minimum_free_bytes:
                raise OSError(f"Insufficient extraction space: need at least {minimum_free_bytes} bytes, have {free_bytes}")
            tar.extractall(destination, members=members, filter="data")
    readability = []
    for member_name in db_members:
        db_path = destination.joinpath(*PurePosixPath(member_name).parts)
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        try:
            quick_check = connection.execute("PRAGMA quick_check(1)").fetchone()
            if not quick_check or quick_check[0] != "ok":
                raise ValueError(f"Extracted database failed SQLite quick_check: {member_name}: {quick_check}")
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "version" not in tables:
                raise ValueError(f"Extracted database has no ChEMBL version table: {member_name}")
            version_cols = {row[1] for row in connection.execute('PRAGMA table_info("version")')}
            version_column = "version" if "version" in version_cols else "name" if "name" in version_cols else None
            if version_column is None:
                raise ValueError(f"Extracted database version table has no supported version/name column: {member_name}")
            import re
            values = [str(row[0]) for row in connection.execute(f'SELECT "{version_column}" FROM "version"')
                      if row[0] is not None]
            release = None
            for value in values:
                match = re.fullmatch(r"ChEMBL[_ -]?(\d+)", value, flags=re.IGNORECASE)
                if match:
                    release = match.group(1)
                    break
                if version_column == "version" and value.isdigit():
                    release = value
                    break
            if release != "34":
                raise ValueError(f"Extracted database release mismatch: expected 34, observed {release}")
            readability.append({"database_member": member_name, "sqlite_quick_check": "ok",
                                "observed_release": release})
        finally:
            connection.close()
    return {"status": "extracted", "archive": str(archive), "sha256": observed,
            "extraction_directory": str(destination), "database_members": db_members,
            "member_count": len(members), "uncompressed_total_member_bytes": total_bytes,
            "free_disk_bytes_before_extraction": free_bytes,
            "database_readability_checks": readability,
            "generated_at_utc": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/private/tmp/chembl_34_sqlite.tar.gz"))
    parser.add_argument("--destination", type=Path, default=Path("/private/tmp/chembl34_extracted"))
    args = parser.parse_args()
    result = prepare(args.archive, args.destination)
    audit = Path("artifacts/chembl_34_source_audit.json")
    obj = json.loads(audit.read_text(encoding="utf-8")) if audit.is_file() else {}
    obj.update(result)
    audit.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
