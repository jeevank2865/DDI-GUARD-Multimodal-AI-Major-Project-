"""Download source-linked drug-label text from the official DailyMed SPL API.

These are general drug-label passages, not evidence that a particular pair had a
DDI. Drug-name matching is exact after normalization and every mapping remains
auditable back to the project's DDInter name table.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
SECTIONS = {"34071-1", "34073-7", "34078-6"}
SECTION_NAMES = {"34071-1": "WARNINGS", "34073-7": "DRUG INTERACTIONS",
                 "34078-6": "CLINICAL PHARMACOLOGY"}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _request(url: str, *, timeout: float = 45) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "DDI-GUARD research ingestion/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != "dailymed.nlm.nih.gov":
            raise ValueError(f"Refusing unexpected DailyMed response host: {response.geturl()}")
        return response.read()


def _label_sections(xml_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    result = []
    for section in root.iter():
        if section.tag.rsplit("}", 1)[-1] != "section":
            continue
        code_node = next((child for child in section if child.tag.rsplit("}", 1)[-1] == "code"), None)
        if code_node is None or code_node.attrib.get("code") not in SECTIONS:
            continue
        code = code_node.attrib["code"]
        title = next((child for child in section if child.tag.rsplit("}", 1)[-1] == "title"), None)
        text = " ".join(" ".join(section.itertext()).split())
        if text:
            result.append({"section_code": code, "section_name": SECTION_NAMES.get(code, "LABEL SECTION"),
                           "section_title": " ".join(title.itertext()).strip() if title is not None else "",
                           "text": text})
    return result


def _list_labels(drug_name: str, *, timeout: float) -> tuple[str, list[dict[str, Any]]]:
    query = urllib.parse.urlencode({"drug_name": drug_name, "name_type": "generic", "pagesize": 100, "page": 1})
    url = f"{BASE}/spls.json?{query}"
    payload = json.loads(_request(url, timeout=timeout))
    candidates = payload.get("data", [])
    wanted = _norm(drug_name)
    # Require an exact leading generic name; reject obvious combination products.
    matched = []
    for row in candidates:
        title = _norm(str(row.get("title", "")))
        remainder = title[len(wanted):].strip() if wanted and title.startswith(wanted) else ""
        if remainder and not re.match(r"^(?:and|with|plus)\b|^[+/]", remainder):
            matched.append(row)
    matched.sort(key=lambda r: (str(r.get("published_date", "")), str(r.get("spl_version", ""))), reverse=True)
    return url, matched


def ingest(mapping_csv: Path, output_jsonl: Path, audit_json: Path, *, limit: int | None = None,
           labels_per_drug: int = 1, timeout: float = 45, pause: float = 0.05,
           resume: bool = True, workers: int = 8) -> dict[str, Any]:
    with mapping_csv.open(newline="", encoding="utf-8-sig") as f:
        mappings = list(csv.DictReader(f))
    required = {"ddinter_id", "normalized_name"}
    if not mappings or not required.issubset(mappings[0]):
        raise ValueError(f"mapping CSV must contain {sorted(required)}")
    if labels_per_drug < 1:
        raise ValueError("labels_per_drug must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if limit is not None:
        mappings = mappings[:limit]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    checkpoint_path = audit_json.with_name(audit_json.stem + ".checkpoint.json")
    mapping_hash = _sha(mapping_csv.read_bytes())
    completed: set[str] = set()
    if resume and checkpoint_path.is_file() and output_jsonl.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("mapping_sha256") != mapping_hash:
            raise ValueError("DailyMed resume checkpoint belongs to a different mapping CSV")
        completed = set(checkpoint.get("processed_ddinter_ids", []))
        errors = list(checkpoint.get("errors", []))
    elif output_jsonl.exists() and output_jsonl.stat().st_size:
        raise FileExistsError(f"Evidence output already exists without a matching resume checkpoint: {output_jsonl}")
    valid_ids = {row["ddinter_id"].strip() for row in mappings}
    if not completed <= valid_ids:
        raise ValueError("DailyMed checkpoint contains drug IDs not present in the current mapping")
    if output_jsonl.is_file() and completed:
        existing = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts["covered_drugs"] = len({row["ddinter_id"] for row in existing})
        counts["evidence_records"] = len(existing)
    counts["processed_drugs"] = len(completed)

    def write_progress(status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        progress = {"status": status, "mapping_sha256": mapping_hash,
                    "processed_ddinter_ids": sorted(completed), "errors": errors,
                    "processed_drugs": len(completed), "covered_drugs": counts["covered_drugs"],
                    "evidence_records": counts["evidence_records"], "updated_at_utc": now}
        checkpoint_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit_json.write_text(json.dumps({
            "source": "U.S. National Library of Medicine DailyMed SPL REST API v2",
            "source_documentation": "https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm",
            "status": status, "mapping_file": str(mapping_csv), "mapping_sha256": mapping_hash,
            "output_file": str(output_jsonl), "input_drugs": len(mappings),
            "processed_drugs": len(completed), "covered_drugs": counts["covered_drugs"],
            "evidence_records": counts["evidence_records"], "errors": errors,
            "checkpoint_file": str(checkpoint_path), "updated_at_utc": now,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not completed:
        output_jsonl.write_text("", encoding="utf-8")
    write_progress("running")
    def fetch_drug(drug: dict[str, str]) -> tuple[str, str, list[dict[str, Any]], int, int, str | None]:
        drug_id, name = drug["ddinter_id"].strip(), drug["normalized_name"].strip()
        if not drug_id or not name:
            return drug_id, name, [], 0, 0, "invalid mapping"
        try:
            list_url, labels = _list_labels(name, timeout=timeout)
            records: list[dict[str, Any]] = []
            relevant_labels = 0
            for label in labels[:labels_per_drug]:
                setid = str(label.get("setid", "")).strip()
                if not setid:
                    continue
                source_url = f"{BASE}/spls/{urllib.parse.quote(setid, safe='')}.xml"
                xml_bytes = _request(source_url, timeout=timeout)
                sections = _label_sections(xml_bytes)
                relevant_labels += bool(sections)
                for section in sections:
                    records.append({
                        "ddinter_id": drug_id, "drug_name": name, "evidence_id": f"dailymed:{setid}:{section['section_code']}",
                        "source_id": "DailyMed-SPL", "source_url": source_url,
                        "listing_url": list_url, "set_id": setid,
                        "source_version": str(label.get("spl_version", "")),
                        "source_date": str(label.get("published_date", "")),
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "evidence_scope": "drug_general", "label_independent": True,
                        "mapping_source": "DDInter audited drug_identity_mapping.csv",
                        "mapping_name": drug["normalized_name"], "mapping_ddinter_id": drug_id,
                        **section, "source_document_sha256": _sha(xml_bytes),
                    })
                time.sleep(pause)
            return drug_id, name, records, int(bool(labels)), relevant_labels, None
        except Exception as exc:
            return drug_id, name, [], 0, 0, f"{type(exc).__name__}: {exc}"

    pending = [drug for drug in mappings if drug["ddinter_id"].strip() not in completed]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_drug, drug) for drug in pending]
        for future in as_completed(futures):
            drug_id, name, drug_records, matched, relevant, error = future.result()
            counts["exact_name_label_matches"] += matched
            counts["labels_with_relevant_sections"] += relevant
            if error:
                errors.append({"ddinter_id": drug_id, "drug_name": name, "error": error})
                counts["request_errors"] += 1
            if drug_records:
                with output_jsonl.open("a", encoding="utf-8") as f:
                    for row in drug_records:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                counts["covered_drugs"] += 1
                counts["evidence_records"] += len(drug_records)
            completed.add(drug_id)
            counts["processed_drugs"] = len(completed)
            write_progress("running")

    write_progress("finalizing")
    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    covered = {row["ddinter_id"] for row in records}
    audit = {
        "source": "U.S. National Library of Medicine DailyMed SPL REST API v2",
        "source_documentation": "https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm",
        "source_version": "DailyMed SPL API v2; per-label SPL version/date retained",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_file": str(mapping_csv), "mapping_sha256": _sha(mapping_csv.read_bytes()),
        "output_file": str(output_jsonl), "output_sha256": _sha(output_jsonl.read_bytes()),
        "input_drugs": len(mappings), "covered_drugs": len(covered),
        "coverage_fraction": len(covered) / len(mappings) if mappings else 0.0,
        "evidence_records": len(records), "relevant_section_codes": sorted(SECTIONS),
        "counts": dict(counts), "errors": errors,
        "scope_note": "These passages are general drug-label evidence. They are not pair-specific proof of a DDI or its severity.",
        "status": ("blocked_network" if errors and len(errors) == len(mappings) else
                   "completed_with_partial_errors" if errors else
                   "completed" if records else "completed_no_coverage"),
    }
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    return audit


def vectorize(corpus_jsonl: Path, vectors_dir: Path, manifest_csv: Path, *, max_features: int = 2048,
              fit_drug_ids: set[str] | None = None, fit_scope: str = "all_corpus") -> int:
    """Create TF-IDF inputs and record which drug documents fitted the vocabulary/IDF."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in corpus_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("evidence_scope") != "drug_general" or not row.get("label_independent"):
            continue
        grouped.setdefault(row["ddinter_id"], []).append(row)
    if not grouped:
        raise ValueError("DailyMed corpus contains no label-independent drug-level evidence")
    all_ids = list(grouped)
    texts = [" ".join(row["text"] for row in grouped[drug_id]) for drug_id in all_ids]
    fit_ids = set(all_ids) if fit_drug_ids is None else set(fit_drug_ids) & set(grouped)
    if not fit_ids:
        raise ValueError("TF-IDF fit population has no covered DailyMed drug text")
    fit_texts = [" ".join(row["text"] for row in grouped[drug_id])
                 for drug_id in all_ids if drug_id in fit_ids]
    vectorizer = TfidfVectorizer(max_features=max_features, dtype=np.float32, stop_words="english")
    vectorizer.fit(fit_texts)
    matrix = vectorizer.transform(texts).toarray()
    vectors_dir.mkdir(parents=True, exist_ok=True)
    corpus_hash = _sha(corpus_jsonl.read_bytes())
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fit_ids_sha256 = _sha("\n".join(sorted(fit_ids)).encode("utf-8"))
    fields = ["ddinter_id", "source_id", "evidence_id", "source_url", "source_urls_json",
              "source_version", "source_versions_json", "source_date", "source_dates_json",
              "source_document_sha256s_json", "fit_scope", "fit_drug_count", "fit_drug_ids_sha256",
              "evidence_scope", "label_independent",
              "embedding_path", "embedding_sha256"]
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index, (drug_id, rows) in enumerate(grouped.items()):
            path = vectors_dir / f"{drug_id}.npy"
            np.save(path, matrix[index])
            writer.writerow({
                "ddinter_id": drug_id, "source_id": "DailyMed-SPL",
                "evidence_id": ";".join(row["evidence_id"] for row in rows),
                "source_url": rows[0]["source_url"],
                "source_urls_json": json.dumps(sorted({row["source_url"] for row in rows})),
                "source_version": f"DailyMed SPL; TF-IDF; corpus_sha256={corpus_hash}",
                "source_versions_json": json.dumps(sorted({row.get("source_version", "") for row in rows})),
                "source_date": max((row.get("source_date", "") for row in rows), default=""),
                "source_dates_json": json.dumps(sorted({row.get("source_date", "") for row in rows})),
                "source_document_sha256s_json": json.dumps(sorted({row["source_document_sha256"] for row in rows})),
                "fit_scope": fit_scope, "fit_drug_count": len(fit_ids),
                "fit_drug_ids_sha256": fit_ids_sha256,
                "evidence_scope": "drug_general", "label_independent": "true",
                "embedding_path": str(path), "embedding_sha256": _sha(path.read_bytes()),
            })
    return len(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=Path("data/processed/drug_identity_mapping.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dailymed_drug_evidence.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/dailymed_evidence_coverage_audit.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--labels-per-drug", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vectorize", action="store_true", help="also build label-independent TF-IDF vectors from an existing corpus")
    parser.add_argument("--vectorize-existing", action="store_true", help="vectorize the existing corpus without network access")
    parser.add_argument("--vectors-dir", type=Path, default=Path("data/processed/dailymed_vectors"))
    parser.add_argument("--text-manifest", type=Path, default=Path("data/processed/text_evidence_embeddings_manifest.csv"))
    parser.add_argument("--fit-split", choices=("pair_disjoint", "cold_drug"),
                        help="fit TF-IDF vocabulary/IDF on covered drugs occurring in this fixed training split only")
    args = parser.parse_args()
    if args.vectorize_existing:
        if not args.audit.is_file():
            parser.error("--vectorize-existing requires an existing evidence coverage audit")
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
    else:
        audit = ingest(args.mapping, args.output, args.audit, limit=args.limit,
                       labels_per_drug=args.labels_per_drug, workers=args.workers)
    if args.vectorize or args.vectorize_existing:
        fit_ids = None
        fit_scope = "all_corpus_unsupervised"
        if args.fit_split:
            project_root = Path.cwd()
            pairs = json.loads((project_root / "data/processed/ddi_pairs_full.json").read_text(encoding="utf-8"))
            split = json.loads((project_root / "artifacts/split_manifest.json").read_text(encoding="utf-8"))[args.fit_split]
            pairs_by_id = {row["pair_id"]: row for row in pairs}
            fit_ids = {pairs_by_id[pair_id][key] for pair_id in split["train_pair_ids"]
                       for key in ("drug_a_id", "drug_b_id")}
            fit_scope = f"{args.fit_split}_train_drug_text_only"
        audit["vectorized_drugs"] = vectorize(args.output, args.vectors_dir, args.text_manifest,
                                              fit_drug_ids=fit_ids, fit_scope=fit_scope)
        audit["text_manifest"] = str(args.text_manifest)
        audit["tfidf_fit_scope"] = fit_scope
        args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
