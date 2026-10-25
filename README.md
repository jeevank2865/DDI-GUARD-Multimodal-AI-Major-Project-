# DDI-GUARD

DDI-GUARD is a B.Tech research project for multimodal, explainable classification of DDInter drug–drug interaction labels (Minor, Moderate, Major). It combines molecular graphs, a ChEMBL 34 drug–target graph, and provenance-backed DailyMed drug-label text.

**Use boundary:** the separate research endpoint is for local academic demonstration only. DDInter labels are dataset annotations, not patient outcomes. This project is not clinically validated and is not for diagnosis, treatment, or prescribing. Production `POST /api/predict` remains disabled and returns HTTP 503.

## Current status

- ChEMBL 34 was imported after archive checksum and SQLite/release verification. The graph has 105,982 filtered source edges, 1,166 drugs, 2,699 targets, and 745 relations.
- DailyMed has 879 source-backed passages and 652 vectorized drugs among 1,939 mapped drugs; 403 retrieval errors (393 HTTP 429) remain. Corpus and manifests are preserved; coverage is partial.
- Full molecular+target+text experiments completed on fixed pair-disjoint and cold-drug common supports. Pair-disjoint full-model test macro-F1 is 0.7473 on 1,802 pairs; cold-drug macro-F1 is 0.3664 on 253 pairs (14 Minor). These are macro-F1 values, not accuracy.
- The all-three-modality candidate was selected using validation macro-F1/loss, not held-out test performance. The hash-pinned research endpoint uses that model when all three modalities exist, then verified molecular+target, molecular-only, or target-only checkpoints when their trained inputs are present. Unsupported pairs receive an explicit error.
- Current verification snapshot (2026-10-21): 73 tests passed, Python compilation and inline JavaScript syntax checks passed, and saved pair-disjoint probability parity passed for all four registered checkpoints. FastAPI TestClient and live loopback curl exercised real inference through the research route; production `/api/predict` returned HTTP 503. Browser/WebGL visual checks remain unverified. The backend now reads assistant settings from the ignored project `.env` automatically; a real provider request is still required to verify live connectivity. See `artifacts/research_prediction_enablement_report.md` for the full audit and exact results.

See [`artifacts/final_audit_report.md`](artifacts/final_audit_report.md), [`artifacts/implementation_matrix.md`](artifacts/implementation_matrix.md), and [`artifacts/model_card.md`](artifacts/model_card.md) for evidence and caveats. The final-year summary and pre-demo checklist are [`artifacts/final_year_project_summary.md`](artifacts/final_year_project_summary.md) and [`artifacts/demo_checklist.md`](artifacts/demo_checklist.md).

## Mac setup

The verified workspace interpreter is `python3` (Python 3.13.12). In a fresh environment, install the API extras and confirm PyTorch, PyTorch Geometric, and RDKit wheels match your Mac CPU/architecture before installing all project dependencies:

```bash
cd "/path/to/DDI-GUARD"
python3 -m pip install -e '.[api]'
python3 -c 'import torch, torch_geometric, rdkit, fastapi; print(torch.__version__, torch_geometric.__version__, rdkit.__version__, fastapi.__version__)'
```

The dependency versions bundled in the verified machine environment may differ from a fresh install. Prefer the preconfigured environment on this Mac for exact reproduction.

## Start the local research platform

```bash
cd "/path/to/DDI-GUARD"
python3 -m uvicorn ddi_guard.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The backend serves both the API and the static frontend; there is no separate frontend build or development server. Do not open `ddi_guard/static/index.html` directly as a `file://` document: its relative `/api/...` calls have no HTTP origin and the browser reports a network failure. The page now explains this and shows the correct local URL; API connection failures also identify the configured backend origin.

The frontend routes backend requests through one API client. By default it uses the current same origin, so no port/proxy/CORS configuration is needed for the documented setup. If you intentionally host the frontend separately, set `DDI_GUARD_PUBLIC_API_BASE_URL` to the backend HTTP(S) origin and `DDI_GUARD_CORS_ORIGINS` to the exact comma-separated frontend origins. Both are public routing settings, not secrets. Set them in the backend environment before startup, for example:

```bash
export DDI_GUARD_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
export DDI_GUARD_CORS_ORIGINS=http://localhost:5173
```

Then start Uvicorn with the command above. For the default same-origin app, leave both variables unset. Do not use the standalone file URL.

The responsive UI has overview, Drug Pair Explorer, Interaction Results, 3D Molecular Lab, Explainability & Evidence, Research Assistant, Dataset & Model Insights, Prediction History, System Status, and About sections. Search reviewed names, IDs, aliases, or PubChem CIDs, select two different drugs, and the selected drug's available local formula, molecular weight, SMILES, PubChem source, structure status, ChEMBL target coverage, and DailyMed passage coverage load independently. Choose **Inspect pair evidence** to retrieve the pair-specific DDInter annotation and source records and to request actual inference from `POST /api/research/predict`. Predicted severity/probabilities are displayed separately from the DDInter annotation. Production `POST /api/predict` continues to return HTTP 503.

Name search is case-insensitive and trims whitespace. Exact canonical names and exact aliases rank ahead of partial synonym hits, so `Aspirin` resolves first to the reviewed `Acetylsalicylic acid` record (`DDInter20`). `Warfarin` resolves to `DDInter1951`. Warfarin has a validated 2D structure but no validated graph-matched 3D conformer in this corpus, so its 3D view stays unavailable and only the 2D fallback is shown. The app reads the existing local, provenance-backed catalog and does not perform live PubChem lookups or claim an external lookup failure means that a drug does not exist.

### 3D Molecular Lab

Choose a pair in the Explorer and open **3D Molecular Lab**. The **Use verified example** button selects Abacavir (`DDInter1`) and Leflunomide (`DDInter1033`); their source conformers were checked against reviewed molecular graphs. Each structure is shown independently, with rotate/zoom/pan and camera reset supported by the viewer. The combined scene places two independent molecules side by side for comparison; it is not a bound pose or interaction geometry. The viewer loads 3Dmol.js from its public CDN, so browser network access is required. Of 1,939 mapped drug IDs, 940 have verified 3D conformers. When a validated structure exists but no verified 3D coordinates do, the UI shows a clearly labeled 2D RDKit depiction. Missing structures remain unavailable. The viewer endpoints and JavaScript syntax are verified; browser/WebGL rendering still needs a check on the demonstration machine.

### AI Research Assistant

The assistant has a backend provider adapter and receives saved project evaluation/coverage/checkpoint facts plus documentation references. Optional selected-pair context adds actual retrieved pair annotation and source records. Responses are clearly labeled AI-generated educational content, not model predictions or medical advice. The assistant does not have a configured provider credential in the current environment, so it returns HTTP 503 until one is configured. To configure OpenAI, copy `.env.example` to `.env`, replace its placeholder locally, then launch the backend with the key loaded into its environment:

```bash
cd "/path/to/DDI-GUARD"
cp .env.example .env
# Edit .env and set a valid server-side key; never paste the key in the browser.
python3 -m uvicorn ddi_guard.api:app --host 127.0.0.1 --port 8000
```

The backend reads `DDI_GUARD_AI_API_KEY` (or `OPENAI_API_KEY`) and the provider/model/base URL from `.env` at startup; no `source .env` step is needed. Existing process environment variables take precedence. `DDI_GUARD_AI_PROVIDER` defaults to `openai`; `DDI_GUARD_AI_MODEL` defaults to `gpt-4.1-mini`. For an OpenAI-compatible provider, set provider to `openai_compatible` and set `DDI_GUARD_AI_BASE_URL` to HTTPS (HTTP is accepted only for localhost). Keys stay server-side. The app makes no claim that the assistant is live until the System Status page reports it configured and a real provider request succeeds. Restart the backend after changing `.env`.

The conversation selector/reset, retry, copy, and Markdown display are part of the UI. Conversation history is held in page memory and is not persisted. Do not submit patient-identifying or sensitive health information.

### Source-backed explainability and reports

Interaction Results and Explainability & Evidence show DDInter row references, ChEMBL 34 activity/target records, and DailyMed label passages only when verified records exist. ChEMBL and DailyMed evidence is drug-level context and does not prove pair-specific causality. Atom/bond attribution is not implemented. Dataset & Model Insights reads saved test artifacts and displays per-class metrics, confusion matrices, calibration, and paired bootstrap results. Prediction History remains empty because research requests are not logged; there is no PDF export implementation.

## Data and research artifacts

GitHub source publication intentionally excludes `data/raw/`, `data/processed/`, and generated artifact files (including model checkpoints and vector stores) because they are very large and may contain restricted source material. These files remain untouched in the local project. A fresh clone contains source code and documentation, but needs the project data and verified artifacts provisioned locally before research inference or saved evaluation can run. Human-readable audit and model documentation under `artifacts/` is retained.


Primary source/coverage audits:

- `artifacts/ddinter_ingestion_audit.json`
- `artifacts/pubchem_coverage_audit.json` and `artifacts/molecular_graph_audit.json`
- `artifacts/chembl_34_source_audit.json` and `artifacts/chembl_coverage_audit.json`
- `artifacts/dailymed_evidence_coverage_audit.json`
- `artifacts/multimodal_coverage_audit.json`
- `artifacts/split_manifest.json`

The selected pair-disjoint academic candidate checkpoint is:

```text
artifacts/multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt
SHA-256: 9259c34a964c48d30c849cb5480c8ac0a1453abf3b4936608c1a2435054d965c
```

Selection was at epoch 7 by validation macro-F1 (0.7605), with validation loss as tie-breaker. It matches the known-reviewed-drug/new-pair academic setting; cold-drug testing remains a separate stress test. Held-out test results were not used to choose it. Checkpoint/config/evaluation/source hashes and modality routing are listed in `artifacts/research_demo_checkpoint.json`. Research access is separately enabled after the documented technical checks; `model_approval.json` remains `approved: false`, with no production checkpoint binding.

### Reproduce saved research validation (no training)

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q ddi_guard tests scripts
python3 scripts/check_frontend_syntax.py
python3 scripts/paired_multimodal_validation.py
# Run the same complete test, syntax, and saved-evaluation checks and record artifacts/qa_status.json
python3 scripts/verify_project.py
```

The most recent full run recorded by `artifacts/qa_status.json` is the source for the exact test count/status. Rerun the verification script after any code change. Drug lookup/3D regression tests cover Warfarin and Aspirin ranking, whitespace/case handling, empty and unknown queries, API-base injection, local structure fallback, viewer-status uncertainty, and retained prediction HTTP 503 behavior.

### Data preparation and training commands

Training is **not required** to run the demo or reproduce evaluation reports. Do not rerun training just for demonstration. For a deliberate new experiment after verifying all required source manifests, use versioned run tags so prior outputs are not overwritten:

```bash
python3 -m ddi_guard.modality_coverage --project-root .
python3 -m ddi_guard.train_multimodal --project-root . --protocol pair_disjoint --modalities molecular,target,text --seed 42 --epochs 20 --batch-size 128 --patience 4 --run-tag new_versioned_run
```

The text vector manifest must reference real source passages, hashes, label-independent fit scope, and drug IDs. The trainer expects all selected modalities to be verified and uses explicit per-pair coverage masks; no missing modality is silently synthesized. Historical checkpoints and result folders must be preserved.

## Prediction and safety behavior

The academic demo endpoint is `POST /api/research/predict`. Example:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/research/predict \
  -H 'Content-Type: application/json' \
  -d '{"drug_a":"DDInter1","drug_b":"DDInter10"}'
```

The response contains the predicted class, model-generated class probabilities, model/checkpoint hash, validation-only calibration, per-drug modality coverage, and source provenance. Full molecular+target+text is selected when available; otherwise only the verified molecular+target, molecular-only, or target-only checkpoint is eligible. Text-only fallback is not enabled. Unknown, ambiguous, identical, or unsupported inputs return explicit errors. A source lookup remains separate and its DDInter label is never copied into the prediction.

Production `POST /api/predict` deliberately remains HTTP 503. The separate research policy was enabled only after hash checks, strict reload, saved-probability parity tests, deterministic actual inference, unsupported-input tests, and the full automated suite. This does not authorize production deployment or clinical use. Production/clinical use still needs independent external and clinical validation, improved cold-drug evidence, deployment security/monitoring/change control, and applicable institutional/regulatory review.

## Known limitations

- In the mapped 1,939-drug catalog, model source loaders have 1,472 molecular, 1,166 target, and 652 text feature records; 517 IDs have all three. The labeled-pair universe has 1,902 drugs, of which 1,436 have molecular, 1,134 target, and 634 DailyMed coverage. These are different denominators and do not mean every pair is supported.
- Pair-disjoint full model uses 8,497/1,798/1,802 common-support pairs, while cold-drug full model uses 5,649/356/253. The model is not trained on every DDInter row.
- Cold-drug test has 14 Minor, 206 Moderate, and 33 Major examples; cold-drug macro-F1 is 0.3664. Do not infer robust new-drug performance from this small support.
- The full model does not outperform text-containing alternatives on the pair-disjoint matched test support. See the paired confidence intervals in the validation report.
- DailyMed passages are general label evidence, and ChEMBL features are bioactivity relations; neither alone establishes pair-specific clinical interaction risk.
- No independent clinical validation or regulatory approval exists. No production model is approved.

## Troubleshooting

- If imports fail, activate a Python environment with compatible PyTorch, PyTorch Geometric, RDKit, and API dependencies. The verified interpreter is shown above.
- If opening a local `file://` copy shows missing data, start Uvicorn and use <http://127.0.0.1:8000>; the frontend needs the API origin.
- If the assistant says not configured, confirm the ignored project `.env` contains `DDI_GUARD_AI_API_KEY` and restart Uvicorn; the backend loads that file automatically. Check System Status afterward. A configured key can still be rejected or rate-limited by its provider, which will produce a separate error.
- If the 3D viewer cannot load, verify network access to the 3Dmol.js CDN. For a selected drug with no conformer, the UI reports that status and may show only the clearly labeled 2D fallback.
- If research inference returns HTTP 422, the pair lacks a verified checkpoint with shared real input features; the endpoint does not fabricate missing data. Production `/api/predict` returning HTTP 503 is expected.
- DailyMed errors include rate limits; retry only failed IDs with conservative backoff and append-only staged artifacts. Preserve original corpus/manifests.
- The workspace has no `.git` metadata, so use file hashes/artifact manifests for this audit; Git history/status is unavailable.
