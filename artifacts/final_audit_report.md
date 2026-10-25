# DDI-GUARD final project audit

**Audit date:** 2026-10-21  
**Workspace:** `<PROJECT_ROOT>`  
**Scope:** Academic B.Tech project demonstration and reproducible research only. This system is not clinically validated, approved, or suitable for diagnosis or treatment decisions.

This is the current evidence-based status. It supersedes older status language retained in historical experiment artifacts. No training was run during this audit. Existing checkpoints, source data, evaluation outputs, and manifests were preserved. This workspace has no Git metadata; `git status` and a Git diff are unavailable.

## Latest research inference update (2026-10-21)

The separate local academic research route `POST /api/research/predict` is now enabled after hash verification, strict reload, saved-probability parity for all four routed checkpoints, deterministic inference, unsupported-input checks, and the full 73-test suite. The Drug Pair Explorer calls that route and shows actual outputs/errors. This supersedes the older “prediction unavailable” statements below. Production `POST /api/predict` remains HTTP 503, `approved=false`, and is not configured; clinical/regulatory approval is not claimed. See [`research_prediction_enablement_report.md`](research_prediction_enablement_report.md) for detailed evidence, exact checkpoint paths, sample probabilities, coverage, and remaining issues.

## Final status

| Area | Verdict | Verified state |
|---|---|---|
| Dataset preparation | **VERIFIED COMPLETE** | DDInter source accounting, fixed split, PubChem review, ChEMBL 34 import, and DailyMed partial corpus/vectorization are present and audited. |
| Molecular modality | **PARTIALLY COMPLETE** | 1,436/1,902 labeled-pair drugs have validated graph features; 466 are unresolved or unusable. |
| ChEMBL target modality | **VERIFIED COMPLETE** | Verified ChEMBL 34 graph: 105,982 filtered source edges, 1,166 drugs, 2,699 targets, 745 relation types; 1,134 IDs overlap the labeled-pair universe. These are bioactivity relations, not clinical DDI evidence. |
| DailyMed text modality | **PARTIALLY COMPLETE** | 652/1,939 mapped drugs and 879 passages; 403 fetch errors (393 HTTP 429). Vectors, provenance, alignment, and train-only fit manifests are verified. |
| Full three-modality training | **VERIFIED COMPLETE** | Pair-disjoint and cold-drug runs completed on the all-three-modality common supports; checkpoints and evaluations are present and hash-checked. This is a covered subset, not every DDInter record. |
| Evaluation integrity | **VERIFIED COMPLETE WITH LIMITATIONS** | Fixed pair-disjoint and cold-drug splits, matched-support ablations, validation-only calibration, paired bootstrap, exact predictions, and hashes are recorded. Cold test has only 14 Minor pairs. |
| Prediction pipeline | **RESEARCH ROUTE VERIFIED; PRODUCTION BLOCKED** | `/api/research/predict` serves real inference for supported feature combinations using four verified checkpoints; `/api/predict` remains 503. |
| Frontend integration | **API-INTEGRATED; BROWSER REVIEW UNAVAILABLE** | Drug Pair Explorer displays real model output and API errors separately from dataset annotation. FastAPI TestClient and loopback HTTP checks passed; browser visual review remains unavailable. |
| 3D Molecular Lab | **STRUCTURE API VERIFIED; RENDER NOT BROWSER-VERIFIED** | 940/1,939 mapped drugs have graph-matched conformers. The verified example pair Abacavir (`DDInter1`) and Leflunomide (`DDInter1033`) has a conformer for each; comparison models remain separate. Actual WebGL rendering still needs a browser check; 3Dmol.js requires CDN access. |
| AI Research Assistant | **IMPLEMENTED; LIVE PROVIDER BLOCKED** | Backend adapter and artifact-backed project context are implemented. No provider key is configured locally; real chat returns HTTP 503 until a server-side key is provided. Mocked provider tests are not evidence of live connectivity. |
| Test suite | **VERIFIED COMPLETE FOR CURRENT VERSION** | `artifacts/qa_status.json` records the full 73-test suite, Python compile, frontend syntax, and saved paired-evaluation checks. |
| Final-year project demonstration readiness | **RESEARCH DEMO READY FOR SUPPORTED PAIRS** | Local research predictions are available where checkpoint features exist. Live AI still requires a configured provider. Production/clinical prediction remains blocked. |

## Data and provenance

### DDInter and split construction

Independent ingestion accounts for 222,383 raw rows, 160,235 canonical pairs including unknown-only pairs, and 130,422 labeled pairs: 6,833 Minor, 96,675 Moderate, and 26,914 Major. It recorded 29,813 unknown-only pairs, collapsed 44,779 duplicate labeled records, and found no self-pairs or conflicting severity pairs. See [`ddinter_ingestion_audit.json`](ddinter_ingestion_audit.json), [`ddi_pair_manifest.csv`](../data/processed/ddi_pair_manifest.csv), and [`split_manifest.json`](split_manifest.json).

The seed-42 pair-disjoint split has 91,295 / 19,563 / 19,564 train/validation/test pairs. The cold-drug split has 63,447 / 3,093 / 2,804; its partition drug sets are disjoint and cross-partition pairs are excluded. Split manifest SHA-256 is `b49366e1a39784f9f79c9652598357e531cf14f4f4b349d4feeaab8922c5773a`.

### Modality coverage

Coverage below is unique drug IDs in the 1,902-drug labeled-pair universe unless stated. A usable pair requires the modality for both drugs. It does not mean the data are available for all pairs.

| Source/modality | Source records / validated units | Unique covered drugs | Usable pair support | Missing / unresolved | Evidence paths |
|---|---:|---:|---:|---:|---|
| DDInter labels | 222,383 source rows; 130,422 unique labeled pairs | 1,902 labeled-drug IDs | 130,422 source labeled pairs | 29,813 canonical unknown-only pairs are not labeled examples | `data/processed/ddi_pairs_full.json`, `data/processed/ddi_pair_manifest.csv`, `artifacts/ddinter_ingestion_audit.json` |
| PubChem molecular | 1,472 validated unique-name structures from 1,939 mapped IDs | 1,436 among labeled-pair drugs | Pair support is split-dependent; full tri-modal support below | 466 labeled-pair drug IDs lack accepted structures | `data/processed/pubchem_compounds.json`, `data/processed/molecular_graph_manifest.json`, `artifacts/pubchem_coverage_audit.json`, `artifacts/molecular_graph_audit.json` |
| ChEMBL 34 targets | 1,050,131 activity manifest rows; 105,982 quality-filtered source edges; 2,699 targets | 1,134 among labeled-pair drugs (1,166 graph drugs total) | Pair support is split-dependent; full tri-modal support below | 768 labeled-pair drug IDs lack target edges | `data/processed/chembl_bioactivity_manifest.csv`, `data/processed/chembl_drug_target_edges.csv`, `data/processed/chembl_34_target_graph/target_graph.pt`, `artifacts/chembl_coverage_audit.json` |
| DailyMed text | 879 provenance-backed passages; 652 vectorized drugs | 634 among labeled-pair drugs (652 mapped IDs overall) | Pair support is split-dependent; full tri-modal support below | 1,268 labeled-pair drug IDs lack text; 403 source requests failed | `data/processed/dailymed_drug_evidence.jsonl`, `data/processed/text_evidence_embeddings_manifest_pair_disjoint_trainfit_v1.csv`, `artifacts/dailymed_evidence_coverage_audit.json` |
| All three modalities | Real features required for both drugs | Not an additional unique-drug statistic | Pair-disjoint 8,497 / 1,798 / 1,802; cold-drug 5,649 / 356 / 253 train/validation/test | All other split pairs excluded from these common-support runs | Each full model's `pair_coverage_manifest.json` under the artifact directories below |

ChEMBL 34 archive verification: 4,859,330,198 bytes and SHA-256 `7d40409e4f31440674a79e26db628015e82b85b056a7d8234efe9c95b6cbd8e7`. The extracted database passed SQLite `quick_check` and release identity validation before import. The graph preserves source activity/assay/target identifiers, relation/evidence types, and provenance. DailyMed passages preserve label set/version/date, URLs, document hashes, retrieval and mapping provenance. DailyMed text is general drug-label evidence, not a pair-specific DDI mechanism.

The DailyMed rate-limit audit found 393 HTTP 429s, six read timeouts, two TLS handshake timeouts, and two connection resets. No retry was issued during the audit. A future retry should target failed IDs only, honor `Retry-After`, otherwise use full-jitter exponential backoff (30 seconds initial, 10-minute cap, maximum five retries), concurrency one and no more than one request per second; stop after three consecutive 429s. Preserve original corpus/manifests and write append-only retry shards before hash-checked staged merging. These are conservative recommendations, not a claimed NLM quota.

## Training and experiment results

The full model architecture consumes molecular graph, ChEMBL target graph, and DailyMed text features. Completed text-enabled matched-support model sets include text-only, molecular+text, target+text, molecular+target control, and full molecular+target+text for both protocols. Every run uses its own split-appropriate training support; pair-disjoint and cold-drug numbers must not be conflated. Full three-modality training did not use all DDInter records: it used only the listed common supports with all required modalities. Missing features are masked or cause support exclusion; there are no fabricated feature values.

| Protocol/model | Train / validation / test | Best epoch; validation macro-F1 | Test macro-F1 | Test macro OVR AUROC | Brier / NLL / ECE |
|---|---:|---:|---:|---:|---:|
| Pair-disjoint molecular + target | 65,435 / 13,869 / 14,057 | 10; 0.6826 | 0.6792 | 0.9089 | 0.2988 / 0.4852 / 0.0151 |
| Pair-disjoint text | 8,497 / 1,798 / 1,802 | 10; 0.7602 | 0.7708 | 0.9333 | 0.1858 / 0.3361 / 0.0198 |
| Pair-disjoint molecular + text | 8,497 / 1,798 / 1,802 | 10; 0.7707 | 0.7836 | 0.9323 | 0.1848 / 0.3377 / 0.0158 |
| Pair-disjoint target + text | 8,497 / 1,798 / 1,802 | 9; 0.7581 | 0.7743 | 0.9379 | 0.1837 / 0.3298 / 0.0188 |
| Pair-disjoint molecular + target, matched control | 8,497 / 1,798 / 1,802 | 6; 0.6021 | 0.6130 | 0.8527 | 0.3168 / 0.5333 / 0.0280 |
| Pair-disjoint full three-modality | 8,497 / 1,798 / 1,802 | 7; 0.7605 | 0.7473 | 0.9280 | 0.2177 / 0.3802 / 0.0163 |
| Cold-drug molecular + target | 45,305 / 2,255 / 2,003 | 4; 0.3918 | 0.3680 | 0.5166 | 0.5185 / 0.8711 / 0.0863 |
| Cold-drug text | 5,649 / 356 / 253 | 7; 0.4579 | 0.3743 | 0.5403 | 0.4080 / 0.7254 / 0.1185 |
| Cold-drug molecular + text | 5,649 / 356 / 253 | 7; 0.3808 | 0.3986 | 0.6097 | 0.3675 / 0.6472 / 0.0579 |
| Cold-drug target + text | 5,649 / 356 / 253 | 5; 0.4088 | 0.3556 | 0.5431 | 0.3838 / 0.6728 / 0.0832 |
| Cold-drug molecular + target, matched control | 5,649 / 356 / 253 | 7; 0.3042 | 0.2821 | 0.4286 | 0.5803 / 0.9515 / 0.0674 |
| Cold-drug full three-modality | 5,649 / 356 / 253 | 10; 0.3989 | 0.3664 | 0.5409 | 0.3624 / 0.6550 / 0.0708 |

These metrics are from the saved calibrated evaluations. Macro-F1 is not accuracy. Checkpoint selection and temperature scaling use validation partitions only; test metrics are held-out descriptions. Full per-class precision/recall/F1/support, confusion matrices, calibration metrics, paired comparisons, pair IDs, manifests, and artifact integrity results are in [`paired_multimodal_validation_v1/paired_multimodal_validation.md`](paired_multimodal_validation_v1/paired_multimodal_validation.md) and its JSON. Test support and class imbalance limit generalization claims.

### Other completed single-modality runs and fingerprint baselines

These runs use their own modality-covered supports and must not be compared directly with the smaller common-support multimodal results above. Earlier artifacts without `no_placeholder_v1` or `trainfit_v1` tags are preserved but are not the corrected canonical runs.

| Protocol/model | Covered train / validation / test | Best epoch; validation macro-F1 | Test macro-F1 | Evaluation artifact |
|---|---:|---:|---:|---|
| Pair-disjoint molecular only, no-placeholder | 64,667 / 13,719 / 13,893 | 18; 0.7254 | 0.7208 | `artifacts/molecular_pair_disjoint_no_placeholder_v1/evaluation.json` |
| Cold-drug molecular only, no-placeholder | 44,913 / 2,207 / 1,983 | 5; 0.4604 | 0.4248 | `artifacts/molecular_cold_drug_no_placeholder_v1/evaluation.json` |
| Pair-disjoint target only | 43,633 / 9,187 / 9,265 | 10; 0.6449 | 0.6274 | `artifacts/multimodal_pair_disjoint_target_diagonal_relation_v1/evaluation.json` |
| Cold-drug target only | 30,580 / 1,557 / 1,175 | 7; 0.3459 | 0.3660 | `artifacts/multimodal_cold_drug_target_chembl34_diagonal_v1/evaluation.json` |
| Pair-disjoint Morgan fingerprint RF | 64,667 / 13,719 / 13,893 | validation 0.7980 | 0.7966 | `artifacts/fingerprint_rf_pair_disjoint_prediction_export_v1/evaluation.json` |
| Cold-drug Morgan fingerprint RF | 44,913 / 2,207 / 1,983 | validation 0.4074 | 0.3528 | `artifacts/fingerprint_rf_cold_drug_prediction_export_v1/evaluation.json` |

The canonical selected checkpoint and all model variants use the saved model configurations, training histories, test-prediction artifacts, and coverage manifests in their listed run folders. Pair-disjoint and cold-drug results remain separate because their supports and generalization questions differ.

### Paired uncertainty and cold-drug details

The paired bootstrap uses 10,000 class-stratified resamples of identical test pair rows. Full-model minus comparator macro-F1 (point delta; 95% CI):

| Protocol | Comparator | Delta | 95% CI |
|---|---|---:|---:|
| Pair-disjoint | Molecular + text | -0.0363 | [-0.0602, -0.0121] |
| Pair-disjoint | Target + text | -0.0270 | [-0.0495, -0.0041] |
| Pair-disjoint | Text | -0.0235 | [-0.0465, +0.0010] |
| Pair-disjoint | Molecular + target | +0.1343 | [+0.1035, +0.1662] |
| Cold-drug | Molecular + text | -0.0322 | [-0.1000, +0.0231] |
| Cold-drug | Target + text | +0.0107 | [-0.0058, +0.0298] |
| Cold-drug | Text | -0.0080 | [-0.0429, +0.0181] |
| Cold-drug | Molecular + target | +0.0843 | [-0.0013, +0.1750] |

Cold-drug test support is **253 pairs: 14 Minor, 206 Moderate, and 33 Major**. The full model correctly classified 2/14 Minor and 1/33 Major; its per-class calibrated F1 values are 0.174 / 0.874 / 0.051. All four cold-drug paired intervals include zero. The 14 Minor examples make class-level estimates unstable; no held-out labels or predictions were changed or used for tuning.

### Selected academic candidate

Candidate: [`multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt`](multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt), SHA-256 `9259c34a964c48d30c849cb5480c8ac0a1453abf3b4936608c1a2435054d965c`. It is suitable for offline reproducibility and known-drug/new-pair academic analysis because it consumes all three real modalities, has verified config/source/split/feature artifacts, and was selected by validation macro-F1 then validation loss at epoch 7 (validation macro-F1 0.7605). The held-out test score did not select it. It is not approved and is not connected to `/api/predict`.

Its config, evaluation, coverage manifest, pair-disjoint train-fit text manifest, target graph and molecular source fingerprints are catalogued in [`research_demo_checkpoint.json`](research_demo_checkpoint.json). It is used when both reviewed drug IDs have all three modalities. Verified M+T, molecular-only, or target-only checkpoints are used only when their trained inputs are present; otherwise the API returns an explicit unsupported error.

## API, frontend, and approval blockers

The Drug Pair Explorer provides reviewed drug search, pair lookup, DDInter annotations, and available ChEMBL/DailyMed evidence with provenance. It reports actual `/api/predict` failures and does not invent scores or evidence. API status and pair endpoints are backed by current audit artifacts.

`POST /api/research/predict` is separately policy-gated and available for local academic use. `artifacts/model_approval.json` records `research_demo_policy.allowed: true` after verification while keeping `approved: false`, no production checkpoint path, and production `api_prediction_enabled: false`. `POST /api/predict` remains **HTTP 503**. The weak cold-drug result (including n=14 Minor), absent independent/clinical validation, and absent production security, monitoring, and change-control review remain blockers. No clinical validity or regulatory approval is claimed.

The single-page interface is served by FastAPI at `/`; it is not intended to be opened directly as a local `file://` document. Drug lookup, dataset annotation, source evidence, research metrics, status, structure, and assistant functions use actual backend routes/artifacts. A verified example pair was checked through the API: Abacavir (`DDInter1`) + Leflunomide (`DDInter1033`) has a real DDInter record and two available graph-matched PubChem conformers. The combined view offsets these separate structures for comparison only. The 3D viewer library loads from its CDN, so an offline browser can fail to render even though the structure endpoint has data. Structure fallback and unavailable states are explicit. Browser interaction could not be reviewed because browser use is disabled in this Work session; JS syntax/API integration are verified, not live WebGL presentation.

The Research Assistant has a server-side provider adapter and its default UI context contains verified saved evaluation, coverage, checkpoint, policy, and documentation facts; selecting the pair-context option adds retrieved pair records. The current environment has **no configured provider key**, so a real chat request returns HTTP 503. Provider-adapter tests use a mocked upstream and do not establish live service connectivity. Setup is documented in `README.md` and `.env.example`.

The Explainability page presents drug-level source records and explicitly states that atom/bond attribution and pair-specific causal mechanism validation are not implemented. Prediction History remains empty because the approved prediction service is disabled; PDF export is not implemented.

The cold-drug evaluation JSON points to a generic text-manifest path even though its recorded hash matches the cold-specific train-fit manifest. This is documented and independently checked; historical evaluation artifacts were not edited. Pair-disjoint manifest paths and hashes agree.

## Verification performed in this audit

Interpreter: `/opt/miniconda3/bin/python3` (project's verified Mac environment).

| Command | Result |
|---|---|
| `/opt/miniconda3/bin/python3 scripts/verify_project.py` | Exit 0 on 2026-10-21; all checks passed, 73 tests. Full command outputs/status are in `artifacts/qa_status.json`. |
| `/opt/miniconda3/bin/python3 -m compileall -q ddi_guard tests scripts` | Exit 0. |
| `/opt/miniconda3/bin/python3 scripts/check_frontend_syntax.py` | Exit 0; inline JavaScript syntax OK. |
| `/opt/miniconda3/bin/python3 scripts/paired_multimodal_validation.py` | Exit 0; supports, comparisons, and confidence intervals reproduced. |
| API smoke checks: research prediction and production guard | Loopback `POST /api/research/predict` returned HTTP 200 with actual checkpoint output; `POST /api/predict` returned HTTP 503. FastAPI tests also verified unknown/unsupported inputs. |
| Browser UI and WebGL rendering | Not observed; browser inventory reports browser use disabled in Work settings. Verify the 3D lab in Chrome/Brave on the demonstration machine with network access to the 3Dmol.js CDN. |
| Approval status inspection | Production `approved=false`; research endpoint enabled only for academic demo after preprocessing parity and full test suite passed. |

## UI functionality and end-to-end QA refresh (2026-10-21)

The separate [`ui_functionality_audit.md`](ui_functionality_audit.md) inventories the discovered routes, page sections, and visible controls, with statuses distinguishing exercised API behavior from unobserved browser behavior. The [`end_to_end_test_report.md`](end_to_end_test_report.md) records the latest loopback checks and known limitations. Fixes this pass address stale pair/evidence state, autocomplete and pair request races, duplicate chat submits, clipboard failure messaging, autocomplete accessibility attributes, WebGL cleanup, SVG object-URL cleanup, and search debounce. The full suite now records 66 passing tests; the regression assertion checks source-level frontend guards rather than browser events.

Browser interaction, responsive layouts, accessibility with assistive technology, actual 3Dmol.js/WebGL rendering, and browser console/network inspection remain **BLOCKED / NOT TESTED** because this Work session reports local browser use disabled. The assistant adapter has mocked-provider tests, but no configured credential exists and no genuine provider call was made. Prediction remains HTTP 503: `artifacts/model_approval.json` is still `approved=false`, `research_demo_policy.allowed=false`, and `inference_preprocessing_matches_checkpoint` remains unresolved. No model, source-data, or historical experiment artifacts were changed.

## Drug lookup and molecular structure update (2026-10-21)

See [`drug_lookup_and_3d_fix_report.md`](drug_lookup_and_3d_fix_report.md) for root-cause evidence and route-level results. The central API client now detects `file://` usage and provides the correct Uvicorn URL, and search prioritizes exact aliases. Warfarin resolves to `DDInter1951` with a 2D fallback only; Aspirin resolves first to Acetylsalicylic acid (`DDInter20`) with a verified 3D conformer. The newest full suite passes 71 tests, and API routes passed a temporary Uvicorn loopback smoke test. Browser rendering remains unverified.

The tests cover checkpoint hash/config loading, deterministic offline inference, normalized probabilities, invalid and unknown inputs, missing modalities, actual frontend/API handling, evidence provenance, and fail-closed API behavior. They do not constitute external validation or production approval.

## Reproducible Mac commands

```bash
cd "<PROJECT_ROOT>"
# Start the local audit console and pair explorer
/opt/miniconda3/bin/python3 -m uvicorn ddi_guard.api:app --host 127.0.0.1 --port 8000
# In another terminal: open http://127.0.0.1:8000

# Verify code and existing research artifacts without training
/opt/miniconda3/bin/python3 scripts/verify_project.py
```

Fresh environment setup can use `/opt/miniconda3/bin/python3 -m pip install -e '.[api]'` after confirming compatible PyTorch, PyTorch Geometric, and RDKit wheels for the Mac architecture. The exact installed dependency set used for current verification is environment-specific; use the verified interpreter above on this machine.

## Unresolved issues and next steps

1. For the academic demonstration, use the separate research route only for supported pairs and clearly state the model is not clinically validated.
2. Production deployment remains blocked pending independent validation, security/monitoring/change control, and the applicable institutional/regulatory review. Do not set production approval to true to bypass these steps.
3. Continue conservative DailyMed recovery only as append-only verified shards; do not overwrite the existing corpus or manifests.
4. Treat cold-drug generalization as unresolved; obtain independent data and sufficient class support before stronger claims. Do not tune on current test examples.
5. Keep the production/clinical gate closed until its explicit safeguards and independent review criteria are satisfied.
