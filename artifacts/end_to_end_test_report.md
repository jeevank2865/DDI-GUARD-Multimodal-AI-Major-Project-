# DDI-GUARD end-to-end test report

**Date:** 2026-10-21  
**Environment:** macOS workspace; `/opt/miniconda3/bin/python3` (Python 3.13.12), Node 20.20.2; FastAPI/Uvicorn; project dependency environment includes RDKit and PyTorch.  
**Training:** not run. No checkpoint, source corpus, feature manifest, split, calibration, or historical experiment output was intentionally modified.  
**Approval:** unchanged; production/research-demo HTTP prediction is disabled.

## Summary by status

| Area | Status | Evidence / limitation |
|---|---|---|
| Uvicorn startup and local HTTP API | **PASS** | Documented command started on loopback; `GET /` and audited routes returned expected codes. Temporary server stopped after tests. |
| Python tests | **PASS** | Latest `scripts/verify_project.py`: 71 tests passed, exit 0. Detailed command output tail in `artifacts/qa_status.json`. |
| Python compilation | **PASS** | `python3 -m compileall -q ddi_guard tests scripts`. |
| Inline JavaScript syntax | **PASS** | `scripts/check_frontend_syntax.py`: JavaScript syntax OK. This does not execute UI events. |
| Saved evaluation integrity and reproducibility | **PASS** | `scripts/paired_multimodal_validation.py` completed; supports, results, and paired CIs reproduced. Pair-disjoint n=1,802; cold-drug n=253 including 14 Minor. |
| Reviewed drug and source APIs | **PASS** | Search/drug/interaction routes return real local catalog/DDInter/ChEMBL/DailyMed records; unknown ID returns 404. |
| Structure APIs | **PASS** | `DDInter1` and `DDInter1033` return real verified conformers; `DDInter1951` returns `available:false` and its validated 2D depiction route returns SVG. Coverage remains 940/1,939. |
| Prediction route | **BLOCKED by explicit policy/technical gate** | Valid known pair returns HTTP 503 with failed `inference_preprocessing_matches_checkpoint`; unknown pair returns 404. No bypass attempted. |
| Real AI provider | **BLOCKED by missing credential** | No key configured. Assistant request returns HTTP 503; no live upstream request was sent. Provider adapter behavior has mocked unit coverage only. |
| Browser navigation/visual polish/console/network | **NOT TESTED / tool blocked** | CUA browser inventory reports local browser use disabled by Work settings; no Playwright/Selenium package or project frontend build is installed. Browser interaction cannot be truthfully marked passed. |
| 3D WebGL and viewer controls | **NOT TESTED / browser blocked** | Verified API coordinates, source match, generated SDF path/code, syntax, and cleanup implementation. Rotation, pan, zoom, style, labels, reset, resize, touch, canvas appearance, and memory behavior need actual browser verification. |
| Responsive viewports (1440×900, 1280×800, 768×1024, 390×844) | **NOT TESTED / browser blocked** | Responsive breakpoints/reduced-motion rules are present in CSS, but no viewport screenshots or overflow measurements were obtained. |
| Accessibility / screen reader | **PARTIAL STATIC REVIEW** | Input labels, focus-visible CSS, live status regions, autocomplete relationships, menu expanded state, label pressed state, and reduced motion exist. Keyboard/screen-reader behavior and contrast audit not run in a browser. |
| Live PubChem/ChEMBL/DailyMed external access | **NOT TESTED** | No bulk retrieval was initiated. Rendered source records are local, hash/provenance checked artifacts; outbound external link targets were not clicked during this audit. |

## Commands run

```bash
/opt/miniconda3/bin/python3 -m compileall -q ddi_guard tests scripts
/opt/miniconda3/bin/python3 scripts/check_frontend_syntax.py
/opt/miniconda3/bin/python3 -m unittest tests.test_api tests.test_research_inference tests.test_assistant -v
/opt/miniconda3/bin/python3 scripts/verify_project.py
```

The prior UI-audit verification snapshot recorded 66 tests. The latest drug-search/3D-fix snapshot is `artifacts/qa_status.json`: status `passed`, 71 tests, all four checks exit 0. The prior focused API/inference/assistant run passed 21 tests after the state, accessibility, and request-race fixes.

## Live HTTP retest

The backend was started as a temporary Uvicorn child process at `127.0.0.1:8765` and stopped in a `finally` cleanup. No persistent server remains from the audit.

| Request | Actual response | Check |
|---|---:|---|
| `GET /` | 200 HTML | Startup and frontend file delivery pass. |
| `GET /api/status` | 200 | Audit status available. |
| `GET /api/audit` | 200 | Audit route available. |
| `GET /api/evaluation` | 200 | Saved evaluation available. |
| `GET /api/research-results` | 200 | Saved paired research artifacts available. |
| `GET /api/system-status` | 200 | Backend, checkpoint integrity, approval, and configuration snapshot available. |
| `GET /api/drugs/search?q=Abacavir` | 200 | Reviewed identity search works. |
| `GET /api/drugs/DDInter1/structure-3d` | 200, `available=true` | Real conformer response. |
| `GET /api/drugs/DDInter1033/structure-3d` | 200, `available=true` | Real conformer response. |
| `GET /api/drugs/DDInter1951/structure-3d` | 200, `available=false` | Missing 3D is explicitly represented. |
| `GET /api/drugs/DDInter1951/structure-2d.svg` | 200 SVG | 2D fallback data served. |
| `GET /api/interactions/lookup?drug_a_id=DDInter1&drug_b_id=DDInter1033` | 200 | Genuine DDInter record labelled Major; model prediction field says unavailable. |
| Lookup with unknown ID | 404 | Unknown drug rejected. |
| `POST /api/predict` known pair | 503 | Explicit gate; preprocessing check named in response. |
| `POST /api/predict` unknown drug | 404 | Unknown identity rejected before gate response. |
| `POST /api/assistant/chat` without key | 503 | Missing real provider configuration is disclosed. |
| Assistant with unknown selected pair ID | 404 | Pair context identity validated. |

The Major value in the example above is a dataset annotation, not a model prediction. The pair is chosen because both real structures have graph-matched conformers.

## Repair and regression status

- Fixed stale pair/evidence display on drug change, remove, swap, and clear; clear also empties both search fields and molecular-lab state.
- Added per-input sequence checks and selected-pair identity guards to prevent late search/lookup/prediction responses from replacing newer UI state or leaving the action disabled.
- Added a molecular request generation guard, WebGL context release when leaving the lab, and object URL revocation for 2D fallback images.
- Prevented duplicate chat sends while a provider request is active; retry is tied to the original conversation turn; clipboard failure reports “Copy unavailable” rather than false success.
- Added autocomplete ARIA relationships, ArrowDown/Escape behavior, `aria-pressed` for atom labels, explicit Jmol element color key for individual views, and element colors for line style.
- Added frontend source guard test `test_frontend_contains_state_and_accessibility_guards`; 66-test suite passes.
- Browser-level retest for these controls remains **BLOCKED**, so these changes are not represented as manually verified.

## Drug search / structure API follow-up

The subsequent route and UI changes are documented in [`drug_lookup_and_3d_fix_report.md`](drug_lookup_and_3d_fix_report.md). Warfarin and Aspirin queries, drug details, explicit Warfarin 3D-unavailable/2D fallback, Aspirin 3D availability, CORS allow-list behavior, and retained prediction 503 were checked through FastAPI's in-process HTTP client and a temporary Uvicorn loopback smoke test. The temporary process stopped after the requests. The complete suite now passes 71 tests; no live browser/WebGL result is claimed.

## Model, provenance, and safety check

- Candidate checkpoint remains `artifacts/multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt`, SHA-256 `9259c34a964c48d30c849cb5480c8ac0a1453abf3b4936608c1a2435054d965c`.
- Recorded checkpoint, configuration, evaluation, pair coverage, text manifest, dataset, split, target graph, graph manifest, and molecular source hashes verified through system status.
- `artifacts/model_approval.json` was not changed. `approved=false`; `research_demo_policy.allowed=false`.
- Actual offline research-inference tests load the checkpoint and verify deterministic, normalized probabilities for known IDs with all three real modalities. That adapter is not exposed by `/api/predict`.
- DailyMed coverage remains partial (652/1,939 mapped drugs; 879 passages; prior audit lists 403 retrieval errors). No external-source retry or data overwrite occurred.
- No secret values were printed. No `.env` credential is configured; `.env.example` contains a placeholder only.

## Remaining prerequisites for a complete UI acceptance

1. In a normal Chrome/Brave desktop browser, serve the application with the README command and exercise every control listed in `artifacts/ui_functionality_audit.md`.
2. Check 1440×900, 1280×800, 768×1024, and 390×844; inspect console/network failures, keyboard access, focus, and actual screen-reader names.
3. With network available, verify 3Dmol.js CDN load and WebGL rendering/rotation/zoom/pan, styles, labels, reset, responsive resize, and repeated exit/re-entry. Confirm combined scene shows distinct molecules only.
4. If a real AI assistant is required for the demo, configure a backend-only provider key, perform one genuine provider request, and verify provider/model and errors without disclosing the secret. Do not use mock output as live acceptance.
5. Keep prediction disabled until the preprocessing/checkpoint issue is resolved under the project’s actual policy and all required approval checks are reviewed.
