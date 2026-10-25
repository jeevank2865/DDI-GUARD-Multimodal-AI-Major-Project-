# DDI-GUARD UI functionality audit

**Audit opened:** 2026-10-21  
**Workspace:** `<PROJECT_ROOT>`  
**Purpose:** Functional QA for an academic research/demo application. This audit does not authorize model predictions or claim clinical validity.

## Baseline and method

- Frontend: one static HTML file with inline CSS/JavaScript (`ddi_guard/static/index.html`); hash-based in-page navigation, no frontend build framework or separate frontend server.
- Backend: FastAPI in `ddi_guard/api.py`, served with Uvicorn. The documented Uvicorn command started successfully on loopback and `GET /` returned 200. No startup exception was observed.
- Local dependency inventory: Python `playwright`, Selenium, and pytest are absent; no `playwright` command was found. Node/npm are installed, but no browser automation package/build system is declared by this project.
- CUA inventory during this audit: Chrome and Brave apps exist, but no browser tabs/providers are exposed; Work settings report “Local browser use is disabled in Work settings.” Browser input, console, screenshots, responsive viewport checks, and actual WebGL interaction are therefore **BLOCKED**, not passed. No new browser-testing dependency was installed.
- Backend route behavior below was exercised through a temporary Uvicorn loopback process. All listed `GET`/`POST` route status codes are observed, not inferred from route declarations. That process was stopped after checks.
- The no-key assistant failure is observed; no real provider credential is configured, so sending a request to an AI provider was not attempted. Mocked provider coverage in unit tests is kept distinct from a live provider test.
- Browser-control inventory behavior is based on source inspection plus backend response tests. The exact browser interaction outcomes remain **NOT TESTED**. Repairs listed below have syntax/backend regression checks only until browser access is available.

## Routes and integration baseline

| Route | Observed baseline | Notes |
|---|---:|---|
| `GET /` | 200 | Static application delivered. |
| `GET /api/status` | 200 | Audit/dataset snapshot. |
| `GET /api/audit` | 200 | Audit snapshot route. |
| `GET /api/evaluation` | 200 | Saved evaluation artifacts. |
| `GET /api/research-results` | 200 | Saved matched evaluation data. |
| `GET /api/system-status` | 200 | Backend, artifact, approval, assistant status. |
| `GET /api/drugs/search?q=Abacavir` | 200 | Reviewed-catalog result. |
| `GET /api/drugs/DDInter1` | 200 | Reviewed drug record. |
| `GET /api/drugs/DDInter1/structure-3d` | 200 | Verified-conformer response available. |
| `GET /api/drugs/DDInter1951/structure-3d` | 200 | Endpoint can return 200 with `available:false`; an HTTP success is not proof coordinates exist. |
| `GET /api/drugs/DDInter1951/structure-2d.svg` | 200 | Validated structure SVG fallback. |
| `GET /api/interactions/lookup` for `DDInter1` + `DDInter1033` | 200 | Real DDInter annotation and available evidence. |
| `GET /api/interactions/lookup` with unknown ID | 404 | Unknown identity rejected. |
| `POST /api/predict` with known pair | 503 | Intentional approval gate; `inference_preprocessing_matches_checkpoint` remains false. |
| `POST /api/predict` with unknown drug | 404 | Identity check precedes prediction gate. |
| `POST /api/assistant/chat` without key | 503 | `DDI_GUARD_AI_API_KEY` / `OPENAI_API_KEY` required. |
| `POST /api/assistant/chat` with unknown selected ID | 404 | Invalid pair context rejected. |

The model approval file remains `approved: false`, `research_demo_policy.allowed: false`; these values were not changed. The full three-modality candidate and historical model artifacts were preserved.

## UI control inventory and outcome

`PASS` means the actual behavior was exercised and passed; `FAIL` means observed broken behavior; `BLOCKED` means a required service/credential/policy is unavailable; `NOT TESTED` means no actual UI behavior was observed. Source/API checks do not upgrade a browser control to `PASS`.

| Page / control | Expected behavior | Observed / test evidence | Status / repair and retest |
|---|---|---|---|
| Sidebar Overview | Show dashboard and active nav state | DOM/source supports hash page switch; browser unavailable | NOT TESTED visually |
| Sidebar Drug Pair Explorer | Navigate to explorer and close mobile nav | Source handler calls `go('explorer')`; browser unavailable | NOT TESTED |
| Sidebar Interaction Results | Navigate to current pair results | `go` function updates active page; browser unavailable | NOT TESTED |
| Sidebar 3D Molecular Lab | Navigate and lazy-load selected structures | route data tested; canvas/browser blocked | NOT TESTED rendering |
| Sidebar Explainability & Evidence | Navigate to current pair evidence | source/API shows real source records; browser unavailable | NOT TESTED |
| Sidebar Research Assistant | Navigate to chat panel | configuration endpoint works; browser unavailable | NOT TESTED |
| Sidebar Dataset & Model Insights | Navigate and request saved paired metrics | `/api/research-results` 200 | API PASS; UI NOT TESTED |
| Sidebar Prediction History | Show honest disabled/empty state | static copy says no history while predictions disabled | NOT TESTED visually |
| Sidebar System Status | Load current status/cards | `/api/system-status` 200 | API PASS; UI NOT TESTED |
| Sidebar About DDI-GUARD | Navigate to scope/caveats | static copy inspected | NOT TESTED |
| Mobile menu button / Escape | Open/close navigation, set `aria-expanded` | Handler/source present; browser unavailable | NOT TESTED |
| Overview “Check a drug pair” CTA | Navigate to explorer | `data-go` destination inspected | NOT TESTED |
| Overview “Explore molecules” CTA | Navigate to Molecular Lab | `data-go` destination inspected | NOT TESTED |
| Overview “Ask research assistant” CTA | Navigate to assistant | `data-go` destination inspected | NOT TESTED |
| Overview “View evaluation” CTA | Navigate to insights | `data-go` destination inspected | NOT TESTED |
| Drug A search input | Query actual reviewed catalog; loading/empty/error feedback | search endpoint 200 for Abacavir; browser states not observed | API PASS; UI NOT TESTED |
| Drug B search input | Query actual reviewed catalog | same search route | API PASS; UI NOT TESTED |
| Autocomplete options (A/B) | Select reviewed name/ID and show identity | catalog returns real records; click/keyboard behavior not browser-tested | NOT TESTED |
| Enter in search input | Submit search | handler inspected; browser unavailable | NOT TESTED |
| ArrowDown / Escape in autocomplete | Focus first option / dismiss options | added in repair phase; browser unavailable | NOT TESTED after repair |
| Selected-drug remove buttons | Remove one drug and clear stale dependent content | initial code left old pair/evidence; repaired in source | FAIL at code inspection; browser retest BLOCKED |
| Swap pair | Reverse order and refresh pair record | baseline API pair route 200; source handler inspected | NOT TESTED browser behavior |
| Clear pair | Clear selections, inputs, evidence, and lab state | initial code left typed input values; repaired | FAIL at code inspection; browser retest BLOCKED |
| Inspect pair evidence | Validate pair, load annotation/evidence, show actual 503 | actual pair lookup 200; prediction 503 | API PASS; UI NOT TESTED |
| Use verified example | Load Abacavir + Leflunomide, real DDInter pair and both conformers | each structure endpoint and pair lookup 200 | API PASS; UI NOT TESTED |
| Results page | Show label and evidence distinct from prediction | static renderer source inspected; no model response | NOT TESTED |
| 3D representation selector | Switch ball-stick, stick, sphere, line | style handler inspected; Jmol applied to element styles after repair | NOT TESTED; WebGL BLOCKED |
| Atom labels button | Toggle atom labels and accessible pressed state | state handler/source; aria state added | NOT TESTED after repair |
| Reset cameras | Center/fit each available viewer | code calls `zoomTo()`/`render()`; official API documents `zoomTo` | NOT TESTED browser |
| Drug A / Drug B viewer canvases | Show selected real 3D structure; rotate/zoom/pan | structure endpoint verified; actual WebGL/CDN unavailable | BLOCKED browser/CDN verification |
| Combined comparison scene | Show separate A/B structures with no binding inference | source constructs two models with visual X offset and disclosure | NOT TESTED rendering |
| 2D fallback | Show only validated 2D depiction if 3D missing | API returned valid SVG for a validated structure | API PASS; image/browser display NOT TESTED |
| Explainability source links | Open actual PubChem/ChEMBL/DailyMed source | source URLs inspected/returned by real records | NOT TESTED click target |
| Evidence expansion/collapse | Not advertised or present | No disclosure/accordion control exists | N/A; no decorative dead control |
| Three assistant quick-prompt buttons | Put selected prompt into composer | handler inspected; browser unavailable | NOT TESTED |
| Pair-context checkbox | Include reviewed selected-pair context on submit | API context exercised in TestClient; no real provider | API PASS under mocked provider; UI NOT TESTED |
| Chat history selector | Switch in-memory conversation | state handler inspected; browser unavailable | NOT TESTED |
| New conversation button | Create/reset conversation | `newSession()` inspected; no persistence claimed | NOT TESTED |
| Chat composer empty/send | Ignore blank input; submit non-empty prompt | no-key request returns 503 through API | BLOCKED live provider; UI NOT TESTED |
| Enter / Shift+Enter | Enter sends, Shift+Enter inserts newline | handler inspected; busy guard repaired against duplicate request | NOT TESTED after repair |
| Send button/loading state | Show composing indicator, disable while request active | no-key backend is 503; source stores honest error; guard repaired | BLOCKED provider; browser retest NOT TESTED |
| Assistant error retry | Retry failed user turn with real provider | mocked adapter retry path tested, no real provider | BLOCKED live provider; UI NOT TESTED |
| Assistant response copy | Copy response; report unavailable clipboard | source inspection found false “Copied” when Clipboard API missing; fixed | FAIL at code inspection; browser retest BLOCKED |
| Markdown/code rendering | Render safe limited headings/lists/code | escaping/source inspection; no DOM browser test | NOT TESTED |
| Prediction-history/export/download/upload controls | Only show supported actions | No export/download/upload or history controls are present | N/A; not claimed |
| Status cards | Show APIs/approval/checkpoint/coverage/assistant/QA | backend endpoints tested; UI rendering unobserved | API PASS; UI NOT TESTED |

No modals, dialogs, tooltips, accordions, filter/sort controls, pagination, upload, or export controls were found in the current UI; none is claimed as implemented.

## Drug search / API / Molecular Lab follow-up (2026-10-21)

- **Root cause verified:** the static page was opened as a `file://` document while its API calls use `/api/...` paths. A browser-style Node fetch against a file URL failed with `TypeError: fetch failed`; local file mode has no HTTP origin for the API. The intended architecture serves page and API from one FastAPI origin.
- **Repair:** the UI now warns users to start Uvicorn and open `http://127.0.0.1:8000/`; backend calls use one configurable API client with timeout and specific network/timeout errors. Optional API origin injection and an exact-origin CORS allow-list are documented; default same-origin use needs no CORS.
- **Search correction:** whitespace is trimmed and names are case-insensitive. Exact identity matches are returned alone before partial alias matches. A real `Aspirin` query now returns only `DDInter20 · Acetylsalicylic acid`; `Warfarin` returns only `DDInter1951`.
- **Selected drug details:** selection now asynchronously fetches `/api/drugs/{id}` and displays available PubChem CID, formula, mass, SMILES, structure provenance/status, and available ChEMBL/DailyMed record counts. Per-slot request IDs keep old detail responses from changing a new selection.
- **Structure outcomes:** Aspirin has a verified 3D conformer. Warfarin reports no graph-matched 3D conformer and displays the validated RDKit 2D SVG fallback, clearly labeled 2D. No coordinates were generated or faked.
- **Observed HTTP checks:** a temporary Uvicorn process on ephemeral loopback returned 200 for root/search/details/structure, 0 results for an unknown drug query, 422 for blank search, CORS allow-origin only for its configured origin, and 503 for prediction. The temporary process was stopped. Full 71-test suite, compileall, JS syntax, and saved paired-evaluation validation passed; current snapshot is `artifacts/qa_status.json`.
- **Still not tested:** browser autocomplete clicks/keyboard use, selected-detail rendering, CDN load, WebGL controls, responsive sizes, console/network panel, and actual visual/accessibility behavior. Browser use remains unavailable in this Work session. Detailed API values and limitations are in [`drug_lookup_and_3d_fix_report.md`](drug_lookup_and_3d_fix_report.md).

## Repairs made during this audit

1. Pair selection/remove/swap now invalidates the previous pair result and clears corresponding result/evidence screens and viewer content, preventing stale source evidence from appearing under a different pair.
2. Clear now also clears search text, autocomplete choices, stored structures, selected pair, and 3D display state.
3. Search results are guarded by a per-field request sequence and query check so a slower earlier response cannot replace newer results.
4. Pair inspection is guarded by request ID and selected drug identity; stale lookup/prediction responses cannot repaint the current pair or leave its button disabled.
5. Molecular loading is guarded by a lab request generation and drug IDs; stale structure/CDN responses cannot replace the newly selected structures. Navigating away releases WebGL contexts and object URLs are revoked.
6. Chat submission blocks duplicate sends while busy. Retry captures the failed session’s actual user turn, rather than a global last-message pointer. Clipboard failure no longer reports success.
7. Individual views now expose the 3Dmol Jmol element-color key; line rendering explicitly uses the Jmol element scheme. Atom-label toggle exposes `aria-pressed`; autocomplete fields expose list relationships, and ArrowDown/Escape have keyboard behavior.

The fixes are source- and API-level. Because browser UI access is unavailable, the above cannot yet be marked UI-ret-tested; see [end-to-end test report](end_to_end_test_report.md) for exact passing and blocked checks.

## Current blockers and next verification

- No real AI API key: only failure handling and mocked adapter tests can be checked; live provider requests remain blocked.
- No browser automation package and browser use disabled in Work settings: UI action, console/network, 1440/1280/768/390 viewport, keyboard/screen-reader, and WebGL interaction checks remain not tested.
- Prediction remains blocked by project approval and checkpoint preprocessing compatibility; the 503 is expected and must not be bypassed.
- Run `/opt/miniconda3/bin/python3 scripts/verify_project.py` after current repairs. Then, in Chrome/Brave on the demo Mac, navigate to `http://127.0.0.1:8000`, exercise each listed control, and confirm 3Dmol CDN/WebGL. Do not claim final acceptance until those browser checks and a live AI-provider test (if chat is part of the demo) are complete.
