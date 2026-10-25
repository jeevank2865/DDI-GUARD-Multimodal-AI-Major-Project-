# DDI-GUARD drug lookup, API connection, and molecular structure fix

**Date:** 2026-10-21  
**Scope:** Drug search, local drug details, API connection diagnostics, structure API, and Molecular Lab error/fallback behavior. No model training, prediction approval change, or dataset rewrite was performed.

## Findings and root cause

The frontend is a static HTML/JavaScript application served by the same FastAPI app as the API. It is not a React application and has no separate frontend server or proxy configuration. Its backend calls use `/api/...` URLs and are intended to be same-origin.

The project UI had been opened at a `file://` path in the desktop session. A root-relative API request has no HTTP application origin in that mode, so browser fetch fails before it reaches FastAPI; the UI showed the browser's generic `Failed to fetch`. This was reproduced with Node's fetch on a `file://` URL (`TypeError: fetch failed`). Browser developer tools were unavailable in this Work session, so no browser console/network capture is claimed.

The backend routes work in FastAPI's HTTP test client. The default sandbox denied the first local TCP bind (`PermissionError: Operation not permitted`); a temporary Uvicorn process was then run on an ephemeral `127.0.0.1` port through the reviewed local-test path and stopped after the requests. Browser developer tools remained unavailable, so this establishes HTTP integration but not browser-rendered behavior.

A second verified issue was search ranking: `Aspirin` matched a number of products whose long PubChem synonym lists merely contained the word, and the canonical `Acetylsalicylic acid` record was not first. Search now ranks exact names and aliases before prefix/substring matches.

## Changes

- `ddi_guard/api.py`
- Search trims whitespace and matches case-insensitively; when exact canonical names, IDs, CIDs, or aliases exist, only those exact identity matches are returned. Partial name/synonym matches are used otherwise.
  - `/` injects only the public `DDI_GUARD_PUBLIC_API_BASE_URL` setting into frontend configuration; invalid or credential-bearing URL values fail closed.
  - Optional exact-origin CORS allow-list is configured by `DDI_GUARD_CORS_ORIGINS`; credentials are disabled. Default same-origin use needs no CORS setup.
  - `/api/system-status` distinguishes the local catalog, local structure artifacts, external lookup (not performed), viewer/CDN status (not checked by an API request), and model readiness.
- `ddi_guard/static/index.html`
  - All backend calls now use one API client with same-origin default, optional injected base URL, a 15-second timeout, actionable connection errors, and explicit instructions when the page was opened as `file://`.
  - Selecting a drug immediately loads its verified local detail record, formula/weight/SMILES when present, PubChem source, structure availability, and available ChEMBL/DailyMed evidence counts. Per-slot request identity guards prevent stale detail results from replacing the current selection. Pair lookup/prediction remains a separate action.
  - Molecular Lab distinguishes a verified 3D conformer from an unavailable conformer and a 2D SVG fallback; a retry action is shown on load failure. API errors are not shown as “no structure” unless the API returned the explicit not-found condition.
  - System Status now avoids claiming that the browser's 3Dmol.js CDN/WebGL viewer was checked by the backend.
- `tests/test_api.py` and `tests/test_research_inference.py`
  - Added regression checks for exact Aspirin/Warfarin results, whitespace/case handling, empty and unknown searches, public API-base injection/validation, status separation, Warfarin's 2D-only structure, central API use, and unchanged HTTP 503 prediction behavior.
  - Updated an existing source assertion to expect the centralized API client.
- `.env.example` and `README.md`
  - Document optional public API-origin/CORS configuration and the required Uvicorn-served URL. No secret was added to frontend configuration.

## Routes and API configuration

- `GET /api/drugs/search?q=<name-or-alias>&limit=8` — reviewed local catalog search.
- `GET /api/drugs/{DDInter_ID}` — canonical ID, verified molecular fields, structure provenance, and available drug-level evidence.
- `GET /api/drugs/{DDInter_ID}/structure-3d` — verified graph-matched conformer atoms/bonds, or `available:false` with reason.
- `GET /api/drugs/{DDInter_ID}/structure-2d.svg` — RDKit depiction from the validated canonical structure, explicitly labeled 2D.
- `GET /api/system-status` — backend, catalog, local structure artifacts, unchecked browser viewer, approval and inference state.
- `POST /api/predict` — still fail-closed; known IDs return HTTP 503 while approval/preprocessing checks remain unresolved.

Normal local setup serves both UI and API from Uvicorn at <http://127.0.0.1:8000>; `DDI_GUARD_PUBLIC_API_BASE_URL` should remain unset. For an intentional cross-origin frontend, set that variable to an HTTP(S) origin and set `DDI_GUARD_CORS_ORIGINS` to the frontend's exact origin(s). CORS was checked with an allowed `Origin: http://localhost:5173`, which received that exact `Access-Control-Allow-Origin` value. No wildcard or credentials are enabled.

## Verified lookup results

These were real calls to the existing local reviewed catalog through `TestClient`; no drug records were fabricated and no live PubChem request was made.

| Query / ID | Observed result |
|---|---|
| `Warfarin` | HTTP 200; exactly 1 result: `DDInter1951 · Warfarin`; CID 54678486; formula C19H16O4; MW 308.3; validated molecular graph and source URL present; ChEMBL and DailyMed evidence statuses available. |
| `Aspirin` / `ASPIRIN` | HTTP 200; exactly 1 result: `DDInter20 · Acetylsalicylic acid`; CID 2244; formula C9H8O4; MW 180.16; canonical SMILES and PubChem provenance present; a verified 3D conformer is available. |
| `not-a-reviewed-drug-987` | HTTP 200 with `result_count: 0` and an empty `results` array. This means no local catalog match, not that the substance does not exist. |
| whitespace-only query | HTTP 422 with `Search query cannot be blank`. |
| unknown DDInter ID detail | HTTP 404 from the existing route tests. |

Warfarin's `/structure-3d` response is HTTP 200 with `available:false`; the stored conformer has a structure/atom-mapping mismatch and is not rendered as 3D. Its validated local 2D depiction route returns HTTP 200, SVG, and `X-Structure-Dimension: 2D`. Aspirin (`DDInter20`) returns `available:true` for verified 3D data. The two verified demonstration conformers (`DDInter1` and `DDInter1033`) remain covered by regression tests.

The local structure corpus stores SMILES, formula, mass, atoms/bonds, PubChem source URL, and conformer validation status. It does not store an InChIKey field for this API, so the UI does not invent or display one. External PubChem is not queried at runtime: the app uses the local source-backed dataset and reports its provenance. There is no evidence here about external-provider availability, 429/5xx behavior, or current network freshness.

## Tests and observed results

Latest full verification: `artifacts/qa_status.json`, status `passed`, **71 tests**, all four recorded checks exit 0.

```bash
cd "<PROJECT_ROOT>"
/opt/miniconda3/bin/python3 -m unittest tests.test_api tests.test_research_inference -v
/opt/miniconda3/bin/python3 scripts/verify_project.py
/opt/miniconda3/bin/python3 -m compileall -q ddi_guard tests scripts
/opt/miniconda3/bin/python3 scripts/check_frontend_syntax.py
```

Observed outcomes:

- Focused API/inference run passed: 22 tests; the full 71-test suite passed. The final verifier snapshot is `2026-10-21T09:55:33.254066+00:00` UTC.
- Python compilation passed; inline JavaScript syntax passed.
- Saved paired evaluation validation passed with the same saved supports/results.
- `GET /`, Warfarin/Aspirin searches, detail calls, structure endpoints, system status, unknown search, and `POST /api/predict` were exercised using FastAPI TestClient. The prediction response remained HTTP 503 with `inference_preprocessing_matches_checkpoint` named.
- Optional CORS allow-list behavior passed a TestClient request in a fresh process configured with `DDI_GUARD_CORS_ORIGINS=http://localhost:5173`.
- A temporary Uvicorn TCP smoke test passed on an ephemeral loopback port and was stopped in cleanup. No claim of browser selection, actual WebGL drawing, browser console inspection, or responsive visual pass is made.

The Uvicorn smoke test observed: `GET /` 200 HTML with injected API configuration; Warfarin/Aspirin search 200 with expected first identities; unknown search 200 with zero results; blank search 422; detail and structure endpoints 200; the configured CORS origin received its exact allow-origin header; and known-pair prediction 503 naming the unresolved preprocessing check.

## Startup and remaining limitations

```bash
cd "<PROJECT_ROOT>"
/opt/miniconda3/bin/python3 -m uvicorn ddi_guard.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, not the `.html` file from Finder. 3Dmol.js is loaded from its CDN, so actual viewer rendering still needs a browser with network access. Browser controls and viewport behavior remain unverified in this session. Live PubChem fallback/retry was intentionally not added: the verified local corpus already supports these examples, and no genuine external-provider check was performed. Configure a real HTTP service and test it separately before claiming live-provider support.

The change does not modify checkpoints, manifests, corpus files, model approval, or prediction readiness. `POST /api/predict` remains HTTP 503 and DDI-GUARD remains a research-data exploration project, not a diagnostic or clinical application.
