# DDI-GUARD Research Prediction Enablement Report

**Verification date:** 2026-10-21  
**Scope:** Local academic research/demo prediction only. Production and clinical use remain blocked.  
**Current routes:** `POST /api/research/predict` is enabled for verified feature support; `POST /api/predict` remains HTTP 503.

## Result

The research endpoint now runs real inference from hash-pinned, reloaded checkpoints and the same training-time feature loaders and input builders. It does not use DDInter labels as model output. The UI displays actual research predictions and errors separately from retrieved DDInter annotations and drug-level source evidence.

The earlier `inference_preprocessing_matches_checkpoint=false` value was static approval metadata: `model_approval_status()` read the boolean without executing a parity check. Inspection found the training and offline inference path share `_load_sources`, `_pair_mask`, `_make_inputs`, molecular graph conversion, and the same label order. I added executed parity checks against saved pair-disjoint test probabilities for each registered candidate. During this check, parity initially failed for one molecular+target pair because the verification script marked every configured modality present instead of using the source-derived training `_pair_mask`. Reusing `_pair_mask` fixed the comparison; all four candidates then passed. This was a checker-mask error, not a checkpoint or feature-pipeline change.

The documented research gate now records the completed verification and explicit project-owner request for a local academic demo. `research_demo_policy.allowed=true` is limited to `/api/research/predict`. `approved=false`, `checkpoint_path=null`, `api_prediction_enabled=false`, and the production endpoint remains 503. No clinical validity or regulatory approval is claimed.

## Checkpoint and routing

The all-three-modality model remains the preferred checkpoint because it was selected by validation macro-F1/loss for the pair-disjoint, known-drug/new-pair research scope. Held-out test results were not used to select it.

| Research route model | Checkpoint | SHA-256 | Trained inputs |
|---|---|---|---|
| Full candidate | `artifacts/multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt` | `9259c34a964c48d30c849cb5480c8ac0a1453abf3b4936608c1a2435054d965c` | Molecular + target + text |
| Pair-disjoint fallback | `artifacts/multimodal_pair_disjoint_molecular_target_chembl34_diagonal_v1/best_validation.pt` | `f6d6f8b760358c5575e314b74b9127bb239cebe495ab0ae00599681cae2d90dc` | Molecular + target |
| Pair-disjoint fallback | `artifacts/molecular_pair_disjoint_no_placeholder_v1/best_validation.pt` | `273c55b26107e13d1da3d370de2ba2367c3df515aeed36be65bf012b477fff2e` | Molecular |
| Pair-disjoint fallback | `artifacts/multimodal_pair_disjoint_target_diagonal_relation_v1/best_validation.pt` | `7c2da588f58185becb4de502c9efe41cc3ced174f4463716a2c8516d1e17ad8b` | Target |

For both drugs, routing prefers full molecular+target+text, then molecular+target, then molecular, then target. Each candidate also verifies its config, evaluation JSON, prediction artifact, source hashes/fingerprint, dataset/split hashes, class order, dependency versions, and strict checkpoint reload before use. Inference reuses the training pipeline and applies the validation-only scalar temperature. No text-only fallback or synthetic missing modality is used. A pair unsupported by every checkpoint returns HTTP 422; unknown inputs return 404; same-drug pairs return 422.

### Feature coverage

- Catalog-level loader coverage: molecular 1,472 / 1,939 IDs; ChEMBL target 1,166 / 1,939; DailyMed text vectors 652 / 1,939; all three modalities 517 / 1,939.
- In the labeled-pair universe (1,902 IDs), the coverage audit reports molecular 1,436, target 1,134, and text 634. The denominator differs because catalog IDs can fall outside the labeled-pair universe.
- DailyMed remains partial (652 of 1,939); the model does not require text for fallback checkpoints. A missing text record is explicitly disclosed and does not imply safety.
- Modality coverage is per-drug feature presence; pair-level support requires both selected drugs to have each modality required by the chosen checkpoint. The system returns an explicit unsupported error otherwise.

## Inference verification

`ddi_guard/research_inference.py` validates the registered checkpoint and all associated artifacts before loading. `ddi_guard/api.py` exposes the adapter only on `/api/research/predict`, after the separate research policy and parity/test checks are true. `/api/predict` continues to require production approval and remains blocked.

A real full-model request for `DDInter1` Abacavir + `DDInter10` Acamprosate returned:

- Predicted research severity: **Moderate**
- Class order: Minor, Moderate, Major
- Probabilities: Minor `0.01247894`, Moderate `0.93773216`, Major `0.04978887`
- Checkpoint: `artifacts/multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt`
- Validation-only temperature: `2.085651159286499`
- Modalities supplied: molecular, target, text for both drugs
- No DDInter pair annotation was returned for this pair; it was not used as a prediction target at inference time.

These are outputs of the research checkpoint, not patient-level risks or validated clinical confidence. The response includes canonical drug IDs/names, checkpoint version/hash, selected modality set, each drug’s actual modality coverage, and source provenance. DailyMed evidence is general drug-label context; ChEMBL records are bioactivity context, neither establishes pair-specific causality.

## UI and API changes

- Drug Pair Explorer now calls `POST /api/research/predict`, shows returned class probabilities and checkpoint/calibration/modality details, and displays HTTP/API errors instead of placeholders.
- DDInter annotations remain a separate “SOURCE DATA · NOT A MODEL OUTPUT” card. Drug-level evidence remains source attributed.
- Pair request sequencing checks still prevent a delayed response from being shown for a newly selected pair.
- System Status distinguishes research endpoint availability from production 503 readiness.
- The research-only disclaimer is visible with prediction output. Prediction history remains empty because no prediction logging has been implemented.

## Verification record

Commands executed in `<PROJECT_ROOT>`:

```bash
python -m unittest discover -s tests -v
python -m compileall -q ddi_guard tests
python - <<'PY'
import re
from pathlib import Path
html = Path('ddi_guard/static/index.html').read_text()
Path('/private/tmp/ddi_guard_inline.js').write_text('\n'.join(re.findall(r'<script>([\s\S]*?)</script>', html)))
PY
node --check /private/tmp/ddi_guard_inline.js
```

Results:

- Full suite: **73 tests passed**, `OK`.
- Targeted research inference suite: **9 tests passed**, `OK`.
- Python `compileall`: passed.
- Inline JavaScript `node --check`: passed.
- Saved calibrated prediction parity: all four candidate checkpoints matched a saved pair-disjoint test prediction with every trained modality present at `rtol=1e-5`, `atol=1e-6`; max absolute differences were approximately `1.40e-08` (full), `1.79e-07` (molecular+target), `2.98e-08` (target), and `2.68e-07` (molecular).
- FastAPI TestClient made real requests through `/api/research/predict`, including successful model inference, unknown/unsupported inputs, closed-gate behavior, and the production `/api/predict` 503 behavior. The molecular+target saved-artifact parity pair was `ddi_0001ee1d1b128b3940df`; all trained inputs were present for that comparison.
- The prior process on port 8000 was verified to be the DDI-GUARD Uvicorn server from this workspace, stopped, and restarted with the updated code. Live loopback curl then returned research HTTP 200 with the expected actual Abacavir+Acamprosate model output and production HTTP 503. The server is running at `127.0.0.1:8000`. Browser visual review and WebGL rendering remain **not observed** because browser use is disabled in Work settings.

No model was retrained. Existing checkpoint, split, source, evaluation, and prediction artifacts were not overwritten. The only model-adjacent changes are the descriptive candidate/approval metadata and new runtime validation/routing code.

## Mac run and request commands

From the project folder, start the API and UI:

```bash
cd "<PROJECT_ROOT>"
/opt/miniconda3/bin/python3 -m uvicorn ddi_guard.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Request a research prediction from another terminal:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/research/predict \
  -H 'Content-Type: application/json' \
  -d '{"drug_a":"DDInter1","drug_b":"DDInter10"}'
```

Production route check (expected 503):

```bash
curl -i -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"drug_a":"DDInter1","drug_b":"DDInter10"}'
```

## Limitations and unresolved blockers

- Pair-disjoint full-model test macro-F1 is 0.7473 on 1,802 matched-support pairs. Cold-drug macro-F1 is 0.3664 on 253 pairs, including only 14 Minor pairs; cold-drug generalization is weak.
- DailyMed coverage is incomplete and 429 source failures remain documented. No retraining was attempted.
- The all-three checkpoint supports only pairs where both drugs have all three real feature types; fallback checkpoints expand technical coverage but have their own pair-disjoint training distributions.
- Evaluation is on DDInter dataset annotations, not clinical outcomes. No independent external or clinical validation exists.
- `approved` remains false; production adapter, deployment monitoring, security/change control, and any applicable regulatory/institutional approvals remain unresolved.
- Browser visual review, responsive interaction, and WebGL rendering remain to be performed on the user's Mac. The updated local Uvicorn server is currently running at `http://127.0.0.1:8000` and both research/production routes were checked by live loopback HTTP requests.
