# DDI-GUARD academic demonstration checklist

## Before the demo

- Use the existing verified workspace and checkpoints; do not train or overwrite artifacts during the presentation.
- Use the Mac environment verified at `/opt/miniconda3/bin/python3`.
- Confirm the source-data and model summary in `artifacts/final_audit_report.md` and `artifacts/model_card.md`.
- State at the start: this is academic research, not diagnosis, treatment, medication advice, clinical validation, or regulatory approval.
- Explain that `/api/predict` is intentionally disabled and returns HTTP 503. Do not describe the offline adapter as a user-facing prediction service.

## Start commands

In Terminal 1:

```bash
cd "<PROJECT_ROOT>"
/opt/miniconda3/bin/python3 -m uvicorn ddi_guard.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. Keep Terminal 1 visible as a recovery/debug option. The application uses local project artifacts and binds to localhost.

Optional pre-demo verification (run before starting the demo):

```bash
cd "<PROJECT_ROOT>"
/opt/miniconda3/bin/python3 -m unittest discover -s tests -v
/opt/miniconda3/bin/python3 -m compileall -q ddi_guard tests scripts
/opt/miniconda3/bin/python3 scripts/check_frontend_syntax.py
```

Last audited result: 55 tests passed; compilation and frontend syntax checks passed. Re-run these commands if any project file changes.

## Suggested walkthrough

1. Search for a reviewed drug in the Drug Pair Explorer and show the identity/source mapping.
2. Select two reviewed drug IDs and inspect actual DDInter annotations and any available ChEMBL/DailyMed source-backed evidence.
3. Point out missing modalities and provenance labels. General drug-label text and ChEMBL bioactivity are not direct proof for the selected pair.
4. Explain that the interface displays the actual HTTP 503 result for prediction. Do not infer a severity or safety conclusion from missing evidence or from an unavailable prediction.
5. Present the saved experiment tables and paired intervals from `artifacts/paired_multimodal_validation_v1/paired_multimodal_validation.md`.
6. Highlight the cold-drug support: 14 Minor, 206 Moderate, 33 Major; cold-drug full-model macro-F1 0.3664. Describe this as a limitation.
7. End with reproducibility and future work: partial DailyMed coverage, independent evaluation, and no clinical approval.

## Expected UI behavior

- Drug search and pair lookups show only reviewed source-backed records.
- Target/text evidence may be missing and is not fabricated.
- Pair annotations are DDInter dataset labels, not patient outcomes.
- A valid prediction request is blocked by policy with HTTP 503; invalid/unknown identifiers return validation/not-found errors.
- No probabilities, severity score, recommendation, or evidence should be invented or manually represented as a model result.

## Test data and fallback plan

Verified pair example: **Warfarin (`DDInter1951`) + Metformin (`DDInter1164`)**. The current lookup returns a DDInter `Moderate` dataset annotation and available target/text source records. The annotation is not a patient outcome, and those records are not proof of pair-specific mechanism. A valid `POST /api/predict` request for the pair still returns HTTP 503 by policy. Other pairs may have different source coverage. If the UI or local server fails:

1. Stop the presentation from displaying any stale or manually prepared score.
2. Show the saved audit/paired-evaluation reports and their exact support/limitations instead.
3. If the server cannot start, use the static report artifacts and explain that interactive evidence lookup is unavailable in that run.
4. Do not start a training job, alter approval metadata, or enable a prediction endpoint as an improvised fallback.

## Presentation checklist

- [ ] Explain target labels and dataset scope.
- [ ] Distinguish unique-drug coverage from pair-level model support.
- [ ] Show the provenance-preserving architecture and source audits.
- [ ] Report validation and held-out test metrics separately; call macro-F1 macro-F1, not accuracy.
- [ ] Mention paired uncertainty, matched supports, and the cold-drug 14-Minor limitation.
- [ ] Demonstrate real pair lookup/evidence only.
- [ ] Show the HTTP 503 gate and explain why user-facing predictions are not enabled.
- [ ] Make no clinical, safety, regulatory, or production-readiness claim.
