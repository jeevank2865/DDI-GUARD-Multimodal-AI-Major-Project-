# DDI-GUARD UI and Research Prediction Implementation Report

**Updated:** 2026-10-23  
**Scope:** UI/UX and presentation of existing verified research inference. No model training, checkpoint, dataset, approval setting, prediction calculation, or backend prediction policy was changed in this pass.

## Delivered

- Reorganized the application around six research workflow sections: Overview, DDI Prediction, Explainability, Risk Assessment, Risk Mitigation, and Complete Report. Existing research tools remain available in their own navigation group.
- Improved visual hierarchy, spacing, responsive layout, result presentation, selected-pair summary, keyboard-accessible search suggestions, and screen-reader navigation state.
- The pair explorer calls the existing `POST /api/research/predict` endpoint and renders its response, probabilities, modality coverage, model version, checkpoint hash, and calibration details. Failures remain visible as errors; the interface does not synthesize predictions or evidence.
- Added source-linked DailyMed label passages in Risk Mitigation with a clear statement that drug-label excerpts are general drug-level evidence and are not a pair-specific treatment plan.
- Added a Complete Report view using only the selected pair's actual application response and available evidence. Its **Print / Save PDF** action uses the browser's print dialog; the app does not claim to create or store a server-side PDF.
- Updated UI contract tests to match the new sections and labels while preserving the production API's disabled-policy assertions.
- Replaced the Overview's simplified three-step strip with a numbered, top-to-bottom, ten-stage architecture. The three real feature pathways remain parallel through construction and encoding, then converge at the implemented attention and gated-fusion stage.
- Added keyboard- and pointer-accessible node details, related-path highlighting, reduced-motion support, and a stacked small-screen layout. DDInter annotations are explicitly separated from feature modalities, and evidence/risk/mitigation stages are labeled as downstream retrieval and reporting.

## Checkpoint and prediction verification

The academic pair-explorer candidate is:

`artifacts/multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt`

SHA-256: `9259c34a964c48d30c849cb5480c8ac0a1453abf3b4936608c1a2435054d965c`.

The file hash matches `artifacts/research_demo_checkpoint.json`. The registry records pair-disjoint scope, all three real modalities, and selection at epoch 7 by validation macro-F1 followed by validation loss (validation macro-F1 0.7605); held-out test results were not used to select the checkpoint. Calibration temperature 2.085651 was fit on validation data only. This checkpoint is appropriate for a local academic demonstration of new pairs among supported, reviewed drug identities. It is not approved for production or clinical use.

A browser pair-explorer smoke run used the verified example Abacavir (DDInter1) + Leflunomide (DDInter1033). The research endpoint returned a real model response: **Major**, with probabilities Minor 0.1%, Moderate 5.2%, Major 94.6%; molecular, target, and text modalities were available for both drugs. The separately retrieved DDInter annotation was also Major and was presented as a dataset annotation, not as the model output. This single example is a UI/integration check, not an estimate of model performance or clinical evidence.

The registry reports verified input coverage of 1,472 molecular, 1,166 target, 652 text, and 517 all-three-modality drugs. DailyMed coverage is 652 of 1,939 mapped drugs (33.6%); known partial-source errors remain documented in `artifacts/dailymed_evidence_coverage_audit.json`. The system selects only a checkpoint compatible with the real available modalities; unsupported pairs return an error rather than fabricated output.

## Evaluation context and limits

The existing registry reports pair-disjoint test macro-F1 0.7473 and cold-drug test macro-F1 0.3664. The cold-drug result is substantially weaker, and its Minor-class support is only 14 pairs. These are saved research evaluation results, not clinical performance guarantees. DailyMed label content is not pair-specific DDI proof, ChEMBL bioactivity is not direct pair-mechanism evidence, and DDInter labels are dataset annotations rather than patient outcomes.

Research inference remains available only on its separate academic route. Production `POST /api/predict` remains disabled by policy and returns HTTP 503. No claim of clinical validity, regulatory approval, diagnosis, or treatment recommendation is made.

## Verification run

From the project root:

```sh
python scripts/check_frontend_syntax.py
python -m unittest discover -s tests -v
```

Observed results on 2026-10-22:

- Frontend inline JavaScript syntax: **passed**.
- Full `unittest` suite: **74 tests passed** in 14.320 seconds.
- The suite includes actual authorized research inference, repeatability, checkpoint hash/reload checks, unsupported-input handling, production HTTP 503 behavior, frontend sections/accessibility contracts, source-annotation separation, and a new top-to-bottom architecture contract test.
- `pytest` is not installed in the active environment; the full repository `unittest` suite was run successfully instead.

## Changed files in this UI pass

- `ddi_guard/static/index.html`
- `tests/test_api.py`
- `tests/test_research_inference.py`
- `artifacts/DDI_GUARD_UI_IMPLEMENTATION_REPORT.md`

## Unresolved checks

- External AI assistant provider connectivity was not exercised as part of this UI pass.
- Browser WebGL rendering and operating-system print-to-PDF behavior are environment-dependent; the report action is wired to the browser print dialog, but an exported PDF file was not generated or inspected here.
- The existing cold-drug generalization gap, small cold-drug Minor support, and incomplete DailyMed coverage remain research limitations.
