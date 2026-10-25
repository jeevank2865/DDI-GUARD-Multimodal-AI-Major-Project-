# DDI-GUARD model card

**Version:** academic research artifact snapshot, 2026-10-20  
**Status:** not approved for production or clinical use; API prediction is disabled (HTTP 503).

## Intended use

Research and classroom demonstration of multimodal classification of DDInter drug-pair severity annotations, plus exploration of source records and feature provenance. The project does not estimate an individual patient's risk and is not a diagnostic, prescribing, treatment, or medication-reconciliation tool.

## Target and labels

The model predicts one of Minor, Moderate, or Major as represented in DDInter. These are curated dataset labels rather than prospective patient outcomes or validated clinical severity outcomes. Label imbalance is substantial: among 130,422 labeled pairs, 6,833 are Minor, 96,675 Moderate, and 26,914 Major.

## Model and chosen research checkpoint

The multimodal model uses an edge-conditioned molecular graph encoder, relation-aware drug–target graph encoder, learned text-vector projection, order-invariant pair features, masked cross-modal attention/gating, and a three-class classifier. The checkpoint chosen for offline academic reproducibility is:

`artifacts/multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt`

SHA-256: `9259c34a964c48d30c849cb5480c8ac0a1453abf3b4936608c1a2435054d965c`. It was selected at epoch 7 using validation macro-F1 (0.7605), then validation loss; held-out test results were not used for selection. Its validation-fitted temperature is 2.085651. The exact config, evaluation, pair coverage, split, feature-manifest, target-graph and source hashes are in `artifacts/research_demo_checkpoint.json`.

This is a technical research candidate only. `ddi_guard/research_inference.py` is offline, tested, and not connected to the API or a prediction CLI. It refuses unsupported drugs and any missing modality; it does not fall back to placeholder features.

## Data and coverage

- **DDInter:** 222,383 raw rows; 130,422 labeled pairs; seed-42 pair-disjoint and cold-drug split manifests.
- **Molecular:** 1,436/1,902 labeled-pair drug IDs have accepted PubChem/RDKit graph features; 466 do not.
- **ChEMBL 34:** 105,982 quality-filtered source edges; graph has 1,166 drugs and 2,699 targets; 1,134 IDs overlap labeled-pair drugs. These are bioactivity relations, not proven DDI mechanisms.
- **DailyMed:** 879 provenance-backed passages and 652/1,939 mapped drugs; 634 overlap labeled-pair drugs. There are 403 retrieval failures, mostly HTTP 429. The passages are general drug-label evidence, not pair-specific support.
- **Full-model support:** pair-disjoint 8,497 train / 1,798 validation / 1,802 test; cold-drug 5,649 / 356 / 253. It was not trained on every DDInter row.

See `artifacts/final_audit_report.md` for source hashes, row counts, and artifact paths.

## Evaluation

| Protocol | Full model test support | Macro-F1 | Macro OVR AUROC | Brier | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Pair-disjoint | 1,802 | 0.7473 | 0.9280 | 0.2177 | 0.3802 | 0.0163 |
| Cold-drug | 253 | 0.3664 | 0.5409 | 0.3624 | 0.6550 | 0.0708 |

Test metrics use validation-only temperature scaling. The pair-disjoint model was below molecular+text (macro-F1 0.7836), target+text (0.7743), and text-only (0.7708) on matched test support; it exceeded the molecular+target control (0.6130). Cold-drug paired intervals for full-model contrasts all include zero. Detailed per-class metrics/confusion matrices and bootstrap intervals are in `artifacts/paired_multimodal_validation_v1/paired_multimodal_validation.md`.

Cold-drug class support is Minor 14, Moderate 206, Major 33. The full model correctly predicted 2/14 Minor and 1/33 Major. This small support makes per-class conclusions uncertain. Pair-disjoint performance should not be generalized to unseen-drug performance.

## Known limitations and failure modes

- Partial and possibly biased availability of structures, target relations, and labels; the full model evaluates only pairs with all selected features.
- DailyMed source acquisition had 393 HTTP 429 errors plus 10 network timeouts/resets. Its partial corpus may not represent all labels or drug formulations.
- ChEMBL bioactivity edges and general drug-label passages do not establish a specific pair's mechanism or clinical severity.
- Cold-drug performance is weak; the 14 Minor test pairs are inadequate for a stable class estimate.
- Pair-disjoint results represent previously seen drug identities with unseen pairs; cold-drug is the new-drug stress test.
- DDInter may contain annotation artifacts, historical label conventions, or duplicate-source dependencies despite audited pair deduplication.
- No independent external dataset validation, prospective validation, clinical study, or regulatory authorization has been established.
- The public-facing API must continue returning HTTP 503 while the approval gate is closed. A valid offline test result is not an approved user prediction.

## Responsible use

For academic presentation, describe results as reproducing DDInter labels on fixed, modality-covered supports. Display source limitations and class support. Do not advise medication changes, infer safety from missing interactions, or present predictions as clinical facts. No production or clinical model approval is claimed.
