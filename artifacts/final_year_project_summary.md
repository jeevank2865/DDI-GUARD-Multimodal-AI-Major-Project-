# DDI-GUARD — Final-year project summary

## Project title

**DDI-GUARD: A multimodal, explainable research framework for drug–drug interaction severity classification**

## Problem statement

Drug–drug interaction (DDI) evidence is distributed across curated interaction datasets, molecular structure resources, target/bioactivity databases, and drug labels. This project studies whether those sources can be combined to classify dataset severity labels while preserving source provenance and reporting limitations.

## Objectives

1. Normalize and audit drug-pair labels from DDInter.
2. Build source-backed molecular, drug–target, and drug-label text features.
3. Train and compare molecular, target, text, and multimodal models under pair-disjoint and cold-drug splits.
4. Use validation-only checkpoint selection/calibration, matched test supports, reproducible artifacts, and paired uncertainty estimates.
5. Build a source-backed Drug Pair Explorer that communicates evidence and limitations honestly.

## Methodology and architecture

DDInter labels are ingested into canonical unordered drug pairs. Fixed seed-42 protocols split by pair identity and by drug identity. PubChem structures are reviewed and converted into RDKit-validated molecular graphs. ChEMBL 34 bioactivity records are quality-filtered while retaining activity, assay, target, relation, evidence, and provenance identifiers, then built into a relation-aware target graph. DailyMed SPL labels are retrieved with exact-name mapping, provenance checked, section passages extracted, and TF-IDF vectors fit without severity labels on protocol-specific training-drug text.

The full architecture contains molecular and target graph encoders, a trainable text projection, an order-invariant pair representation, masked multimodal fusion, and a three-class classifier. Checkpoint selection and temperature scaling use validation data only. Missing modalities are explicit; the full experiment support requires all three real modalities for both drugs.

## Datasets and verified coverage

- DDInter ingestion: 222,383 raw rows and 130,422 labeled pairs after accounting for duplicate and unknown-only records.
- PubChem/RDKit: 1,436 accepted molecular-feature drug IDs among 1,902 labeled-pair drug IDs; 466 remain unresolved or unusable.
- ChEMBL 34: verified archive; 105,982 filtered source edges; 1,166 graph drugs; 2,699 targets and 745 relations; 1,134 graph drug IDs overlap the labeled-pair universe.
- DailyMed: 879 source-backed passages and 652/1,939 mapped drugs; 634 overlap labeled-pair drugs. 403 request failures (393 HTTP 429) remain.
- Full model support: pair-disjoint train/validation/test 8,497/1,798/1,802; cold-drug 5,649/356/253. These are usable subsets, not all DDInter pairs.

## Experiments and results

| Protocol | Experiment | Best validation macro-F1 | Test macro-F1 | Test support |
|---|---|---:|---:|---:|
| Pair-disjoint | Text only | 0.7602 | 0.7708 | 1,802 |
| Pair-disjoint | Molecular + text | 0.7707 | 0.7836 | 1,802 |
| Pair-disjoint | Target + text | 0.7581 | 0.7743 | 1,802 |
| Pair-disjoint | Molecular + target | 0.6021 | 0.6130 | 1,802 |
| Pair-disjoint | Molecular + target + text | 0.7605 | 0.7473 | 1,802 |
| Cold-drug | Text only | 0.4579 | 0.3743 | 253 |
| Cold-drug | Molecular + text | 0.3808 | 0.3986 | 253 |
| Cold-drug | Target + text | 0.4088 | 0.3556 | 253 |
| Cold-drug | Molecular + target | 0.3042 | 0.2821 | 253 |
| Cold-drug | Molecular + target + text | 0.3989 | 0.3664 | 253 |

Cold-drug support contains 14 Minor, 206 Moderate, and 33 Major pairs. Full-model cold-drug recall is 2/14 for Minor and 1/33 for Major. Paired bootstrap comparisons are in `paired_multimodal_validation_v1/paired_multimodal_validation.md`; the full model does not reliably outperform alternatives on cold-drug support, and it is below text-containing ablations on pair-disjoint support.

## Research candidate and application

The selected academic candidate is `artifacts/multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text/best_validation.pt`, SHA-256 `9259c34a964c48d30c849cb5480c8ac0a1453abf3b4936608c1a2435054d965c`. Selection was validation-driven (epoch 7; validation macro-F1 0.7605), not based on test results. `ddi_guard/research_inference.py` verifies this checkpoint and preprocessing offline for reproducibility, but the adapter is not exposed to users.

The frontend supports source-backed search and pair exploration. The prediction API returns HTTP 503 because project policy does not allow a separate research-demo prediction route and production approval criteria remain unmet. This preserves the existing safeguard.

## Contributions

- Auditable ingestion and provenance across DDInter, PubChem, ChEMBL 34, and DailyMed.
- Pair-disjoint and cold-drug evaluation with matched multimodal ablations.
- Saved checkpoint, exact split/coverage manifests, calibration artifacts, and paired uncertainty analysis.
- Evidence-first pair explorer with explicit missing-data and prediction-unavailable behavior.

## Limitations and future work

The model does not use every source pair because modalities are incomplete. Cold-drug scores are weak, Minor support is small, DailyMed acquisition is partial, and the full model does not dominate simpler text-containing variants on pair-disjoint evaluation. DDInter labels are not patient outcomes; target bioactivity and general label passages are not pair-specific clinical evidence. Future research should recover DailyMed failures conservatively, assess additional independent datasets, improve new-drug generalization without using the current held-out test for tuning, and complete a separately reviewed demo inference policy if interactive predictions are desired.

## Responsible-use statement

This project demonstrates academic classification and data provenance. It is not for diagnosis, treatment, prescribing, or clinical decision-making. No clinical validity or regulatory approval is claimed. The prediction API remains disabled.
