# Paired Multimodal Research Validation

Generated from saved artifacts only; no model was retrained and no test data were used for selection, calibration, or fitting.

Class order and confusion matrix orientation: Minor, Moderate, Major; rows are true labels and columns are predicted labels.

All probabilities below are calibrated with a scalar temperature fitted on validation only. ECE uses 15 equal-width bins.

## Pair Disjoint

Matched test support: 1802; pair-ID SHA-256 `c260601e9e8f4f7c5732f6fa1f42887e51cbf449dc09e6d15b853fafffe4e481`; class supports {'Minor': 122, 'Moderate': 1376, 'Major': 304}.

| Model | Validation macro-F1 | Test macro-F1 | AUROC OVR macro | Brier | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Text | 0.7602 | 0.7708 | 0.9333 | 0.1858 | 0.3361 | 0.0198 |
| Molecular + Text | 0.7707 | 0.7836 | 0.9323 | 0.1848 | 0.3377 | 0.0158 |
| Target + Text | 0.7581 | 0.7743 | 0.9379 | 0.1837 | 0.3298 | 0.0188 |
| Molecular + Target | 0.6021 | 0.6130 | 0.8527 | 0.3168 | 0.5333 | 0.0280 |
| Full Molecular + Target + Text | 0.7605 | 0.7473 | 0.9280 | 0.2177 | 0.3802 | 0.0163 |

Paired bootstrap for full model minus comparator (10,000 resamples; stratified within observed class supports):

| Comparator | Δ macro-F1 | 95% CI |
|---|---:|---:|
| Molecular + Text | -0.0363 | [-0.0602, -0.0121] |
| Target + Text | -0.0270 | [-0.0495, -0.0041] |
| Text | -0.0235 | [-0.0465, +0.0010] |
| Molecular + Target | +0.1343 | [+0.1035, +0.1662] |

Per-class calibrated test metrics (precision / recall / F1 / support):

| Model | Minor | Moderate | Major |
|---|---|---|---|
| Text | 0.640 / 0.582 / 0.609 / 122 | 0.923 / 0.932 / 0.927 / 1376 | 0.778 / 0.773 / 0.776 / 304 |
| Molecular + Text | 0.701 / 0.615 / 0.655 / 122 | 0.925 / 0.928 / 0.927 / 1376 | 0.756 / 0.783 / 0.769 / 304 |
| Target + Text | 0.735 / 0.615 / 0.670 / 122 | 0.916 / 0.928 / 0.922 / 1376 | 0.729 / 0.734 / 0.731 / 304 |
| Molecular + Target | 0.391 / 0.500 / 0.439 / 122 | 0.864 / 0.842 / 0.853 / 1376 | 0.546 / 0.549 / 0.548 / 304 |
| Full Molecular + Target + Text | 0.559 / 0.697 / 0.620 / 122 | 0.932 / 0.879 / 0.905 / 1376 | 0.668 / 0.773 / 0.716 / 304 |

Confusion matrices (true rows × predicted columns; [Minor, Moderate, Major]):

- **Text:** `[[71, 42, 9], [36, 1282, 58], [4, 65, 235]]`
- **Molecular + Text:** `[[75, 42, 5], [27, 1277, 72], [5, 61, 238]]`
- **Target + Text:** `[[75, 39, 8], [24, 1277, 75], [3, 78, 223]]`
- **Molecular + Target:** `[[61, 54, 7], [86, 1158, 132], [9, 128, 167]]`
- **Full Molecular + Target + Text:** `[[85, 25, 12], [61, 1210, 105], [6, 63, 235]]`

Artifact-integrity checks passed for evaluation, prediction, checkpoint, split IDs, common-support manifest, checkpoint reload flag, and protocol-specific text feature manifest hash. The cold-drug evaluation records declare the generic text-manifest path, whose current bytes are pair-disjoint scoped, while recording the cold-drug manifest hash. The cold-drug feature hash was independently verified against the cold-drug train-fit manifest; original evaluation artifacts were preserved. Pair-disjoint generic and specific manifest bytes match.

## Cold Drug

Matched test support: 253; pair-ID SHA-256 `8c1e93c8eed2083b081d3583d4fc4bdfdd5617cbdbf452305926a67de77da087`; class supports {'Minor': 14, 'Moderate': 206, 'Major': 33}.

| Model | Validation macro-F1 | Test macro-F1 | AUROC OVR macro | Brier | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Text | 0.4579 | 0.3743 | 0.5403 | 0.4080 | 0.7254 | 0.1185 |
| Molecular + Text | 0.3808 | 0.3986 | 0.6097 | 0.3675 | 0.6472 | 0.0579 |
| Target + Text | 0.4088 | 0.3556 | 0.5431 | 0.3838 | 0.6728 | 0.0832 |
| Molecular + Target | 0.3042 | 0.2821 | 0.4286 | 0.5803 | 0.9515 | 0.0674 |
| Full Molecular + Target + Text | 0.3989 | 0.3664 | 0.5409 | 0.3624 | 0.6550 | 0.0708 |

Paired bootstrap for full model minus comparator (10,000 resamples; stratified within observed class supports):

| Comparator | Δ macro-F1 | 95% CI |
|---|---:|---:|
| Molecular + Text | -0.0322 | [-0.1000, +0.0231] |
| Target + Text | +0.0107 | [-0.0058, +0.0298] |
| Text | -0.0080 | [-0.0429, +0.0181] |
| Molecular + Target | +0.0843 | [-0.0013, +0.1750] |

Per-class calibrated test metrics (precision / recall / F1 / support):

| Model | Minor | Moderate | Major |
|---|---|---|---|
| Text | 0.250 / 0.143 / 0.182 / 14 | 0.813 / 0.908 / 0.858 / 206 | 0.133 / 0.061 / 0.083 / 33 |
| Molecular + Text | 0.200 / 0.214 / 0.207 / 14 | 0.822 / 0.898 / 0.858 / 206 | 0.231 / 0.091 / 0.130 / 33 |
| Target + Text | 0.200 / 0.143 / 0.167 / 14 | 0.810 / 0.908 / 0.856 / 206 | 0.083 / 0.030 / 0.044 / 33 |
| Molecular + Target | 0.000 / 0.000 / 0.000 / 14 | 0.810 / 0.621 / 0.703 / 206 | 0.108 / 0.212 / 0.143 / 33 |
| Full Molecular + Target + Text | 0.222 / 0.143 / 0.174 / 14 | 0.815 / 0.942 / 0.874 / 206 | 0.167 / 0.030 / 0.051 / 33 |

Confusion matrices (true rows × predicted columns; [Minor, Moderate, Major]):

- **Text:** `[[2, 12, 0], [6, 187, 13], [0, 31, 2]]`
- **Molecular + Text:** `[[3, 10, 1], [12, 185, 9], [0, 30, 3]]`
- **Target + Text:** `[[2, 12, 0], [8, 187, 11], [0, 32, 1]]`
- **Molecular + Target:** `[[0, 11, 3], [23, 128, 55], [7, 19, 7]]`
- **Full Molecular + Target + Text:** `[[2, 12, 0], [7, 194, 5], [0, 32, 1]]`

Artifact-integrity checks passed for evaluation, prediction, checkpoint, split IDs, common-support manifest, checkpoint reload flag, and protocol-specific text feature manifest hash. The cold-drug evaluation records declare the generic text-manifest path, whose current bytes are pair-disjoint scoped, while recording the cold-drug manifest hash. The cold-drug feature hash was independently verified against the cold-drug train-fit manifest; original evaluation artifacts were preserved. Pair-disjoint generic and specific manifest bytes match.

## DailyMed rate-limit audit

The preserved ingestion audit reports 1939 attempted IDs, 652 covered drugs, 879 passages, and 403 errors: {'<urlopen error _ssl.c:1015: The handshake operation timed out>': 2, 'The read operation timed out': 6, '[Errno 54] Connection reset by peer': 2, 'HTTP Error 429: Too Many Requests': 393}.

No retries were sent during this analysis. For a future, separately staged retry, the recommended conservative client policy is to retry failed IDs only; honor Retry-After; otherwise use full-jitter exponential backoff starting at 30 s, capped at 10 min, no more than five retries, global concurrency one and at most one request/s; stop after three consecutive 429s and defer the remainder. Record each attempt and response provenance in an append-only shard, verify hashes, and build a staged merged corpus/manifest without overwriting the existing corpus. These are operational recommendations, not a claimed NLM quota.

## Interpretation and limitations

The cold-drug matched test contains only 253 pairs: 14 Minor, 206 Moderate, and 33 Major. The 14-pair Minor support makes its precision/recall/F1 highly unstable; a stratified bootstrap preserves n=14 but does not create additional independent Minor examples. Cold-drug performance is materially below pair-disjoint performance, especially on Major and Minor classes. Results are descriptive of the fixed split and limited feature coverage. This analysis did not inspect or alter held-out pairs to tune models. It does not establish clinical validity.

The prediction approval gate remains closed (`approved=false`, `approval_status=blocked`). The existing approval manifest contains stale blocked reasons that say ChEMBL, text evidence, and full multimodal training are absent. That approval file was left unchanged; the API must remain disabled until the project's full approval criteria are reviewed and satisfied.

Reproduce: `python scripts/paired_multimodal_validation.py`
