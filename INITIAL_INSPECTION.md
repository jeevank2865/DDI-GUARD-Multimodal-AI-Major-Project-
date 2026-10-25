# Initial repository and data inspection

Date: 2026-10-20

## Repository state

- The project workspace was created as the requested writable work area during the initial inspection.
- An unrelated nearby checkout contained only Git metadata at the time; it was not treated as the project repository. The project implementation was developed in this workspace.
- Since there is no code baseline, no source-code checkpoint could be made before implementation. Original source datasets remain in place and are read only.

## Local data inventory

- DDInter 2.0 source folder contains all eight expected ATC CSVs: A, B, D, H, L, P, R, V.
- Read-only CSV audit counted 222,383 rows. Canonicalizing unordered DDInter ID pairs gives 160,235 unique pairs: 130,422 with a valid severity and 29,813 labeled `Unknown`.
- The 130,422 valid pairs have counts Minor 6,833, Moderate 96,675, Major 26,914. No self-pairs or conflicting severity labels were found in the canonical ID-pair audit.
- PubChem structures are not present locally. Initial sandbox name resolution failed for the official PUG REST endpoint; a user-approved external request succeeded and a resumable enrichment job was started.
- The local ChEMBL folder contains ChEMBL 37 sequence files and a 314 MB HDF5 file, not a verified ChEMBL 34 activity database. `h5py` is not installed, and no ChEMBL drug-target edge table is present.
- Runtime audit: Python 3.13.12, PyTorch 2.11.0, PyTorch Geometric 2.7.0, RDKit 2026.3.1, scikit-learn 1.8.0, FastAPI 0.136.1; 8 CPU cores, 8 GiB RAM, no CUDA GPU. `h5py`/PyTables are absent.

## Initial implementation boundary

The first verified deliverable is a lossless DDInter ingestion/audit pipeline and trainable model components with gradient tests. Full molecular/target enrichment, independent model evaluation, serving, and UI work require authentic molecular and ChEMBL activity data plus a real application baseline. No metrics or model-training results will be claimed until those are run on such data.
