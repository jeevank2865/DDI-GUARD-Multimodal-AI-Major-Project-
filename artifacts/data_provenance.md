# Data provenance and licensing notes

## DDInter 2.0

The eight local ATC CSV exports are read without modification. DDInter's terms identify the data as CC BY-NC-SA 4.0 and state that the resource may be incomplete and is not guaranteed error-free. This project is for non-commercial scholarly research. Attribution and ShareAlike obligations must be considered before distributing derived dataset artifacts or integrating them into a broader release. See the [DDInter terms](https://ddinter.scbdd.com/terms/).

## PubChem

Structures are fetched from PubChem PUG REST and each accepted result records its CID, API URL, query name, retrieval time, returned synonyms, and RDKit validation. PubChem aggregates records from many contributors; reuse terms may be source-specific. The project does not treat all PubChem annotations as having one blanket license. Record-level source/license review remains necessary before redistribution of bulk auxiliary annotations. See [PubChem data sources](https://pubchem.ncbi.nlm.nih.gov/docs/data-sources) and [PubChem downloads](https://pubchem.ncbi.nlm.nih.gov/docs/downloads).

## ChEMBL 34

The official release README and LICENSE are preserved in `data/raw/chembl_34/`. The release identifies CC BY-SA 3.0 Unported and requests attribution including the ChEMBL resource URL and release `chembl_34`; the distribution should also cite the release's primary ChEMBL paper. The project preserves ChEMBL IDs and release provenance. The source checksum file is retained alongside the license. The release requests this attribution: “ChEMBL data is from https://www.ebi.ac.uk/chembl — version chembl_34.” Its primary citation is Mendez et al., *Nucleic Acids Research* 47(D1):D930–D940 (2019), DOI: 10.1093/nar/gky1075.

Required attribution text from the release is recorded in `data/raw/chembl_34/REQUIRED.ATTRIBUTION`. Do not remove it when redistributing ChEMBL-derived data.

## Combined data and redistribution

DDInter's non-commercial ShareAlike terms and ChEMBL's ShareAlike terms differ, and PubChem contributor terms vary. This repository preserves sources and provenance separately. Redistribution or commercial deployment of a combined dataset/model requires a separate license compatibility review; no such permission is inferred here.
