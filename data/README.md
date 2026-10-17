# Data

The downloader references ds000002, ds000011, ds000017 and ds000052. Obtain each dataset under its original terms and verify its task metadata before assigning labels. Historical descriptions in this repository have not been reconciled with the source datasets.

`python scripts/download.py` downloads and extracts raw archives into `data/raw/`. It does not produce the `.npy` files expected by `DatasetManager` in `data/processed/`.

The volume preprocessing helper in `learnedSpectrum/data.py` resizes and normalizes volumes. It does not perform motion correction, slice-timing correction or atlas-based region extraction. A complete, verified raw-to-training workflow is not supplied.

The current train/validation/test split operates on files. A new evaluation must define and verify participant grouping and task labels before interpreting its results.
