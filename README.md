# Learned Spectrum

An experimental fMRI learning-stage classifier with temporal, causal and physiological-state modules.

## Setup

```sh
git clone https://github.com/Intempus-Inc/learnedSpectrum.git
cd learnedSpectrum
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The training code assumes CUDA. A CPU training path has not been verified. Dependencies are not locked to a reproduced environment.

## Code and data

- `learnedSpectrum/`: model, data processing, training and visualization modules.
- `notebooks/learnedSpectrum.ipynb`: original experiment workflow, with saved outputs cleared.
- `scripts/download.py`: downloader for ds000002, ds000011, ds000017 and ds000052.

Datasets must be obtained under their original terms. The downloader uses historical archive URLs; availability and downloaded sizes have not been rechecked. Running `python scripts/download.py` starts downloads for all four datasets.

## Status

This is research code, not a verified reproduction package. The notebook's evaluation call and expected metric fields differ from the current training module. Review those interfaces before running an experiment. There is no command-line training entry point. The downloader only extracts raw files; the notebook expects processed `.npy` inputs and does not run an end-to-end preprocessing step. Its data paths depend on the working directory.

Learning-stage labels are derived from task and file metadata. The current split operates on files, so it does not establish performance on held-out participants. No new performance or causal claim is made by this cleanup.

## License

Copyright 2024 Teddy Warner. See [LICENSE](LICENSE) for the existing attribution terms. External datasets retain their own terms.
