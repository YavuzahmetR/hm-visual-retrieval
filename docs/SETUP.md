# Environment and local assets

Tested with Windows, Python 3.12.13, PyTorch 2.6.0+cu124,
Torchvision 0.21.0+cu124, Streamlit 1.65.0 and an RTX 3050 Laptop GPU (4 GB).
The existing training/extraction scripts require CUDA; demo inference also
supports CPU.

For a new environment:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
./.venv/Scripts/python.exe -m pip install -r requirements.txt
cp project_config.example.json project_config.json
```

Point `data_root` to the existing images, laid out as
`<data_root>/images/010/0108775015.jpg`. For this completed local project, keep the
current environment and extracted data; raw CSVs and the ZIP need not be
downloaded again. `requirements-lock.txt` records the development environment,
including notebook/development dependencies beyond the runtime requirements. Automated test/lint tools were removed on 2026-10-09; it is a filtered historical environment record, not a minimal runtime lock. The untouched original is recoverable from the source commit.

The demo needs these local assets:

- `artifacts/subset_manifest.csv` and the configured image directory.
- `model_cache/hub/checkpoints/resnet18-f37072fd.pth`.
- `models/projection_head_robust.pt` and, for comparison,
  `models/projection_head_epoch15.pt`.
- `results/test_evaluation/`: frozen 512D features, original head's 128D
  features, row CSV, cache metadata and original metrics.
- `results/robust_head_v1/test_evaluation/`: selected head's 128D features,
  row CSV and completed metrics.

Hashes, dimensions, ID order and encoder/preprocessing metadata are checked.
The demo does not download weights or rebuild missing galleries. Dataset images,
manifest and large caches are excluded from Git; a fresh checkout alone cannot
run the local demo.

For data preparation on a new machine, run only code cells 2 and 4 of
`notebooks/kaggle_data_preparation.ipynb` on Kaggle with the competition data.
Download `subset_manifest.csv` into `artifacts/`. Obtain the official local
ResNet weights using:

```bash
./.venv/Scripts/python.exe -c "import torch; from torchvision.models import ResNet18_Weights; torch.hub.set_dir('model_cache/hub'); ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"
```
