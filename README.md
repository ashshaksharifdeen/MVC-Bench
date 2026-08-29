# MVC-Bench: histopathology and chest X-ray branch

This branch contains the MVC-Bench experiments for histopathology and chest X-ray datasets, including modality-specific medical vision-language backbones.

> **Environment isolation is required.** Use the BAPLe-based environment for MedCLIP/BioMedCLIP and the DAC-based environment for PLIP/QuiltNet. Keep them in separate Conda environments and preferably separate checkouts of this branch. Use the `main` branch and MaPLe environment for general CLIP backbones.

For the benchmark motivation, findings, complete author list, and citation, see the [`main` branch README](https://github.com/ashshaksharifdeen/MVC-Bench).

## Supported setup

| Backbone family | Checkout | Environment | Launcher |
|---|---|---|---|
| MedCLIP, BioMedCLIP | `MVC-Bench-medclip-biomedclip` | `mvc-baple` | `bash scripts/all_fewshot_medclip_new.sh` |
| PLIP, QuiltNet | `MVC-Bench-plip-quiltnet` | `mvc-dac` | `bash scripts/all_fewshot_plip_new.sh` |
| CLIP ViT-B/16, ViT-B/32, RN50, RN101 | `MVC-Bench-main` | `maple` | `bash scripts/coop/base2new_train_coop_datasets.sh` |

Do not run a backbone with the wrong environment merely because its module imports successfully. The upstream projects pin different Python, PyTorch, CUDA, OpenCLIP, and model-package versions.


## Environment A: MedCLIP and BioMedCLIP

This environment follows [BAPLe](https://github.com/asif-hanif/baple), which uses Python 3.8 and supplies a `setup_env.sh` installer.

### 1. Create the environment

```bash
conda create -y -n baple python=3.8
conda activate baple
```

### 2. Install the BAPLe stack

Use a separate dependency checkout:

```bash
git clone https://github.com/asif-hanif/baple.git baple-upstream
cd baple-upstream
bash setup_env.sh
cd ../MVC-Bench-medclip-biomedclip
```

If this branch includes additional requirements, install them after the upstream setup:

```bash
python -m pip install -r requirements.txt
```

Do not re-run `setup_env.sh` in another active environment. Save a frozen environment record after a successful smoke test:

```bash
conda env export --no-builds > environment-mvc-baple.yml
python -m pip freeze > requirements-mvc-baple-lock.txt
```

### 3. Add model weights

Follow [CalibPrompt's medical-model guide](https://github.com/iabh1shekbasu/CalibPrompt/blob/main/docs/MODELS.md) and the model licenses.

Expected locations beneath the configured `MODEL_ROOT` are:

```text
model-weights/
├── medclip/
│   └── pretrained/
│       └── medclip-vit/
│           └── pytorch_model.bin
└── biomedclip/
```

BioMedCLIP is commonly loaded through OpenCLIP from `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`; the first run may download files into the Hugging Face cache. Pre-download the model on compute nodes without internet access, keep the license notices, and point the branch configuration to the cache/model root actually used.

### 4. Verify the model environment

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
PY
```

Also run the branch's model-construction path for each backbone before training. A successful `import` is insufficient; verify tokenizer creation, weight loading, device placement, and one forward pass.

## Environment B: PLIP and QuiltNet

This environment follows the [DAC/CLIP_Calibration installation](https://github.com/ml-stat-Sustech/CLIP_Calibration/blob/main/docs/INSTALL.md): Ubuntu 20.04, Python 3.10, PyTorch 2.1.0, torchvision 0.16.0, torchaudio 2.1.0, and CUDA 12.1 wheels.

### 1. Create the environment and install PyTorch

```bash
conda create -y -n dac python=3.10
conda activate dac

python -m pip install \
  torch==2.1.0 \
  torchvision==0.16.0 \
  torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Dassl

```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
python -m pip install -r requirements.txt
python setup.py develop
cd ../MVC-Bench-plip-quiltnet
```

### 3. Install the branch requirements

```bash
python -m pip install -r requirements.txt
python -m pip install setuptools==59.5.0
```

### 4. Add model weights

Following [CalibPrompt's model layout](https://github.com/iabh1shekbasu/CalibPrompt/blob/main/docs/MODELS.md):

```text
model-weights/
├── plip/
│   └── plip_vit_b32.pt
└── quiltnet/
    └── quiltnet_b32.pt
```

## Dataset preparation

Dataset download and preprocessing follow [CalibPrompt](https://github.com/iabh1shekbasu/CalibPrompt), especially its [dataset guide](https://github.com/iabh1shekbasu/CalibPrompt/blob/main/docs/DATASETS.md), with the preprocessing utilities documented in [BAPLe](https://github.com/asif-hanif/baple/blob/main/datasets/DATASETS.md).

| Modality | Dataset | Benchmark role |
|---|---|---|
| Chest X-ray | RSNA18 | In-domain |
| Chest X-ray | COVIDX/COVID radiography | Domain shift |
| Histopathology | PanNuke | In-domain |
| Histopathology | DigestPath | Domain shift |
| Histopathology | Kather | Domain shift |


Each dataset directory also needs `classnames.txt`. The first MVC-Bench/Dassl run may create `preprocessed.pkl` and `split_fewshot/shot_<k>-seed_<s>.pkl`. Do not share a cache between datasets or between incompatible label mappings.

See [`docs/DATASETS_HISTO_XRAY.md`](docs/DATASETS_HISTO_XRAY.md) for dataset-specific source archives, preprocessing scripts, dependencies, validation, and leakage checks.

## Run MedCLIP/BioMedCLIP experiments

```bash
conda activate baple
cd /absolute/path/to/MVC-Bench-histo-xray
bash run/classification/fewshot/all_fewshot_medclip_new.sh
```

Before the complete sweep, open `run/classification/fewshot/all_fewshot_medclip_new.sh` and verify:

- whether the current run selects MedCLIP or BioMedCLIP;
- `DATA_ROOT`, `MODEL_ROOT`, and output root;
- dataset list and ID/DS roles;
- trainer/calibration method;
- shot count, seeds, batch size, and GPU;
- checkpoint loading and evaluation flags.

If the script hard-codes MedCLIP, create a clearly named BioMedCLIP configuration or pass the supported backbone argument; do not assume the launcher switches models automatically.

## Run PLIP/QuiltNet experiments

```bash
conda activate dac
cd /absolute/path/to/MVC-Bench-histo-xray
bash run/classification/fewshot/all_fewshot_plip_new.sh
```

Run this command from the repository root. Before the complete sweep, open `run/classification/fewshot/all_fewshot_plip_new.sh` and verify the same path and experiment variables, especially whether the selected configuration is PLIP or QuiltNet. If the script hard-codes PLIP, use a separate QuiltNet configuration or the branch's supported backbone argument.

See [`docs/HISTO_XRAY_FILE_PLACEMENT.md`](docs/HISTO_XRAY_FILE_PLACEMENT.md) for the exact launcher-move and documentation-copy commands.

## Acknowledgements

Environment and data preparation are adapted from [BAPLe](https://github.com/asif-hanif/baple), [DAC/CLIP_Calibration](https://github.com/ml-stat-Sustech/CLIP_Calibration), and [CalibPrompt](https://github.com/iabh1shekbasu/CalibPrompt). Cite these repositories and the original MedCLIP, BioMedCLIP, PLIP, QuiltNet, and dataset publications when using their assets or code.
