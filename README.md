# MVC-Bench

Official implementation of **MVC-Bench: Benchmarking Calibration of Medical Vision-Language Models**, accepted to Findings of EMNLP 2026.

MVC-Bench is a calibration-centric benchmark for medical image classification with vision-language models (VLMs) and medical VLMs. It evaluates whether predictive confidence remains reliable across backbones, medical modalities, domain shifts, prompt-tuning methods, hard-prompt templates, and random seeds.

> **This branch is the general-backbone branch.** Use it for OpenAI CLIP ViT-B/16, ViT-B/32, RN50, and RN101 in the MaPLe environment. For MedCLIP, BioMedCLIP, PLIP, or QuiltNet, use the [`histo-xray`](https://github.com/ashshaksharifdeen/MVC-Bench/tree/histo-xray) branch and its separate environment instructions.

## Authors

| Author | Affiliation | Google Scholar |
|---|---|---|
| Ashshak Sharifdeen | Mohamed bin Zayed University of Artificial Intelligence | [Profile](https://scholar.google.com/citations?hl=en&user=rd9zSX8AAAAJ) |
| Shihab Aaqil Ahamed | Mohamed bin Zayed University of Artificial Intelligence | [Profile](https://scholar.google.com/citations?hl=en&user=r9-fRcEAAAAJ) |
| Ufaq Khan | Mohamed bin Zayed University of Artificial Intelligence | [Profile](https://scholar.google.com/citations?hl=en&user=7gHFVw4AAAAJ) |
| Muhammad Akhtar Munir | Mohamed bin Zayed University of Artificial Intelligence | [Profile](https://scholar.google.com/citations?hl=en&user=sT-epZAAAAAJ) |
| Sujair Ibrahim | Sabaragamuwa University of Sri Lanka | [Profile](https://scholar.google.com/citations?hl=en&user=k8-yPXAAAAAJ) |
| Mohamed Rafeek Mareer Ahamed | Digital Platform Development, SLT PLC | [Profile](https://scholar.google.com/citations?hl=en&user=uTLtZYoAAAAJ) |
| Yutong Xie | Mohamed bin Zayed University of Artificial Intelligence | [Profile](https://scholar.google.com/citations?hl=en&user=ddDL9HMAAAAJ) |
| Imran Razzak | Mohamed bin Zayed University of Artificial Intelligence | [Profile](https://scholar.google.com/citations?user=GlXI4N8AAAAJ) |
| Muhammad Haris Khan | Mohamed bin Zayed University of Artificial Intelligence | [Profile](https://scholar.google.com/citations?hl=en&user=ZgERfFwAAAAJ) |

## What MVC-Bench evaluates

MVC-Bench organizes calibration evaluation along three axes:

1. **Robustness** to backbone, modality, and in-domain (ID) to domain-shift (DS) transfer.
2. **Effectiveness** of post-hoc, train-time, zero-shot calibration, and prompt-tuning strategies.
3. **Stability** under hard-prompt choice and random-seed variation.

The benchmark contains more than **1,638 controlled experiments** across:

| Component | Coverage |
|---|---|
| Modalities | Diabetic-retinopathy fundus images, histopathology, and chest X-ray |
| General VLMs | CLIP ViT-B/16, ViT-B/32, RN50, RN101 |
| Medical VLMs | PLIP, QuiltNet, MedCLIP, BioMedCLIP |
| Prompt tuning | CoOp, KgCoOp, MaPLe, PromptSRC, ProGrad, HiCroPL |
| Calibration | Base, zero-shot, MDCA, label smoothing, MBLS, ECCV-ZS, ECCV-Penalty, temperature scaling, and MCM |
| Primary metrics | Accuracy and 20-bin Expected Calibration Error (ECE) |
| Complementary metrics | Maximum Calibration Error (MCE) and Adaptive Calibration Error (ACE) |

## Main findings

1. **In-domain ranking does not reliably predict domain-shift ranking.** Across 30 ID-to-DS Kendall rank tests, mean Kendall's tau-b is `-0.0598` and median tau-b is `-0.0476`; 17 correlations are negative and no significantly positive correlation is observed at `p < 0.05`.
2. **Medical pretraining often improves calibration, but architecture alone is insufficient.** BioMedCLIP generally calibrates chest X-rays better than MedCLIP; pretraining scale, diversity, and label alignment also matter.
3. **Accuracy and calibration are method- and backbone-dependent.** MDCA, and MBLS often show a favorable negative accuracy-ECE relationship, while ECCV-Penalty tends to show a positive relationship. There is no universal accuracy-calibration trade-off.
4. **No calibration method wins in every ID and DS setting.** Base, MBLS, and MDCA are relatively consistent in-domain, while domain shift frequently increases ECE. Some zero-shot penalties help particular shifts but do not generalize universally.
5. **Prompt-tuning accuracy is not a proxy for calibration.** HiCroPL gives the best reported balance (`75.50%` accuracy, `3.55%` ECE), while PromptSRC attains high accuracy but `11.52%` ECE and ProGrad reaches `23.08%` ECE.
6. **Hard-prompt wording changes calibration.** Generic-to-moderately descriptive templates are more stable than terse label-style prompts. The generic “a photo of a” template has mean ECE `4.14` and standard deviation `2.05`; “The DR level is” has mean ECE `5.29` and standard deviation `4.21`.
7. **Seed sensitivity is backbone-dependent.** BioMedCLIP on DR is stable (`0.25` ECE standard deviation), whereas MedCLIP on DR is much more variable (`2.49`). Report multiple seeds rather than a single run.

## Multi-Class Margin regularization

The paper proposes Multi-Class Margin (MCM) regularization. For sample `i`, true class `y_i`, and non-target class `k`, the true-versus-rest logit margin is

```math
m_{i,k} = z_{i,y_i} - z_{i,k}.
```

MCM increases the batch-level mean margin while controlling all-pairs margin dispersion:

```math
L_{MCM} = -\alpha\mu + \beta\sigma^2_{all-pairs},
```

with total objective

```math
L_{total} = L_{CE} + L_{MCM}.
```

The paper uses `alpha = 0.1` and `beta = 0.01`, selected using validation data only. MCM achieves the lowest ID ECE in 10 of 12 reported settings and remains competitive, but is not uniformly best under domain shift. Its intended use is low-margin underconfidence; it should not be treated as a universal correction for already-overconfident models.

## Branch and environment matrix

| Backbone | Branch | Required environment | Runner |
|---|---|---|---|
| CLIP ViT-B/16, ViT-B/32, RN50, RN101 | `main` | `maple` | `bash scripts/coop/base2new_train_coop_datasets.sh` |

Do not install all three dependency stacks into one Conda environment. PyTorch, Python, OpenCLIP, and Dassl versions differ across the upstream projects.

## Installation: general CLIP backbones

The main branch follows the official [MaPLe installation](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/INSTALL.md). 

### 1. Clone the main branch

```bash
git clone --branch main --single-branch \
  https://github.com/ashshaksharifdeen/MVC-Bench.git MVC-Bench-main
cd MVC-Bench-main
```

### 2. Create and activate the MaPLe environment

```bash
conda create -y -n maple python=3.8
conda activate maple

python -m pip install \
  torch==1.9.0+cu111 \
  torchvision==0.10.0+cu111 \
  torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

If the CUDA 11.1 wheel is incompatible with the machine, select a mutually compatible PyTorch/torchvision/CUDA combination and record the change. Do not silently mix versions.

### 3. Install Dassl

From a directory alongside the MVC-Bench checkout:

```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
python -m pip install -r requirements.txt
python setup.py develop
cd ../MVC-Bench-main
```
then replace the ```/Dassl.pytorch/dassl/evaluation/evaluator.py``` with our python file

### 4. Install MVC-Bench requirements

```bash
python -m pip install -r requirements.txt
python -m pip install setuptools==59.5.0
```

### 5. Verify the environment

```bash
python - <<'PY'
import torch
import torchvision
import dassl
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("Dassl import: OK")
PY
```


## Dataset preparation: diabetic retinopathy

The DR setup follows [SPSD-ViT](https://github.com/Chumsy0725/SPSD-ViT) for source acquisition and domain organization. See [`docs/DATASET_DR.md`](docs/DATASET_DR.md) for licensing, source-layout preservation, MVC-Bench conversion, validation, and the exact ID/DS protocol.

The benchmark uses:

| Role | Dataset/domain |
|---|---|
| In-domain | Messidor, with class 4 excluded by the experiment split/configuration |
| Domain shift | APTOS, EyePACS, Messidor-2 |

Do not delete Messidor class 4 from the raw archive. Exclude it only in the processed split or the branch configuration so the source data remain recoverable.

Set the dataset-root variable used by `scripts/coop/base2new_train_coop_datasets.sh` and the YAML files under `configs/datasets/` to the processed DR root. Use absolute paths when possible.

## Running the general-backbone experiments

Activate the correct environment and move to the main-branch checkout:

```bash
conda activate maple
cd /absolute/path/to/MVC-Bench-main
bash scripts/coop/base2new_train_coop_datasets.sh
```

Run the command from the repository root. The launcher belongs under `scripts/coop/`, following the MaPLe method-grouped script hierarchy; it should not be stored at repository root. See [`docs/MAIN_BRANCH_FILE_PLACEMENT.md`](docs/MAIN_BRANCH_FILE_PLACEMENT.md) for the exact move and documentation-copy commands.

Before launching the full sweep, inspect the script and verify:

- the dataset root;
- dataset/config names;
- backbone (`ViT-B/16`, `ViT-B/32`, `RN50`, or `RN101`);
- trainer and calibration method;
- `N_CTX`, number of shots, seed list, output directory, and GPU selection;
- base-to-new or ID-to-DS split semantics expected by the experiment.

Paper-aligned defaults are 16 shots for CoOp, learning rate `0.005`, and seeds `1`, `2`, and `3`. Treat the checked-in script/configuration as the executable source of truth and report any deliberate changes.

For a smoke test, temporarily select one dataset, one backbone, and one seed. Check that training, checkpoint saving, test evaluation, and metric reporting all complete before starting the full grid.

## Prompt-tuning implementations

MVC-Bench integrates or adapts the following methods. Use the implementation checked into this repository for benchmark comparability; the links below provide attribution and method-specific background.

| Method | Official repository |
|---|---|
| CoOp | [KaiyangZhou/CoOp](https://github.com/KaiyangZhou/CoOp) |
| KgCoOp | [htyao89/KgCoOp](https://github.com/htyao89/KgCoOp) |
| MaPLe | [muzairkhattak/multimodal-prompt-learning](https://github.com/muzairkhattak/multimodal-prompt-learning) |
| PromptSRC | [muzairkhattak/PromptSRC](https://github.com/muzairkhattak/PromptSRC) |
| ProGrad | [BeierZhu/Prompt-align](https://github.com/BeierZhu/Prompt-align) |
| HiCroPL | [zzeoZheng/HiCroPL](https://github.com/zzeoZheng/HiCroPL) |

Do not install each upstream repository into the same environment unless developing those projects directly. MVC-Bench's integrated trainers and branch requirements should control the benchmark runtime.

## Calibration methods and reporting

For every run, report at least:

- accuracy;
- 20-bin ECE;
- seed and backbone;
- dataset, ID/DS role, prompt-tuning method, and calibration method;
- MCE and ACE when reproducing the supplementary evaluation.


Aggregate the three seeds using mean and standard deviation. A result directory should be uniquely identifiable from dataset, trainer, backbone, calibration method, shot count, and seed. See [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md) for a recommended naming scheme and the configuration-to-execution flow.

## Reproducibility checklist

- [ ] Correct branch and Conda environment are active.
- [ ] Dataset licenses and access requirements were followed.
- [ ] Raw data are immutable; processed splits are versioned separately.
- [ ] Messidor class 4 is excluded by split/config, not destructively deleted.
- [ ] The same class-name order is used by the dataset loader and text prompts.
- [ ] 16-shot sampling and seeds 1/2/3 match the paper.
- [ ] Calibration hyperparameters are selected on validation data only.
- [ ] ID and DS results are written to separate, unambiguous directories.
- [ ] Accuracy, ECE, MCE, and ACE implementations are recorded.
- [ ] Environment, GPU, commit hash, and command are saved with results.

## Repository structure

MVC-Bench follows the MaPLe/Dassl configuration pattern: dataset registration, trainer selection, YAML configuration, a central `train.py` entry point, and shell launchers for experiment grids. A detailed file-role map and debugging flow are provided in [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md).

## Citation

Please cite the paper if MVC-Bench is useful in your work. Confirm the final ACL Anthology metadata before publication:

```bibtex
@inproceedings{
anonymous2026mvcbench,
title={{MVC}-Bench: Benchmarking Calibration of Medical Vision-Language Models},
author={Anonymous},
booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing},
year={2026},
url={https://openreview.net/forum?id=4DDcig293B}
}
```

## Acknowledgements

This codebase builds on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), and the prompt-learning repositories listed above. DR dataset preparation follows [SPSD-ViT](https://github.com/Chumsy0725/SPSD-ViT). Please also cite the original datasets, backbones, and methods used in each experiment.
