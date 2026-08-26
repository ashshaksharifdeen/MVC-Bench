# MVC-Bench repository structure and execution guide

MVC-Bench follows a MaPLe/Dassl-style experiment architecture. The shell launcher selects datasets, trainers, backbones, seeds, and output locations; `train.py` merges command-line and YAML configuration; dataset classes build the splits; trainers construct the backbone and prompt learner; and evaluation code writes accuracy and calibration metrics.

Because backbone packages have incompatible dependency stacks, the same source branch should be checked out separately for the BAPLe and DAC environments.

## Physical workspace structure

```text
workspace/
├── MVC-Bench-main/                 # branch: main; env: maple
├── MVC-Bench-medclip-biomedclip/   # branch: histo-xray; env: mvc-baple
├── MVC-Bench-plip-quiltnet/        # branch: histo-xray; env: mvc-dac
├── data/
│   ├── dr/raw/
│   ├── dr/processed/
│   └── med-datasets/
├── model-weights/
└── outputs/
    ├── general-clip/
    ├── medclip-biomedclip/
    └── plip-quiltnet/
```

Benefits of this layout:

- each checkout has one unambiguous active environment;
- model-specific dependencies and generated caches cannot overwrite one another;
- datasets and weights are stored once;
- result paths identify the backbone family;
- branch changes can be tested independently.

## Logical repository map

The exact checked-in filenames are authoritative, but a MaPLe-derived MVC-Bench tree has the following roles:

```text
MVC-Bench/
├── clip/                         # general CLIP implementation/tokenizer
├── lpclip/                       # optional linear-probe/CLIP helpers
├── datasets/                     # dataset registry and dataset classes
├── trainers/                     # prompt and calibration trainers
├── configs/
│   ├── datasets/                 # dataset roots, class/split settings
│   └── trainers/                 # trainer, optimizer, prompt, loss settings
├── scripts/                      # experiment launchers/helpers
│   ├── coop/                     # MaPLe-style CoOp launchers on main
│   │   └── base2new_train_coop_datasets.sh
│   
│   
├── med-vlms/                     # medical-backbone adapters, if vendored
│   ├── medclip/
│   ├── biomedclip/
│   ├── plip/
│   └── quiltnet/
├── train.py                      # central train/evaluate entry point
├── parse_test_res.py             # aggregate experiment logs, if retained
└── requirements.txt
```

Some paths may exist only on one branch. Do not copy medical-backbone modules into `main` merely to make the trees identical.

## File responsibilities

| Area | Responsibility | What to verify when adding an experiment |
|---|---|---|
| `datasets/` | Register dataset names, load paths/labels, create train/val/test and few-shot splits | Class order, ID/DS role, cache keys, leakage prevention |
| `configs/datasets/` | Dataset roots and dataset-specific options | Root portability, split/filter settings, matching registry name |
| `trainers/` | Prompt learner, backbone invocation, losses, optimization, evaluation hooks | Frozen/trainable parameters, logits, MCM/calibration loss, checkpoint keys |
| `configs/trainers/` | Prompt depth/context, optimizer, shots, epochs, calibration parameters | Paper-aligned values and validation-only tuning |
| `clip/` | General CLIP loading, tokenization, preprocessing | Backbone name, input resolution, mean/std, tokenizer |
| `med-vlms/` or adapters | MedCLIP/BioMedCLIP/PLIP/QuiltNet loading | Weight path, package version, output-logit convention, transform |
| `scripts/` | Experiment grids and command construction | Environment, paths, quoting, dataset/backbone/seed loops, output uniqueness |
| `train.py` | Parse arguments, merge configuration, seed runtime, dispatch trainer | CLI precedence, eval-only/resume behavior, deterministic flags |
| result parser | Summarize completed logs | Missing-run detection, metric names, mean/std, ID/DS separation |

## Configuration-to-result flow

```text
shell launcher
  -> dataset YAML + trainer YAML + command-line overrides
  -> train.py / Dassl configuration merge
  -> dataset registry and split/cache construction
  -> backbone adapter + tokenizer + preprocessing
  -> prompt learner and calibration objective
  -> checkpoint
  -> test logits/probabilities
  -> accuracy, ECE, MCE, ACE
  -> seed aggregation and ID/DS tables
```

When a result looks wrong, trace this flow from left to right. Most reproducibility failures come from a path/config override, stale split cache, class-order mismatch, wrong model preprocessing, or non-unique output directory.

## Branch-specific entry points

### Main branch: general CLIP

```bash
conda activate maple
cd /absolute/path/to/MVC-Bench-main
bash scripts/coop/base2new_train_coop_datasets.sh
```

Use for CLIP ViT-B/16, ViT-B/32, RN50, and RN101. The branch is based on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning).

### Histo-Xray branch: MedCLIP/BioMedCLIP

```bash
conda activate mvc-baple
cd /absolute/path/to/MVC-Bench-medclip-biomedclip
bash scripts/all_fewshot_medclip_new.sh
```

Use the [BAPLe](https://github.com/asif-hanif/baple) environment. Verify the launcher/configuration explicitly selects the intended model.

### Histo-Xray branch: PLIP/QuiltNet

```bash
conda activate mvc-dac
cd /absolute/path/to/MVC-Bench-plip-quiltnet
bash scripts/all_fewshot_plip_new.sh
```

Use the [DAC/CLIP_Calibration](https://github.com/ml-stat-Sustech/CLIP_Calibration) environment. Verify the launcher/configuration explicitly selects the intended model.


