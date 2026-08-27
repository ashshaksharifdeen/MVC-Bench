# MVC-Bench repository structure and execution guide

MVC-Bench follows a MaPLe/Dassl-style experiment architecture. The shell launcher selects datasets, trainers, backbones, seeds, and output locations; `train.py` merges command-line and YAML configuration; dataset classes build the splits; trainers construct the backbone and prompt learner; and evaluation code writes accuracy and calibration metrics.

Because backbone packages have incompatible dependency stacks, the same source branch should be checked out separately for the BAPLe and DAC environments.

## Physical workspace structure

```text
workspace/
├── MVC-Bench-main/         # branch: main; env: maple
├── MVC-Bench-histo-xray/   # branch: histo-xray; env: baple & dac
├── data/
│   ├── dr/raw/
│   └── med-datasets/
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

### Main branch: general CLIP

```bash
conda activate maple
cd /absolute/path/to/MVC-Bench-main
```

Use for CLIP ViT-B/16, ViT-B/32, RN50, and RN101. The branch is based on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning).

### Histo-Xray branch: MedCLIP/BioMedCLIP

```bash
conda activate baple
cd /absolute/path/to/MVC-Bench-histo-xray
```

Use the [BAPLe](https://github.com/asif-hanif/baple) environment. Verify the launcher/configuration explicitly selects the intended model.

### Histo-Xray branch: PLIP/QuiltNet

```bash
conda activate dac
cd /absolute/path/to/MVC-Bench-histo-xray
```

Use the [DAC/CLIP_Calibration](https://github.com/ml-stat-Sustech/CLIP_Calibration) environment. Verify the launcher/configuration explicitly selects the intended model.


