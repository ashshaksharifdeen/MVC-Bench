# MVC-Bench documentation pack

This directory contains branch-ready documentation for **MVC-Bench: Benchmarking Calibration of Medical Vision-Language Models**.

## Files

| File | Destination |
|---|---|
| [`README-main.md`](README-main.md) | Copy to `README.md` on the `main` branch |
| [`DATASET-DR.md`](DATASET-DR.md) | Copy to `docs/DATASET_DR.md` on the `main` branch |
| [`MAIN-BRANCH-FILE-PLACEMENT.md`](MAIN-BRANCH-FILE-PLACEMENT.md) | Copy to `docs/MAIN_BRANCH_FILE_PLACEMENT.md` on the `main` branch |
| [`README-histo-xray.md`](README-histo-xray.md) | Copy to `README.md` on the `histo-xray` branch |
| [`DATASETS-HISTO-XRAY.md`](DATASETS-HISTO-XRAY.md) | Copy to `docs/DATASETS_HISTO_XRAY.md` on the `histo-xray` branch |
| [`HISTO-XRAY-FILE-PLACEMENT.md`](HISTO-XRAY-FILE-PLACEMENT.md) | Copy to `docs/HISTO_XRAY_FILE_PLACEMENT.md` on the `histo-xray` branch |
| [`REPOSITORY-STRUCTURE.md`](REPOSITORY-STRUCTURE.md) | Copy to `docs/REPOSITORY_STRUCTURE.md` on both branches |

The two README files deliberately recommend separate repository checkouts and Conda environments for each backbone family. Do not combine the MaPLe, BAPLe, and DAC dependency stacks in one environment.

## Recommended deployment

```text
workspace/
├── MVC-Bench-main/                 # main branch; maple environment
├── MVC-Bench-medclip-biomedclip/   # histo-xray branch; mvc-baple environment
├── MVC-Bench-plip-quiltnet/        # histo-xray branch; mvc-dac environment
├── data/
│   ├── dr/
│   └── med-datasets/
└── model-weights/
```

Keeping datasets and checkpoints outside the Git repositories avoids duplicated large files. Point each branch's dataset/model configuration to these shared absolute paths.

## Before publishing

Run this short branch-specific check after copying the files:

1. Confirm the dataset names used in each shell script match the names in `configs/datasets/`.
2. Confirm every `DATA`, `DATA_ROOT`, `MODEL_ROOT`, `OUTPUT_DIR`, or similar variable in the shell scripts points to the intended machine.
3. Confirm `scripts/all_fewshot_medclip_new.sh` selects MedCLIP/BioMedCLIP and `scripts/all_fewshot_plip_new.sh` selects PLIP/QuiltNet in the current branch implementation.
4. Replace the provisional BibTeX venue metadata if the final ACL Anthology record differs.
5. Run one one-seed smoke test before launching all seeds and domain-shift combinations.
