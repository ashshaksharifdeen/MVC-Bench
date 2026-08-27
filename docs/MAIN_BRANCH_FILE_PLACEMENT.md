# Main branch: file and launcher placement

This guide shows exactly where to place the MVC-Bench documentation and general-CLIP experiment launcher on the `main` branch.

The main branch follows MaPLe's method-grouped script hierarchy. The CoOp launcher therefore belongs in `scripts/coop/`, not at repository root.

## Required final hierarchy

```text
MVC-Bench/                         # main branch
├── README.md
├── configs/
│   ├── datasets/
│   └── trainers/
├── datasets/
├── trainers/
├── scripts/
│   ├── coop/
│   │   └── base2new_train_coop_datasets.sh
│   ├── maple/
│   ├── kgcoop/                    # if present in the branch
│   ├── promptsrc/                 # if present in the branch
│   ├── prograd/                   # if present in the branch
│   └── hicropl/                   # if present in the branch
├── docs/
│   ├── DATASET_DR.md
│   ├── MAIN_BRANCH_FILE_PLACEMENT.md
│   └── REPOSITORY_STRUCTURE.md
├── train.py
├── parse_test_res.py
└── requirements.txt
```

Do not create empty method directories solely to match this illustration. Preserve the method directories already present on the branch.



