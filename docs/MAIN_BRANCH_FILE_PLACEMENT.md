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

## 1. Switch to the main branch

```bash
cd /absolute/path/to/MVC-Bench
git switch main
```

Check that the working tree is clean before moving the launcher:

```bash
git status --short
```

If the output contains unrelated changes, commit or safely preserve them before continuing.

## 2. Move the launcher from root into `scripts/coop/`

```bash
mkdir -p scripts/coop
git mv base2new_train_coop_datasets.sh \
  scripts/coop/base2new_train_coop_datasets.sh
```

If the script is untracked, ordinary `mv` may be used instead of `git mv`, followed by `git add`.

## 3. Make the launcher location-independent

Moving a shell script can break relative paths. The safest rule is:

- invoke the script from the repository root; and
- make the script change to the repository root before calling `train.py` or reading `configs/`.

Add the following immediately after the shebang in `scripts/coop/base2new_train_coop_datasets.sh` if equivalent logic is not already present:

```bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
```

After this block, repository-relative references such as `train.py`, `configs/`, `datasets/`, and `output/` resolve from the repository root.

Do not add a second `set -euo pipefail` or root-resolution block if the script already has one.

## 4. Copy the Markdown files

From the main-branch repository root:

```bash
mkdir -p docs

cp /absolute/path/to/MVC-Bench-documentation/README-main.md \
  README.md

cp /absolute/path/to/MVC-Bench-documentation/DATASET-DR.md \
  docs/DATASET_DR.md

cp /absolute/path/to/MVC-Bench-documentation/MAIN-BRANCH-FILE-PLACEMENT.md \
  docs/MAIN_BRANCH_FILE_PLACEMENT.md

cp /absolute/path/to/MVC-Bench-documentation/REPOSITORY-STRUCTURE.md \
  docs/REPOSITORY_STRUCTURE.md
```

`README-main.md` is a distribution filename. Inside the branch it must be named `README.md` at repository root.

## 5. Validate the launcher after moving it

Check shell syntax:

```bash
bash -n scripts/coop/base2new_train_coop_datasets.sh
```

Check that the old root copy no longer exists:

```bash
test ! -e base2new_train_coop_datasets.sh
```

Search for stale root-level references:

```bash
git grep -n 'bash base2new_train_coop_datasets.sh' || true
git grep -n 'base2new_train_coop_datasets.sh'
```

All user-facing run instructions should use:

```bash
bash scripts/coop/base2new_train_coop_datasets.sh
```

## 6. Run a one-seed smoke test

```bash
conda activate maple
cd /absolute/path/to/MVC-Bench
bash scripts/coop/base2new_train_coop_datasets.sh
```

Temporarily restrict the script to one DR dataset, one CLIP backbone, one trainer, and one seed. Confirm that it finds `train.py`, YAML configurations, the dataset root, and the output directory.

## 7. Commit the branch update

```bash
git status
git diff --check
git diff -- README.md docs/ scripts/coop/

git add README.md docs/ scripts/coop/base2new_train_coop_datasets.sh
git commit -m "Organize main-branch launcher and documentation"
git push origin main
```

## Main-branch placement checklist

- [ ] `README.md` is at repository root.
- [ ] DR documentation is under `docs/`.
- [ ] The CoOp launcher exists only under `scripts/coop/`.
- [ ] The script changes to the repository root before using relative paths.
- [ ] README commands use `bash scripts/coop/base2new_train_coop_datasets.sh`.
- [ ] Shell syntax and a one-seed smoke test succeed.
- [ ] The changes are committed only to `main`.

