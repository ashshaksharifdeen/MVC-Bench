# Histo-Xray branch: file and launcher placement

This guide shows exactly where to place the MVC-Bench documentation and medical-backbone experiment launchers on the `histo-xray` branch.

The medical branch follows BAPLe's layout, where backbone launchers live directly under `scripts/`. They should not remain at repository root.

## Required final hierarchy

```text
MVC-Bench/                         # histo-xray branch
├── README.md
├── configs/
│   ├── datasets/
│   └── trainers/
├── datasets/
├── trainers/
├── scripts/
│   ├── all_fewshot_medclip_new.sh
│   ├── all_fewshot_plip_new.sh
│   └── <other existing branch launchers>.sh
├── docs/
│   ├── DATASETS_HISTO_XRAY.md
│   ├── HISTO_XRAY_FILE_PLACEMENT.md
│   └── REPOSITORY_STRUCTURE.md
├── train.py                       # or the branch's existing entry point
├── parse_test_res.py              # if present
└── requirements.txt
```

Do not create duplicate `medclip/` or `plip/` directories if the branch already follows BAPLe's flat `scripts/` layout.

## 1. Switch to the medical branch

```bash
cd /absolute/path/to/MVC-Bench
git switch histo-xray
git status --short
```

If the working tree contains unrelated changes, commit or safely preserve them before moving files.

## 2. Move both launchers into `scripts/`

```bash
mkdir -p scripts

git mv all_fewshot_medclip_new.sh \
  scripts/all_fewshot_medclip_new.sh

git mv all_fewshot_plip_new.sh \
  scripts/all_fewshot_plip_new.sh
```

If either script is untracked, use ordinary `mv` for that file and stage it afterward.

## 3. Make each launcher location-independent

Both launchers should execute repository-relative paths from the repository root. Add this block immediately after the shebang when equivalent logic is not already present:

```bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
```

Because the scripts are one directory below the repository root, this version uses `${SCRIPT_DIR}/..`. This differs from the main-branch CoOp launcher, which is two directories below root and therefore uses `${SCRIPT_DIR}/../..`.

Do not duplicate this block if the scripts already resolve their own location.

## 4. Copy the Markdown files

From the `histo-xray` repository root:

```bash
mkdir -p docs

cp /absolute/path/to/MVC-Bench-documentation/README-histo-xray.md \
  README.md

cp /absolute/path/to/MVC-Bench-documentation/DATASETS-HISTO-XRAY.md \
  docs/DATASETS_HISTO_XRAY.md

cp /absolute/path/to/MVC-Bench-documentation/HISTO-XRAY-FILE-PLACEMENT.md \
  docs/HISTO_XRAY_FILE_PLACEMENT.md

cp /absolute/path/to/MVC-Bench-documentation/REPOSITORY-STRUCTURE.md \
  docs/REPOSITORY_STRUCTURE.md
```

`README-histo-xray.md` is a distribution filename. Inside this branch it must be named `README.md` at repository root.

## 5. Validate the moved launchers

```bash
bash -n scripts/all_fewshot_medclip_new.sh
bash -n scripts/all_fewshot_plip_new.sh

test ! -e all_fewshot_medclip_new.sh
test ! -e all_fewshot_plip_new.sh
```

Find stale root-level run instructions:

```bash
git grep -n 'bash all_fewshot_medclip_new.sh' || true
git grep -n 'bash all_fewshot_plip_new.sh' || true
git grep -n 'all_fewshot_medclip_new.sh'
git grep -n 'all_fewshot_plip_new.sh'
```

All user-facing commands should now be:

```bash
bash scripts/all_fewshot_medclip_new.sh
bash scripts/all_fewshot_plip_new.sh
```

## 6. Test the BAPLe checkout

```bash
conda activate mvc-baple
cd /absolute/path/to/MVC-Bench-medclip-biomedclip
bash scripts/all_fewshot_medclip_new.sh
```

Restrict the first run to one dataset, one backbone, and one seed. Verify model and dataset roots before starting the complete grid.

## 7. Test the DAC checkout

After committing and pushing the `histo-xray` update, synchronize the second checkout:

```bash
cd /absolute/path/to/MVC-Bench-plip-quiltnet
git switch histo-xray
git pull origin histo-xray

conda activate mvc-dac
bash scripts/all_fewshot_plip_new.sh
```

Do not repeat the same documentation commit from the second checkout. Both local directories track the same GitHub branch; they are separate only to isolate environments.

## 8. Commit the branch update

From the checkout where the files were moved:

```bash
git status
git diff --check
git diff -- README.md docs/ scripts/

git add README.md docs/ \
  scripts/all_fewshot_medclip_new.sh \
  scripts/all_fewshot_plip_new.sh

git commit -m "Organize medical launchers and branch documentation"
git push origin histo-xray
```

## Histo-Xray placement checklist

- [ ] `README.md` is at repository root.
- [ ] Histo-Xray documentation is under `docs/`.
- [ ] Both medical launchers exist only under `scripts/`.
- [ ] Each script changes to the repository root before using relative paths.
- [ ] README commands start with `bash scripts/`.
- [ ] Shell syntax succeeds for both launchers.
- [ ] A MedCLIP/BioMedCLIP smoke test succeeds in `mvc-baple`.
- [ ] A PLIP/QuiltNet smoke test succeeds in `mvc-dac`.
- [ ] The update is committed once to `histo-xray` and pulled into the second checkout.

