# Diabetic-retinopathy dataset preparation

This guide prepares the diabetic-retinopathy (DR) data used by MVC-Bench on the `main` branch. It follows the acquisition and domain organization used by [SPSD-ViT](https://github.com/Chumsy0725/SPSD-ViT), while keeping the final processed tree compatible with the dataset loader and YAML configuration in MVC-Bench.

## Datasets and benchmark roles

| Dataset/domain | MVC-Bench role | Notes |
|---|---|---|
| Messidor | In-domain | Class 4 is excluded for the paper's ID experiment |
| APTOS | Domain shift | Keep labels aligned with the MVC-Bench class order |
| EyePACS | Domain shift | Keep labels aligned with the MVC-Bench class order |
| Messidor-2 | Domain shift | Do not merge with Messidor |

Obtain each dataset from its official distributor or from the data bundle referenced by SPSD-ViT. Dataset licenses may prohibit redistribution; MVC-Bench should therefore provide preparation code and links, not re-host protected images.

## 1. Keep raw and processed data separate

Recommended storage:

```text
data/
└── dr/
    ├── raw/
    │   ├── aptos/
    │   ├── eyepacs/
    │   ├── messidor/
    │   └── messidor_2/
    └── processed/
        ├── aptos/
        ├── eyepacs/
        ├── messidor/
        └── messidor_2/
```

Treat `raw/` as immutable. Fix naming, label mapping, split creation, and optional preprocessing only under `processed/`.

SPSD-ViT/DomainBed commonly represents source data as `dataset/domain/class/image`. Preserve that download layout first so source provenance remains auditable. Convert or link it to the exact layout required by the MVC-Bench dataset classes only after checking `datasets/` and `configs/datasets/` on the `main` branch.

## 2. Use a class-folder layout for MVC-Bench

The recommended processed layout is:

```text
data/dr/processed/
├── aptos/
│   ├── images/
│   │   ├── train/<class_name>/*.png
│   │   ├── val/<class_name>/*.png
│   │   └── test/<class_name>/*.png
│   └── classnames.txt
├── eyepacs/
│   └── ...
├── messidor/
│   └── ...
└── messidor_2/
    └── ...
```

If the checked-in MVC-Bench loader expects `train/` and `test/` directly beneath a dataset rather than beneath `images/`, follow the loader. Do not create two competing layouts. The loader implementation and `configs/datasets/*.yaml` are authoritative.

`classnames.txt` must contain one class per line, in exactly the same index order used by the labels. Use disease-aware names that are consistent across all four domains. A typical five-grade DR mapping is:

```text
0 no diabetic retinopathy
1 mild diabetic retinopathy
2 moderate diabetic retinopathy
3 severe diabetic retinopathy
4 proliferative diabetic retinopathy
```

Verify the actual labels and wording used by the branch before generating results. A mismatch between label indices and class-name order invalidates both accuracy and calibration.

## 3. Create deterministic splits

Use the repository's checked-in split/preprocessing utility when available. Store the seed and generated manifest. Each manifest should contain at least:

```text
relative_image_path,label,source_dataset,split
```

Requirements:

- use patient-level splitting where patient identifiers are available;
- prevent the same image or patient from appearing in multiple splits;
- preserve original image-to-label provenance;
- report class counts before and after filtering;
- use a validation split for model selection and calibration fitting;
- do not tune calibration parameters on the test set.

For few-shot runs, MVC-Bench/Dassl may create cached files such as `split_fewshot/shot_16-seed_1.pkl`. Generate independent caches for each dataset and seed. Remove a cache only when deliberately regenerating it after a dataset/configuration change.

## 4. Apply the paper's Messidor protocol safely

The paper uses Messidor as the in-domain dataset and excludes class 4 in that experiment. Preserve class 4 in the raw data. Implement exclusion through one of the following reproducible mechanisms:

1. a versioned processed manifest that omits class 4;
2. a named split file committed with the experiment configuration; or
3. an explicit dataset-loader/config option.

Record the resulting class counts. All downstream class names, label indices, and prompts must use the reduced class space consistently. Do not retain a five-class prompt list while evaluating a four-class logit tensor.

## 5. Image preprocessing

Use the transform configured by the selected CLIP backbone and trainer. Do not apply an additional normalization on already normalized tensors. If preprocessing images offline, record:

- color-space conversion;
- crop or border-removal policy;
- resize/interpolation method;
- output format and compression;
- whether left/right-eye images are treated independently;
- the script version and parameters.

Domain-shift evaluation should preserve meaningful acquisition differences. Avoid domain-specific enhancement that leaks target-domain knowledge or makes DS images artificially resemble the ID domain unless that transformation is a declared experimental condition.

## 6. Point MVC-Bench to the data

Update the root field used in the relevant file under `configs/datasets/` and the variable consumed by `scripts/coop/base2new_train_coop_datasets.sh`. Prefer one absolute root:

```text
/absolute/path/to/data/dr/processed
```

Then verify that the dataset registry resolves all four names before running the complete sweep. A minimal loader test should print:

- resolved dataset root;
- train/validation/test sizes;
- class names in index order;
- per-class counts;
- one decoded image tensor shape and label.

## 7. Run a smoke test

```bash
conda activate maple
cd /absolute/path/to/MVC-Bench-main
bash scripts/coop/base2new_train_coop_datasets.sh
```

Temporarily restrict the launcher to one dataset, one CLIP backbone, one trainer, and one seed. Confirm that few-shot sampling, checkpoint saving, evaluation, and ECE reporting finish successfully. Restore the full grid only after this check.

## Validation checklist

- [ ] Raw data remain unchanged.
- [ ] Every image path in a manifest exists and is readable.
- [ ] No image hash occurs in more than one split.
- [ ] Patient-level leakage is ruled out where metadata permit.
- [ ] Label indices and `classnames.txt` have the same order.
- [ ] Messidor class-4 exclusion is explicit and reversible.
- [ ] ID and DS domains have separate names and output paths.
- [ ] Few-shot caches are unique per dataset, shot count, and seed.
- [ ] Calibration fitting uses validation data only.
- [ ] Dataset source, license, preprocessing, split seed, and counts are recorded.

## Source and citation

Dataset preparation follows [SPSD-ViT](https://github.com/Chumsy0725/SPSD-ViT). Cite SPSD-ViT and the original APTOS, EyePACS, Messidor, and Messidor-2 dataset publications or official challenge pages in any derived work.
