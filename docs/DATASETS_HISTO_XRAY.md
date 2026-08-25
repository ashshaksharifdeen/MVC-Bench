# Histopathology and chest X-ray dataset preparation

This guide prepares the datasets used by the MVC-Bench `histo-xray` branch. It follows the [CalibPrompt dataset guide](https://github.com/iabh1shekbasu/CalibPrompt/blob/main/docs/DATASETS.md) and the preprocessing utilities documented by [BAPLe](https://github.com/asif-hanif/baple/blob/main/datasets/DATASETS.md).

Always obtain data under the original license and terms. Do not commit images, DICOM files, private metadata, or generated caches to Git.

## Final common layout

```text
med-datasets/
├── covid/
│   ├── images/train/<class_name>/*
│   ├── images/test/<class_name>/*
│   └── classnames.txt
├── rsna18/
│   └── ...
├── kather/
│   └── ...
├── pannuke/
│   └── ...
└── digestpath/
    └── ...
```

Depending on the branch's dataset loader, the first run may create:

```text
<dataset>/preprocessed.pkl
<dataset>/split_fewshot/shot_<k>-seed_<s>.pkl
```

These files encode dataset state. Regenerate them whenever paths, splits, labels, or class-name order change.

## Preprocessing environment

Run preprocessing in the same family environment that will consume the data, unless a script documents a separate requirement:

- `mvc-baple` for MedCLIP/BioMedCLIP;
- `mvc-dac` for PLIP/QuiltNet.

The upstream BAPLe utilities call for dataset-specific packages such as `pydicom==2.4.4`, `pandas==2.2.2`, `scikit-learn==1.5.1`, and `multiprocess==0.70.16`. Install only the packages needed by the script being run:

```bash
python -m pip install \
  pydicom==2.4.4 \
  pandas==2.2.2 \
  scikit-learn==1.5.1 \
  multiprocess==0.70.16
```

CalibPrompt's installation uses Python 3.10, PyTorch 2.1/CUDA 12.1, and NumPy 1.26.3. This aligns most closely with `mvc-dac`. When using `mvc-baple`, do not upgrade core packages blindly; if a preprocessing utility conflicts, run it in a temporary preprocessing environment and copy only the generated images/manifests to `med-datasets/`.

## Chest X-ray: COVIDX/COVID radiography

Upstream preparation uses the COVID-19 Radiography dataset archive and begins from a directory named similarly to `COVID-19_Radiography_Dataset`.

For the binary MVC-Bench task:

1. extract the archive under a raw-data directory;
2. map `COVID/images` to the COVID class;
3. map `Normal/images` to the normal class;
4. create deterministic train/test manifests;
5. run the upstream `train_test_split_covid.py` utility when using the BAPLe scripts;
6. place the results under `covid/images/train/` and `covid/images/test/`;
7. create `covid/classnames.txt` in the exact label order used by the loader.

Recommended source-preserving structure:

```text
raw/covid/COVID-19_Radiography_Dataset/
processed/covid/images/train/{covid,normal}/
processed/covid/images/test/{covid,normal}/
```

Do not include unrelated classes from the archive unless the branch configuration explicitly defines a multiclass experiment.

## Chest X-ray: RSNA18

RSNA18 preparation begins from the challenge labels/metadata and DICOM images, commonly including:

```text
unprocessed/
├── stage_2_train_images/
├── stage_2_test_images/
├── *.csv
└── *.txt
```

Procedure:

1. obtain the challenge files from the official source;
2. retain the original DICOMs and label tables under `raw/rsna18/`;
3. install `pydicom`, `pandas`, and `scikit-learn` as listed above;
4. run the upstream `train_test_split_rsna18.py` preprocessing utility;
5. verify pixel decoding and output class assignment;
6. arrange processed images under `rsna18/images/{train,test}/<class_name>/`;
7. create `rsna18/classnames.txt` in label-index order.

Check for patient-level overlap by patient identifier, not just filename. Record the DICOM windowing, normalization, bit-depth conversion, and output image format used by the script.

## Histopathology: Kather

The upstream setup uses:

- **NCT-CRC-HE-100K** as training data;
- **CRC-VAL-HE-7K** as test data.

Procedure:

1. download both archives from their official source;
2. extract them separately under `raw/kather/`;
3. arrange or point the upstream utility at the 100K training and 7K validation/test directories;
4. install `multiprocess==0.70.16` if required;
5. run `process_kather.py` from the referenced preprocessing code;
6. write the final data to `kather/images/train/` and `kather/images/test/`;
7. create `kather/classnames.txt` using the loader's exact class order.

Do not mix tiles from the fixed external test set into training. Preserve the canonical class abbreviations and their expanded prompt names in a versioned mapping file.

## Histopathology: PanNuke

The upstream setup uses PanNuke folds 1, 2, and 3 and converts nuclei/tissue annotations into the benchmark's benign/malignant classification organization.

Procedure:

1. obtain all required folds from the official source;
2. retain each fold separately under `raw/pannuke/`;
3. create processed `benign` and `malignant` class targets as defined by the upstream mapping;
4. run `process_pannuke.py`;
5. run `train_test_split_pannuke.py` with a recorded seed;
6. write outputs to `pannuke/images/{train,test}/{benign,malignant}/`;
7. create `pannuke/classnames.txt` with the same order used by labels.

Document the exact tissue/nucleus-to-binary mapping. This mapping is part of the experiment definition and must be identical across backbones.

## Histopathology: DigestPath

The upstream setup uses the negative and positive tissue archives, commonly named:

```text
tissue-train-neg.zip
tissue-train-pos-v1.zip
```

Procedure:

1. obtain the archives from the official source;
2. extract them beneath `raw/digestpath/`;
3. create processed benign/malignant targets;
4. install `multiprocess==0.70.16` if required;
5. run the three documented stages of `process_digestpath.py` in order (`--step 1`, `--step 2`, `--step 3`);
6. run `train_test_split_digestpath.py` with a recorded seed;
7. write outputs to `digestpath/images/{train,test}/{benign,malignant}/`;
8. create `digestpath/classnames.txt` in label order.

Each step should be restartable. Preserve intermediate manifests until image counts and labels have been verified.

## Dataset roles in MVC-Bench

| Modality | In-domain | Domain-shift evaluation |
|---|---|---|
| Chest X-ray | RSNA18 | COVIDX/COVID |
| Histopathology | PanNuke | DigestPath, Kather |

Keep the target-domain test data isolated from training, prompt selection, calibration fitting, and hyperparameter selection. Domain-shift calibration must not use target test labels.

## Point the branch to the datasets

Set the dataset root consumed by the shell launchers and `configs/datasets/*.yaml` to the absolute `med-datasets` directory. Verify both checkouts use the same processed source but write to different output directories:

```text
DATA_ROOT=/absolute/path/to/med-datasets
MODEL_ROOT=/absolute/path/to/model-weights
```

The actual variable names in the checked-in scripts are authoritative. Do not add unused environment variables and assume the code reads them.

## Validation script requirements

Before training, validate each dataset with a small read-only checker that reports:

- missing/corrupt files;
- train/test image counts and per-class counts;
- duplicate file hashes within and across splits;
- patient/slide overlap where identifiers exist;
- class-name order and unique label indices;
- sample tensor shape, dtype, value range, and label;
- cached split filename, shot count, and seed.

Manually inspect a grid of samples after preprocessing, especially DICOM conversion and pathology patches. Automated counts do not reveal inverted intensity, blank images, color-channel swaps, or incorrect crops.

## Smoke tests

MedCLIP/BioMedCLIP:

```bash
conda activate mvc-baple
cd /absolute/path/to/MVC-Bench-medclip-biomedclip
bash scripts/all_fewshot_medclip_new.sh
```

PLIP/QuiltNet:

```bash
conda activate mvc-dac
cd /absolute/path/to/MVC-Bench-plip-quiltnet
bash scripts/all_fewshot_plip_new.sh
```

Restrict each launcher to one dataset, one backbone, and one seed for the first run. Confirm the dataset root, class names, model preprocessing, checkpoint path, accuracy, and ECE before launching the full experiment grid.

## Final checklist

- [ ] Dataset terms and citations are recorded.
- [ ] Raw archives and metadata remain immutable.
- [ ] Preprocessing scripts, arguments, seeds, and package versions are recorded.
- [ ] No image, patient, or slide leaks across train/test splits.
- [ ] Class-index mappings are identical across environments and backbones.
- [ ] RSNA DICOM conversion has been visually inspected.
- [ ] PanNuke and DigestPath binary mappings are documented.
- [ ] First-run caches were generated only after finalizing labels and paths.
- [ ] Target-domain test labels were not used for fitting calibration parameters.
- [ ] A one-seed smoke test succeeds for each backbone family.
