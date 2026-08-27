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

- `baple` for MedCLIP/BioMedCLIP;
- `dac` for PLIP/QuiltNet.

The upstream BAPLe utilities call for dataset-specific packages such as `pydicom==2.4.4`, `pandas==2.2.2`, `scikit-learn==1.5.1`, and `multiprocess==0.70.16`. Install only the packages needed by the script being run:

```bash
python -m pip install \
  pydicom==2.4.4 \
  pandas==2.2.2 \
  scikit-learn==1.5.1 \
  multiprocess==0.70.16
```


