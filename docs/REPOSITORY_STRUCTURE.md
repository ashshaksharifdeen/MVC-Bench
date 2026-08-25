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
│   ├── all_fewshot_medclip_new.sh # histo-xray only
│   └── all_fewshot_plip_new.sh    # histo-xray only
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

## Shell-launcher audit

Before a long experiment, inspect the active launcher and answer all of the following:

1. Which Conda environment and checkout does it assume?
2. Which datasets and ID/DS roles does it enumerate?
3. Which dataset and trainer YAML files are passed to `train.py`?
4. Which backbone identifier and weight root are used?
5. Which trainer/calibration method is selected?
6. Are shots, seeds, learning rate, batch size, and epochs paper-aligned?
7. Does every run receive a unique output directory?
8. Is evaluation loading the intended checkpoint rather than an older run?
9. Does a failed subprocess stop the sweep or silently continue?
10. Is the complete resolved command written to the log?

Use `bash -n <script>.sh` to catch shell syntax errors. Use `bash -x <script>.sh` only for a small smoke test because a full sweep creates very large logs.

## Dataset-loader audit

For every dataset class, verify:

- its registry name exactly matches the YAML and launcher;
- train/validation/test directories resolve beneath the intended root;
- class names have a deterministic order;
- source labels map to contiguous indices beginning at zero;
- ID-only filters such as the Messidor class-4 exclusion are explicit;
- cached preprocessed/few-shot splits include enough information to prevent collisions;
- images are opened in the color mode expected by the backbone;
- transformations are applied once;
- patient/slide groups do not cross splits where identifiers are available.

Print the resolved class list and split sizes at startup. These should appear in every experiment log.

## Backbone-adapter audit

Different VLM packages expose different APIs. Each adapter should normalize them to the same benchmark contract:

```text
input batch -> image encoder -> normalized image features
class prompts -> tokenizer/text encoder -> normalized text features
similarity/logit scale -> class logits -> softmax probabilities
```

Verify:

- correct tokenizer and prompt format;
- matching image preprocessing and input resolution;
- feature normalization convention;
- learned or fixed logit scale handling;
- class-logit dimension and class-name order;
- frozen versus trainable parameters;
- checkpoint key translation;
- device and mixed-precision behavior.

Never reuse CLIP preprocessing automatically for a medical backbone without checking its upstream model definition.

## Prompt-trainer and MCM audit

For prompt-tuning methods, record which tensors are trainable and confirm the backbone remains frozen where required. For MCM, compute true-versus-rest margins from the same logits used by cross-entropy:

```math
m_{i,k}=z_{i,y_i}-z_{i,k}, \qquad
L_{MCM}=-\alpha\mu+\beta\sigma^2_{all-pairs}, \qquad
L=L_{CE}+L_{MCM}.
```

Paper values are `alpha=0.1` and `beta=0.01`. Add unit checks for:

- correct exclusion of the true class from rest-class margins;
- finite mean/variance for batch size and class count;
- gradient flow only through intended parameters;
- equality with plain cross-entropy when MCM is disabled;
- configuration logging of alpha and beta.

## Calibration metric audit

ECE must be computed from predicted confidence and correctness using 20 bins for the primary paper result. Record bin boundaries and empty-bin behavior. MCE and ACE implementations should be named/versioned because libraries differ.

For temperature scaling or other fitted post-hoc methods:

1. save validation logits/labels;
2. fit the parameter using validation data only;
3. freeze the parameter;
4. apply it once to test logits;
5. save both uncalibrated and calibrated metrics.

Calibration should not change accuracy when it only rescales logits monotonically. If top-1 accuracy changes after scalar temperature scaling, inspect the implementation.

## Result-directory convention

Use an unambiguous hierarchy such as:

```text
outputs/
└── <modality>/
    └── <id_dataset>-to-<target_dataset>/
        └── <backbone>/
            └── <prompt_method>/
                └── <calibration_method>/
                    └── shots_<n>/seed_<seed>/
```

Each leaf should contain:

- complete console log;
- resolved configuration;
- command and Git commit;
- environment name or lock file identifier;
- checkpoint or a pointer/checksum;
- final accuracy, ECE, MCE, and ACE;
- calibration parameters fitted on validation data.

Never infer run identity only from the parent directory or terminal history.

## Paper-aligned experiment settings

| Setting | Value |
|---|---|
| CoOp shots | 16 |
| Learning rate | 0.005 |
| Batch size | 8 |
| Seeds | 1, 2, 3 |
| Primary ECE bins | 20 |
| MCM alpha | 0.1 |
| MCM beta | 0.01 |
| Reported reference GPU | NVIDIA RTX A6000 |

If a model requires a changed batch size or precision mode, report that deviation and keep the effective optimization settings comparable where possible.

## Minimum smoke-test matrix

Run one dataset and one seed for each environment/backbone family:

| Environment | Minimum test |
|---|---|
| `maple` | one general CLIP backbone on one DR dataset |
| `mvc-baple` | one MedCLIP forward/train/eval cycle and one BioMedCLIP model-load/forward test |
| `mvc-dac` | one PLIP forward/train/eval cycle and one QuiltNet model-load/forward test |

The smoke test passes only if it resolves data and weights, creates a checkpoint, evaluates it, writes accuracy/ECE, and uses the intended output directory.

## Full reproducibility checklist

- [ ] Branch, commit, checkout, and environment match the backbone family.
- [ ] Dependency lock/export is saved after verification.
- [ ] Dataset root, model root, and output root are absolute and logged.
- [ ] Raw data are immutable and processed data are traceable.
- [ ] Dataset split and few-shot seeds are recorded.
- [ ] Class-name/index order is identical across all backbones.
- [ ] Medical backbones use their own tokenizer and image preprocessing.
- [ ] Validation data alone determine calibration parameters/hyperparameters.
- [ ] Three-seed mean and standard deviation are reported.
- [ ] ID and DS metrics cannot overwrite each other.
- [ ] Result aggregation detects missing or failed runs.
- [ ] Environment, GPU, command, and Git commit accompany each result table.

## Upstream references

- [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)
- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)
- [BAPLe](https://github.com/asif-hanif/baple)
- [DAC/CLIP_Calibration](https://github.com/ml-stat-Sustech/CLIP_Calibration)
- [CalibPrompt](https://github.com/iabh1shekbasu/CalibPrompt)
- [SPSD-ViT](https://github.com/Chumsy0725/SPSD-ViT)
