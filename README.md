# AC-GAN GC-IMS Pipeline

Config-driven PyTorch pipeline for class-conditioned synthetic GC-IMS spectra.
The project keeps the core model as an Auxiliary Classifier GAN (AC-GAN), while
making the surrounding experiment reproducible: preprocessing, shared ROI
selection, tensor sizing, training, synthetic sample export, and evaluation are
all controlled from `configs/acgan_balanced_peakcrop.json`.

The current default is conservative for small GC-IMS datasets:

- native `.mea` loading through `gc-ims-tools`
- RIP-relative drift alignment
- percentile baseline subtraction, high-intensity clipping, and `log1p`
  compression
- shared peak-aware crop and automatic tensor shape selection
- position-aware AC-GAN discriminator class head
- optional class-template anchoring for the generator
- structure-aware generator penalties and early stopping
- PCA + SVM downstream evaluation

## Project Structure

```text
ACGAN_pipeline/
+-- acgan_pipeline/              # Source package
|   +-- data/                    # MEA loading, label inference, previews
|   +-- diagnostics/             # Data profile and classifier probe scripts
|   +-- models/                  # Generator, discriminator, AC-GAN wrappers
|   +-- training/                # Training loop and sampling utilities
|   +-- utils/                   # Loss and metric helpers
|   +-- visualization/           # GC-IMS plots and report exports
+-- configs/                     # Reproducible experiment configurations
+-- data/
|   +-- external/                # Third-party downloaded datasets
|   +-- interim/                 # Intermediate transformed data
|   +-- processed/               # Canonical modelling data
|   +-- raw/                     # Immutable raw data dumps
+-- docs/                        # Technical documentation
+-- models/                      # Trained model artifacts kept outside git
+-- notebooks/                   # Exploratory notebooks
+-- references/                  # Papers, notes, data dictionaries
+-- reports/
|   +-- figures/                 # Figures used in reports or slides
+-- requirements.txt
+-- README.md
```

Generated run folders such as `out_*`, `runs_*`, `outputs*`, checkpoints, and
large data files are ignored by git. The empty data/report/model folders are
kept with `.gitkeep` files so the documentation structure is visible.

See [docs/project_structure.md](docs/project_structure.md) for the folder
conventions and [docs/technical_pipeline.md](docs/technical_pipeline.md) for the
methodological details.

## Installation

Create or activate a virtual environment, then install the dependencies:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

On Beast:

```bash
cd ~/ACGAN_pipeline
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

For `.mea` input, provide a folder containing native GC-IMS files. Labels can be
supplied by CSV or inferred from the fermentation folder layout.

Accepted CSV file columns: `file`, `filename`, `path`, `sample`, `name`.
Accepted label columns: `label`, `class`, `target`, `group`.

Example fermentation layout:

```text
data_fermentation/
  GCIMS_mixed_cultures/
    E. coli and S. cerevisiae/
      211007_EC_SC_Batch_1/
        t1_211007_132333.mea
  GCIMS_pure_cultures/
    E. coli/
      210429_EC_Batch_1/
        t1_*.mea
```

Use `--mea-label-mode class` for the default organism/combination classes, or
`--mea-label-mode culture_type` for a broader pure-vs-mixed experiment.

The `.npz` fallback expects:

- `samples`: array shaped `[N, H, W]`
- `labels`: integer class labels shaped `[N]`

## Standard Run

Fermentation dataset, local Windows:

```powershell
.\.venv\Scripts\python.exe -m acgan_pipeline.main `
  --config configs\acgan_balanced_peakcrop.json `
  --data C:\Users\user\PycharmProjects\PythonProject\data_fermentation `
  --output-dir out_acgan `
  --classifier svm
```

Fermentation dataset, Beast:

```bash
cd ~/ACGAN_pipeline
git pull
source .venv/bin/activate
python -m acgan_pipeline.main \
  --config configs/acgan_balanced_peakcrop.json \
  --data ~/ACGAN_pipeline/data_fermentation \
  --output-dir out_acgan \
  --classifier svm
```

Honey dataset example:

```bash
python -m acgan_pipeline.main \
  --config configs/acgan_balanced_peakcrop.json \
  --data ~/ACGAN_pipeline/data_honey_botanical \
  --output-dir out_honey \
  --classifier svm
```

The balanced config keeps `shape_mode` set to `auto`, so standard runs should
not need manual `--height` or `--width`. The loader computes one shared
peak-aware crop across the dataset and rounds the tensor size to a multiple of
16 for the convolutional AC-GAN.

## Preprocessing

The `.mea` preprocessing path is:

1. read the spectrum with `gc-ims-tools`
2. convert drift time to RIP-relative coordinates
3. optionally apply retention/drift cuts
4. subtract a per-spectrum intensity baseline percentile
5. clip high-intensity outliers by percentile
6. apply `log1p` compression
7. compute one shared peak-aware crop for all samples
8. resize to the automatically selected GAN tensor shape
9. normalize for AC-GAN training

Current recommended intensity settings live in the config:

```json
"intensity_baseline_percentile": 25.0,
"intensity_clip_high_percentile": 99.9,
"intensity_log1p": true
```

Keep `peak_crop` enabled for normal `.mea` training. Raw `.mea` files may have
different shapes, and the shared crop is the step that makes the dataset
consistently stackable without per-sample alignment drift.

## Diagnostics

Inspect inferred fermentation labels:

```bash
python -m acgan_pipeline.data.inspect_fermentation ~/ACGAN_pipeline/data_fermentation
```

Profile the loaded dataset:

```bash
python -m acgan_pipeline.diagnostics.data_profile \
  --config configs/acgan_balanced_peakcrop.json \
  --data ~/ACGAN_pipeline/data_fermentation \
  --output-dir runs_data_profile
```

Check whether the discriminator-style class head can learn real spectra before
trusting GAN training:

```bash
python -m acgan_pipeline.diagnostics.classifier_probe \
  --config configs/acgan_balanced_peakcrop.json \
  --data ~/ACGAN_pipeline/data_fermentation \
  --output-dir runs_classifier_probe \
  --epochs 80 \
  --batch-size 16 \
  --lr 0.001
```

If this probe is near chance, fix preprocessing, labels, or architecture before
interpreting GAN losses.

## Outputs

Each run writes:

- `effective_config.json`
- `preprocessing_report.json`
- `preprocessing_examples/`
- `checkpoints/`
- `samples/`
- `synthetic_samples.npz`
- `synthetic_samples_denormalized_for_visualization.npz` when requested
- `evaluation/metrics.json`
- `evaluation/*_confusion_matrix.csv`
- `evaluation/*_confusion_matrix.png`

Export separate report images for one class:

```bash
python -m acgan_pipeline.visualization.export_run_examples \
  --run-dir out_acgan \
  --data ~/ACGAN_pipeline/data_fermentation \
  --output-dir out_acgan/separate_spectra \
  --class-id 0
```

This writes:

- `01_raw_native_ims.png`
- `02_processed_sample.png`
- `03_generated_sample.png`

## Evaluation

The evaluation suite reports four cases:

- `real_only_test_real`: baseline real classifier performance
- `real_plus_synthetic_test_real`: augmentation usefulness
- `real_only_test_synthetic`: whether generated samples land in real learned
  class regions
- `synthetic_only_test_real`: whether generated data alone transfers to real
  spectra

Compact metric summary:

```bash
python - <<'PY'
import json
m = json.load(open("out_acgan/evaluation/metrics.json"))
keys = [
    "real_only_test_real",
    "real_plus_synthetic_test_real",
    "real_only_test_synthetic",
    "synthetic_only_test_real",
]
for key in keys:
    r = m[key]
    print(
        key,
        "| acc", round(r["accuracy"], 4),
        "| bal", round(r["balanced_accuracy"], 4),
        "| f1", round(r["macro_f1"], 4),
        "| train", r["num_train"],
        "| test", r["num_test"],
        "| synth_train", r.get("num_synthetic_in_train", 0),
    )
PY
```

The important interpretation is conservative: `real_plus_synthetic_test_real`
can improve even when cross-domain tests remain weak. Report that as
augmentation behavior, not proof that the synthetic spectra are fully faithful.
