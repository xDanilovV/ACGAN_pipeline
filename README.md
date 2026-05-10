# AC-GAN GC-IMS Pipeline

Reusable PyTorch pipeline for class-conditioned synthetic GC-IMS spectra. The
default workflow is config-driven: GC-IMS preprocessing, shared region-of-interest
detection, tensor shape selection, AC-GAN training, synthetic sample export, and
evaluation are all controlled from `configs/acgan_balanced_peakcrop.json`.

## Inputs

The pipeline supports native `.mea` folders and a simple `.npz` fallback.

For `.mea`, provide a folder containing `.mea` files. Labels can be supplied with
a CSV or inferred from the fermentation dataset folder layout.

Accepted CSV file columns: `file`, `filename`, `path`, `sample`, `name`.
Accepted label columns: `label`, `class`, `target`, `group`.

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

Use `--mea-label-mode class` for the default culture/combination classes, or
`--mea-label-mode culture_type` for a broader pure-vs-mixed experiment.

The `.npz` fallback expects:

- `samples`: array shaped `[N, H, W]`
- `labels`: integer class labels shaped `[N]`

## Run

Install dependencies:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Inspect the fermentation labels:

```powershell
.\.venv\Scripts\python.exe -m acgan_pipeline.data.inspect_fermentation `
  C:\Users\user\PycharmProjects\PythonProject\data_fermentation
```

Run the config-driven AC-GAN pipeline locally:

```powershell
.\.venv\Scripts\python.exe -m acgan_pipeline.main `
  --config configs\acgan_balanced_peakcrop.json `
  --data C:\Users\user\PycharmProjects\PythonProject\data_fermentation `
  --output-dir out_acgan
```

Run the same experiment on Beast:

```bash
cd ~/ACGAN_pipeline
git pull
source .venv/bin/activate
python -m acgan_pipeline.main \
  --config configs/acgan_balanced_peakcrop.json \
  --data ~/ACGAN_pipeline/data_fermentation \
  --output-dir out_acgan
```

The config defaults keep `shape_mode` set to `auto`, so normal runs should not
need manual `--height` or `--width`. The loader computes one shared peak-aware
crop across the dataset, then rounds the resulting model tensor size to a
multiple of 16. This keeps all spectra aligned while avoiding unnecessary empty
background.

## Preprocessing

For `.mea` data, the preprocessing path is:

1. read the spectrum with `gc-ims-tools`
2. convert drift time to RIP-relative coordinates
3. apply optional drift/retention cuts
4. subtract a per-spectrum intensity baseline percentile
5. clip high-intensity outliers by percentile
6. apply `log1p` compression
7. compute one shared peak-aware crop for all samples
8. resize to the automatically selected GAN tensor shape

Current recommended intensity settings are stored in
`configs/acgan_balanced_peakcrop.json`:

```json
"intensity_baseline_percentile": 25.0,
"intensity_clip_high_percentile": 99.9,
"intensity_log1p": true
```

Do not use `--no-peak-crop` for normal `.mea` training. It is only a debugging
switch; raw `.mea` files may not have identical shapes, and the shared crop is
the step that makes the dataset consistently stackable.

Preview preprocessing without training:

```powershell
.\.venv\Scripts\python.exe -m acgan_pipeline.data.export_preprocessing_preview `
  --data C:\Users\user\PycharmProjects\PythonProject\data_fermentation `
  --peak-crop `
  --output-dir outputs_preprocessing_preview
```

## Diagnostics

Before trusting GAN training, verify that the discriminator class head can learn
real spectra:

```bash
python -m acgan_pipeline.diagnostics.classifier_probe \
  --config configs/acgan_balanced_peakcrop.json \
  --data ~/ACGAN_pipeline/data_fermentation \
  --output-dir runs_classifier_probe \
  --epochs 80 \
  --batch-size 16 \
  --lr 0.001
```

If this probe is near chance, the issue is preprocessing, labels, or architecture
before it is a GAN issue.

## Outputs

Each run writes to the selected output directory:

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

The preprocessing examples include a raw/processed/synthetic triplet for visual
inspection plus `real_tensor_vs_generated_class_*.png`, which compares the real
training tensor and generated tensor at the exact same model resolution.

## Evaluation

The evaluation suite reports:

- train real, test real
- train real + synthetic, test real
- train real, test synthetic
- train synthetic, test real

The default downstream classifier is PCA + SVM. Metrics include accuracy,
balanced accuracy, macro precision/recall/F1, weighted F1, per-class metrics,
support, and confusion matrices.

Compact metric summary:

```bash
python - <<'PY'
import json
m = json.load(open("out_acgan/evaluation/metrics.json"))
for k in ["real_only_test_real", "real_plus_synthetic_test_real", "real_only_test_synthetic", "synthetic_only_test_real"]:
    r = m[k]
    print(k, "| acc", round(r["accuracy"], 4), "| bal", round(r["balanced_accuracy"], 4), "| f1", round(r["macro_f1"], 4))
PY
```

GAN checkpoints can differ sharply. When training is unstable, evaluate saved
checkpoints rather than assuming the final epoch is best.

The balanced config also enables early stopping on `g_structure`, a generator
structure penalty that tracks intensity, peak-density, and border-artifact
matching against real training batches. The best structural checkpoint is saved
as `checkpoints/best_early_stopping.pt`.

## Documentation

See `docs/technical_pipeline.md` for the preprocessing rationale, AC-GAN design
notes, evaluation interpretation, and literature grounding.
