# AC-GAN GC-IMS Pipeline

Reusable PyTorch pipeline for class-conditioned synthetic GC-IMS-like 2D spectra.

## Inputs

The pipeline supports native `.mea` folders and a simple `.npz` fallback.

For `.mea`, provide:

- a folder containing `.mea` files
- optionally, a labels CSV with one file column and one label column

Accepted file columns: `file`, `filename`, `path`, `sample`, `name`.
Accepted label columns: `label`, `class`, `target`, `group`.

If no labels CSV is provided, labels are inferred from folders. This supports
the fermentation dataset layout:

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

Use `--mea-label-mode class` for the default 10 culture/combination classes, or
`--mea-label-mode culture_type` for a broader pure-vs-mixed experiment.

Example:

```csv
filename,label
sample_001.mea,control
sample_002.mea,treatment
```

The `.npz` fallback expects:

- `samples`: array shaped `[N, H, W]`
- `labels`: integer class labels shaped `[N]`

Dataset-specific loading should be handled by replacing `load_npz_dataset` in
`acgan_pipeline/main.py` or by passing arrays directly to `GCIMSDataset`.

## Run

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -m acgan_pipeline.data.inspect_fermentation C:\Users\user\PycharmProjects\PythonProject\data_fermentation
.\.venv\Scripts\python.exe -m acgan_pipeline.main `
  --input-format mea `
  --data C:\Users\user\PycharmProjects\PythonProject\data_fermentation `
  --epochs 100 `
  --batch-size 32 `
  --synthetic-viz-denormalized
```

Optional preprocessing controls:

```powershell
.\.venv\Scripts\python.exe -m acgan_pipeline.main `
  --input-format mea `
  --data C:\Users\user\PycharmProjects\PythonProject\data_fermentation `
  --rip-drift-start 1.05 `
  --rip-drift-stop 2.00 `
  --crop-rt-start 80 `
  --crop-rt-stop 500
```

For `.mea` input, RIP removal uses gc-ims-tools:

```python
ims.Spectrum.read_mea(path).riprel().cut_dt(1.05, stop)
```

The default keeps drift-time values after `1.05` in RIP-relative coordinates.
This is a practical first setting from the documented gc-ims-tools workflow;
we should tune it by visually inspecting your spectra.

Before long training runs, compare resize choices:

```powershell
.\.venv\Scripts\python.exe -m acgan_pipeline.data.export_preprocessing_preview `
  --data C:\Users\user\PycharmProjects\PythonProject\data_fermentation `
  --height 384 `
  --width 128 `
  --resize-mode area
```

The original processed spectra are roughly `6123 x 1900`, so square `128 x 128`
is too compressed for thesis-quality synthetic spectra. Prefer rectangular
model inputs such as `384 x 128` or `512 x 160`, both divisible by 16.

For a peak-aware crop, compute a shared high-intensity window before resizing:

```powershell
.\.venv\Scripts\python.exe -m acgan_pipeline.data.export_preprocessing_preview `
  --data C:\Users\user\PycharmProjects\PythonProject\data_fermentation `
  --peak-crop `
  --height 512 `
  --width 128 `
  --resize-mode area
```

Training with the shared crop:

```powershell
.\.venv\Scripts\python.exe -m acgan_pipeline.main `
  --input-format mea `
  --data C:\Users\user\PycharmProjects\PythonProject\data_fermentation `
  --mea-label-mode class `
  --peak-crop `
  --height 512 `
  --width 128 `
  --resize-mode area `
  --epochs 100 `
  --batch-size 4 `
  --samples-per-class 30 `
  --output-dir outputs_peakcrop_512x128_svm `
  --classifier svm `
  --synthetic-viz-denormalized
```

## Outputs

Each run writes to `outputs/` by default:

- `preprocessing_report.json`
- `preprocessing_examples/`
- `checkpoints/`
- `samples/`
- `synthetic_samples.npz`
- `synthetic_samples_denormalized_for_visualization.npz` when requested
- `evaluation/metrics.json`
- `evaluation/*_confusion_matrix.csv`

## Evaluation Experiments

The evaluation suite runs:

- train on real, test on real
- train on real + synthetic, test on real
- train on real, test on synthetic
- train on synthetic, test on real

Reported metrics include accuracy, balanced accuracy, macro precision/recall/F1,
weighted F1, per-class precision/recall/F1, support, and confusion matrices.

