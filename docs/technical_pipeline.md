# Technical Pipeline

This document describes the current AC-GAN GC-IMS workflow, with emphasis on
the failure points we have been debugging: preprocessing, class conditioning,
synthetic spectra quality, and downstream evaluation.

## Design Goal

The pipeline is intentionally still an AC-GAN pipeline. The goal is not to
switch to diffusion, VAE, or ReACGAN as the main method, but to make the AC-GAN
experiment reproducible and scientifically defensible:

- labels must be assigned before training without ambiguity
- preprocessing must preserve chemically meaningful GC-IMS structure
- tensor shape must be derived from the spectra, not guessed manually
- synthetic samples must be evaluated as both augmentations and standalone data

This distinction matters because a synthetic set can slightly improve a real
classifier while still failing cross-domain tests. In that case the synthetic
samples may be acting as a regularizer or perturbation source, not as faithful
GC-IMS records.

## MEA Label Flow

For `.mea` input, labels are assigned in `acgan_pipeline/data/mea_loader.py`.

The loader accepts a labels CSV, but if no CSV is supplied it infers labels from
the fermentation folder structure. With `mea_label_mode = "class"`, each culture
or mixed-culture folder becomes a class. The encoded integer labels are written
to `preprocessing_report.json` under `label_summary`.

Checks to run before training:

```bash
python -m acgan_pipeline.data.inspect_fermentation ~/ACGAN_pipeline/data_fermentation
```

```bash
python -m acgan_pipeline.diagnostics.data_profile \
  --config configs/acgan_balanced_peakcrop.json \
  --data ~/ACGAN_pipeline/data_fermentation \
  --output-dir runs_data_profile
```

If the class counts or class names are wrong at this stage, the GAN will learn
the wrong conditional problem no matter how stable the losses look.

## Preprocessing Flow

The `.mea` preprocessing path is:

1. Read the spectrum with `gc-ims-tools`.
2. Convert the drift-time axis to RIP-relative coordinates.
3. Apply optional drift-time or retention-time cuts.
4. Apply intensity preprocessing:
   - subtract a per-spectrum baseline percentile
   - clamp negative values to zero
   - optionally clip low/high percentiles
   - apply `log1p` compression
5. Compute one shared peak-aware crop across the dataset.
6. Resize to the selected GAN tensor shape.
7. Normalize the final dataset to the range expected by the AC-GAN.

The recommended intensity preprocessing is stored in
`configs/acgan_balanced_peakcrop.json`:

```json
"intensity_baseline_percentile": 25.0,
"intensity_clip_low_percentile": null,
"intensity_clip_high_percentile": 99.9,
"intensity_log1p": true
```

Clipping is part of the preprocessing contract. It should not be repeatedly
typed into the training command, because that makes results hard to reproduce.

The percentile operations use a bounded pixel sample for speed:

```json
"intensity_percentile_max_pixels": 250000,
"peak_percentile_max_pixels": 250000
```

This keeps large `.mea` files from spending excessive time in exact percentile
partitioning while preserving deterministic behavior for a fixed array layout.

## Region Of Interest And Shape

Normal `.mea` training should keep `peak_crop = true`.

The shared crop is not cosmetic. It solves three problems:

- different raw `.mea` files may not have identical array shapes
- most pixels can be low-information background
- per-sample crops would destroy alignment and leak sample-specific geometry

The crop is computed once from high-intensity support across the whole dataset,
then every sample is cropped to the same retention/drift window. After that,
`shape_mode = "auto"` chooses a model input size using:

```json
"auto_max_pixels": 65536,
"auto_max_height": 512,
"auto_max_width": 256,
"auto_multiple": 16
```

The multiple-of-16 constraint is required by the current convolutional generator
and discriminator down/up-sampling path.

Manual `height` and `width` are still available for controlled ablations, but
they should not be part of the standard experiment.

## AC-GAN Architecture Notes

The model follows the AC-GAN idea from Odena et al.: the discriminator predicts
both source validity and class, while the generator is conditioned on class.

Important local choices:

- The discriminator class head is position-aware. This fixed the earlier probe
  failure where the classifier stayed near chance.
- The discriminator class head is pretrained on real spectra before adversarial
  training.
- The generator does not use the discriminator's image shortcut class head for
  its class loss. That prevents the generator from exploiting image-level class
  artifacts instead of producing plausible spectra.
- Fake class loss for the discriminator is available but defaults to `0.0`.
  It is useful as an ablation, not as the default.

The most suspicious GAN signal remains a discriminator that becomes confidently
correct while synthetic samples are visually off. In our runs this appears as
high discriminator class accuracy plus poor `real_only_test_synthetic` and
`synthetic_only_test_real` metrics.

## Evaluation Interpretation

The pipeline evaluates four cases:

- `real_only_test_real`: baseline real classifier performance
- `real_plus_synthetic_test_real`: augmentation usefulness
- `real_only_test_synthetic`: whether synthetic samples land in the real
  classifier's learned class regions
- `synthetic_only_test_real`: whether synthetic data alone can train a classifier
  that transfers to real spectra

The second number can improve while the third and fourth are poor. That means
the synthetic samples may help decision boundaries but are not faithful enough
to stand in for real GC-IMS spectra.

For thesis-quality claims, prioritize:

- visual triplets in `preprocessing_examples/`
- cross-domain confusion matrices
- checkpoint sweeps, especially early checkpoints
- real-only classifier probe performance

## Recommended Run

```bash
python -m acgan_pipeline.main \
  --config configs/acgan_balanced_peakcrop.json \
  --data ~/ACGAN_pipeline/data_fermentation \
  --output-dir out_sparse_log \
  --epochs 80 \
  --lr-d 0.00005 \
  --discriminator-update-every 2 \
  --classifier svm
```

After training, evaluate checkpoints if the final model is poor:

```bash
for e in 0020 0030 0040 0050 0060 0070 0080; do
  python -m acgan_pipeline.main \
    --config configs/acgan_balanced_peakcrop.json \
    --data ~/ACGAN_pipeline/data_fermentation \
    --generator-checkpoint out_sparse_log/checkpoints/epoch_${e}.pt \
    --output-dir out_sparse_log_eval_${e} \
    --classifier svm \
    --skip-visualization
done
```

## Source Grounding

The preprocessing and evaluation design is aligned with the GC-IMS and
generative-model literature already collected for the project:

- GC-IMS workflows and non-targeted screening require consistent preprocessing
  and careful chemometric validation: `freire2021gcims`, `capitain2021nts`,
  `christmann2024fermentation`.
- AC-GAN is the core architecture: `odena2017acgan`, building on
  `goodfellow2014gan` and `mirza2014conditional`.
- Limited-data conditional GANs are prone to class-conditioning collapse:
  `shahbazi2022collapse`. ReACGAN discusses AC-GAN instability directly
  (`kang2021reacgan`), even though this pipeline keeps the main model as AC-GAN.
- Synthetic spectral augmentation has precedent but requires task-specific
  validation: `zhu2020began`, `zhu2023ccgan`, `frischia2020raman`,
  `gracia2023spectroscopy`, `wang2019oracgan`, `li2022acwgangp`.
- Generative model evaluation cannot rely on one number:
  `theis2016evaluation`, `borji2019gan`, `ravuri2019cas`.

These sources support the current stance: report augmentation performance, but
do not treat it as sufficient evidence that the synthetic spectra are faithful.
