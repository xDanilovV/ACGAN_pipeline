# Project Structure

The repository follows a lightweight research-project layout: source code stays
in `acgan_pipeline/`, experiment settings stay in `configs/`, technical notes
stay in `docs/`, and large generated artifacts stay out of git.

```text
ACGAN_pipeline/
+-- acgan_pipeline/
|   +-- data/              # MEA loading, label inference, preprocessing previews
|   +-- diagnostics/       # Dataset profiling and real-data classifier probes
|   +-- models/            # AC-GAN model definitions
|   +-- training/          # Training loop, checkpointing, sampling
|   +-- utils/             # Losses and metrics
|   +-- visualization/     # Figures for inspection and reporting
+-- configs/               # JSON experiment configurations
+-- data/
|   +-- external/          # Third-party datasets as downloaded
|   +-- interim/           # Intermediate transformed data
|   +-- processed/         # Final data prepared for modelling
|   +-- raw/               # Immutable raw data dumps
+-- docs/                  # Technical documentation
+-- models/                # Local trained models and checkpoints
+-- notebooks/             # Exploratory notebooks
+-- references/            # Papers, notes, and data dictionaries
+-- reports/
|   +-- figures/           # Figures used in reports and presentations
+-- requirements.txt
+-- README.md
```

## Conventions

- Keep raw `.mea` data out of git. Place local datasets under `data/raw/`,
  `data/external/`, or a local ignored folder such as `data_fermentation/`.
- Keep generated training runs in short `out_*` folders.
- Keep diagnostic runs in `runs_*` folders.
- Copy only final selected figures into `reports/figures/` if they are needed
  for a written report or presentation.
- Store reproducible run settings in `configs/` instead of relying on long
  command-line overrides.
- Store methodological notes in `docs/technical_pipeline.md`.
