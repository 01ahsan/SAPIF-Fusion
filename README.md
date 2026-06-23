# sapif-fusion

Repository for building and evaluating fusion workflows from external20, MUST, and PulseDB SBP base predictions.

## Input assumption

This repo expects model outputs from another project in `data/input/`:

- `external20_base_predictions.csv`
- `must_base_predictions.csv`
- `pulsedb_sbp_base_predictions.csv`

## Layout

- `scripts/` contains the runnable pipeline scripts.
- `data/` stores input placeholders and schema files.
