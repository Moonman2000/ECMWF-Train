# Using observations in an Anemoi model

Notebook: [`use_observations.ipynb`](use_observations.ipynb)

We take the deterministic O48 model of session 5 and add an observation dataset as a second dataset,
in three ways:

| Setup | Config | Observations are | Starts from |
|---|---|---|---|
| 1. Decode to observations | `finetune_obs_decoder.yaml` | target of a new decoder (output-only dataset) | pretrained model, frozen |
| 2. Encode observations | `encode_obs.yaml` | input of a new encoder (input-only dataset) | pretrained model |
| 3. Observations only (DOP-like) | `dop_obs_only.yaml` | input (`obs_in`) and target (`obs_out`), no ERA5 | scratch |

The pretrained model is `pretrain_o48.yaml`.

## Requirements

- **anemoi-core `main`**, installed from source (tested with commit 7bbffef39, which installs as
  anemoi-training 0.9.1.post39 and anemoi-models 0.12.1.post39). The configs use the new `task:`,
  `training.optimization` and `training.checkpoint` sections. They do not run with anemoi-training 0.9.0,
  the version of session 5.
- One GPU. The graph transformer uses triton kernels, so running the models (also in the notebook) needs
  CUDA.
- The ERA5 toy dataset of session 5 and a **gridded** observation dataset. The configs were tested with
  radiosondes binned on O96 (`aifsdop-ea-ofb-oper-0001-mars-o96-2010-2024-6h-v1-observations-radiosondes`).
  Set the paths at the top of the notebook (`DATASET`, `OBS_DATASET`, `PRETRAINED`).

## Files

| File | |
|---|---|
| `configs/obs/radiosondes.yaml` | everything about the observation dataset: variables, processors, dates, graph nodes. Copy it for another observation dataset and select it with `obs=<name>`. |
| `configs/data/toy.yaml`, `configs/dataloader/toy_o48.yaml`, `configs/system/course.yaml` | ERA5 toy dataset, periods, paths |
| `configs/{data,dataloader,graph,model}/obs_only.yaml`, `configs/training/*/obs_only.yaml` | setup 3 |
| `obs_imputer.py` | `ObsImputer`: fills the gaps of the observations and masks the loss at the target time (see below) |
| `helpers.py` | config loading, training from the notebook, MLflow, running a model on one date |

All other config files (e.g. `model/graphtransformer.yaml`) come from the installed anemoi-training package.

## Training in a terminal

From this folder (`obs_imputer.py` must be importable):

```bash
export PYTHONPATH=$PWD ANEMOI_BASE_SEED=42
D="system.input.dataset=<era5 toy zarr> system.input.obs_dataset=<observation zarr>"

anemoi-training train --config-path=$PWD/configs --config-name=pretrain_o48 $D
anemoi-training train --config-path=$PWD/configs --config-name=finetune_obs_decoder $D training.checkpoint.source.path=<pretrained last.ckpt>
anemoi-training train --config-path=$PWD/configs --config-name=encode_obs $D training.checkpoint.source.path=<pretrained last.ckpt>
anemoi-training train --config-path=$PWD/configs --config-name=dop_obs_only $D

# control run for setup 2: the pretrained model, 1000 more steps without observations
anemoi-training train --config-path=$PWD/configs --config-name=pretrain_o48 $D \
  training/checkpoint/source=local training/checkpoint/loading=transfer_learning \
  training.checkpoint.source.path=<pretrained last.ckpt> training.max_steps=1000 \
  training.optimization.lr=1e-4 training.optimization.lr_scheduler.warmup_t=50 diagnostics.log.mlflow.run_name=no-obs-control
```

Wall times on one A100, including start-up and building the graph: pretraining (2000 steps) 8 min,
setup 1 (1000 steps) 4 min, setup 2 (1000 steps) 4.5 min, control run 3.5 min, setup 3 (2000 steps) 7.5 min.

## Things to know (anemoi-core `main`)

- **Dates are matched by position.** The samples of the datasets are matched by their position in the
  selected period, not by date. All datasets must have the same frequency and the same first date. The
  test observation dataset starts at 06 UTC, the toy ERA5 dataset at 00 UTC, so all periods start on
  2010-01-02.
- **No NaNs after the processors.** `main` checks that a dataset has no NaNs after its processors, so the
  gaps of gridded observations must be filled in (imputer), also when they are only a target. anemoi's
  `InputImputer` masks the loss with the gaps of the first time step of the sample (an input time), which
  is wrong for observations. `obs_imputer.ObsImputer` uses the gaps of the target time instead (valid
  for rollout 1), and puts NaNs back at those gaps after the model, so the validation metrics with
  `ignore_nans: True` only count real observations. In inference, it leaves the prediction everywhere.
- **Custom processors need `config_validation: False`.** The schema only accepts anemoi's own processor
  classes.
- **Output-only datasets** (all variables diagnostic or forcing) have a decoder but no encoder. Their decoder
  cannot use `encoded_data`; use `coordinates`, `forcings` or `trainable_parameters`.
- **Input-only datasets** (all variables forcing) have an encoder but no decoder and no loss. They still need
  an entry in `model.residual.datasets`.
- **Freezing** names are module paths from the LightningModule: `model.model.encoder.<group>`,
  `model.model.processor`, `model.model.decoder.<group>`, plus `model.model.node_attributes` and
  `model.model.*_graph_provider` for the trainable graph parameters.

## Richer batches

The notebook ends with what changes with the richer batches of anemoi-core (`feat/richer-batch`): tabular
observations with a time `window`, graphs built per batch, `target_template` for decoding at arbitrary
locations, and config changes. With tabular observations, `ObsImputer` and `nan_mask_weights` are no longer
needed: use `ignore_nans: True` on the loss.

## Results of the test runs (for instructors)

With the radiosonde test dataset, seed 42, one A100:

- **Setup 1**: the ERA5 validation loss stays exactly constant (everything pretrained is frozen); the obs
  validation loss halves in 1000 steps. Forecast from 2021-03-01 06 UTC, against the ~560-600 soundings at
  12 UTC (RMSE, obs decoder / climatology): `t_850` 2.1 / 15.7 K, `z_500` 341 / 2859 m²/s², `u_500`
  7.1 / 14.2 m/s. The model never had wind as a variable. The ERA5 decoder at the nearest O48 point is
  still better after 1000 steps (`t_850` 1.8 K, `z_500` 203 m²/s²): the new decoder starts from scratch.
- **Setup 3** (2000 steps from scratch) vs setup 1, same date: `t_850` 3.1 vs 2.2 K, `z_500` 498 vs 339
  m²/s², but `u_500` 6.1 vs 7.4 m/s (better: it sees past wind observations).
- **Setup 2** vs. the control run without observations (same 1000 steps, same learning rate): `z_500` MSE
  8460 vs 9315 (better), `t_850` MSE 0.75 vs 0.72 (worse), total loss about the same. A mixed result,
  good for discussing why (ERA5 already assimilated these soundings; imputed gaps; short training).
- The obs validation metrics in MLflow are scaled by the fraction of observed nodes (area weights summed
  over all obs nodes), so they are not an MSE: the notebook computes the errors itself.
- **Setup 3** (observations only, 2000 steps from scratch), same forecast as above (RMSE, obs only / ERA5 +
  obs decoder of setup 1): `t_850` 3.1 / 2.2 K, `z_500` 498 / 339 m²/s², `u_500` 6.1 / 7.4 m/s, `v_500`
  6.5 / 6.0 m/s. Worse for temperature and geopotential, but similar or better for wind.
