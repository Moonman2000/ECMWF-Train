# Using observations in an Anemoi model

Notebook: [`use_observations.ipynb`](use_observations.ipynb)

We take the deterministic O48 model of session 5 and add a second dataset, the way we would add observations,
in two ways:

| Setup | Config | The second dataset (`obs`) is | Starts from |
|---|---|---|---|
| 1. Decode to observations | `finetune_obs_decoder.yaml` | target of a new decoder (output-only dataset) | pretrained model, frozen |
| 2. Encode observations | `encode_obs.yaml` | input of a new encoder (input-only dataset) | pretrained model |

The pretrained model is `pretrain_o48.yaml`.

For simplicity, `obs` is **ERA5 on the O96 grid** (complete fields, no gaps), with variables the pretrained
model never had (wind, humidity). The notebook explains what changes for real (sparse) observations.

## Requirements

- **anemoi-core `main`**, installed from source (configs checked against commit 7bbffef39). The configs use the
  new `task:`, `training.optimization` and `training.checkpoint` sections. They do not run with
  anemoi-training 0.9.0, the version of session 5.
- One GPU. The graph transformer uses triton kernels, so running the models (also in the notebook) needs CUDA.
- Two gridded datasets:
  - the ERA5 O48 toy dataset of session 5 (`aifs-ea-an-oper-0001-mars-o48-2010-2022-6h-v1-toy`)
  - the ERA5 O96 toy dataset of session 6 (`aifs-ea-an-oper-0001-mars-o96-6h-v8-profiling-toy`), only a few
    days at the end of 2020 and the start of 2021.

  Set the paths at the top of the notebook (`DATASET`, `OBS_DATASET`) and the run id of the pretrained model
  (`PRETRAINED_RUN_ID`). If its checkpoint does not exist, the notebook trains `pretrain_o48` first.

## Files

| File | |
|---|---|
| `configs/pretrain_o48.yaml` | the pretrained model (ERA5 O48 only) |
| `configs/finetune_obs_decoder.yaml`, `configs/encode_obs.yaml` | setups 1 and 2 |
| `configs/obs/era5_o96.yaml` | the dataset `obs`: variables, files, dates. Copy it for another dataset and select it with `obs=<name>`. |
| `configs/graph/{pretrain_o48,finetune_obs_decoder,encode_obs}.yaml` | the full graph of each case (nodes, edges, attributes). `data`, `hidden` and their edges must stay the same as in the pretrained model. |
| `configs/data/toy.yaml`, `configs/dataloader/toy_o48.yaml`, `configs/system/course.yaml` | ERA5 O48 toy dataset, periods, paths |
| `helpers.py` | config loading, training from the notebook, MLflow, running a model on one date |

All other config files (e.g. `model/graphtransformer.yaml`) come from the installed anemoi-training package.

## Training in a terminal

From this folder. The periods of the setups must be the exact first and last dates of the O96 toy dataset in
2020 (training) and 2021 (validation); the notebook prints them (`DATES`).

```bash
export ANEMOI_BASE_SEED=42
D="system.input.dataset=<era5 O48 zarr> system.input.obs_dataset=<era5 O96 zarr>"
DATES="dataloader.training.datasets.obs.start=<first 2020 date> dataloader.training.datasets.obs.end=<last 2020 date> \
  dataloader.validation.datasets.obs.start=<first 2021 date> dataloader.validation.datasets.obs.end=<last 2021 date>"
CKPT=<output root>/checkpoint/<pretrained run id>/last.ckpt

anemoi-training train --config-path=$PWD/configs --config-name=pretrain_o48 $D
anemoi-training train --config-path=$PWD/configs --config-name=finetune_obs_decoder $D $DATES training.checkpoint.source.path=$CKPT
anemoi-training train --config-path=$PWD/configs --config-name=encode_obs $D $DATES training.checkpoint.source.path=$CKPT

# control run for setup 2: the pretrained model, 1000 more steps on the same dates, without obs
anemoi-training train --config-path=$PWD/configs --config-name=pretrain_o48 $D ${DATES//.obs./.data.} \
  training/checkpoint/source=local training/checkpoint/loading=transfer_learning \
  training.checkpoint.source.path=$CKPT training.max_steps=1000 \
  training.optimization.lr=1e-4 training.optimization.lr_scheduler.warmup_t=50 \
  dataloader.batch_size.validation=1 dataloader.limit_batches.validation=null \
  diagnostics.log.mlflow.run_name=no-obs-control
```

## Things to know (anemoi-core `main`)

- **Dates are matched by position.** The samples of the datasets are matched by their position in the
  selected period, not by date. All datasets must have the same frequency and exactly the same dates. The O96
  toy dataset holds only a few days, so `configs/obs/era5_o96.yaml` sets the periods of `data` from those of
  `obs`, and the notebook reads them from the dataset (and checks they have no gaps).
- **Few training samples.** With the O96 toy dataset, the setups train on ~10 days: 1000 steps are many
  epochs. Good for a demo, and for discussing overfitting.
- **Output-only datasets** (all variables diagnostic or forcing, decoder but no encoder) cannot use
  `encoded_data` in their decoder; use `coordinates`, `forcings` or `trainable_parameters`.
- **Input-only datasets** (encoder but no decoder) have no loss. They still need an entry in
  `model.residual.datasets`.
- **Every dataset needs an `model.output_mask.datasets.<name>` entry.**
- **Freezing** names are module paths from the LightningModule: `model.model.encoder.<group>`,
  `model.model.processor`, `model.model.decoder.<group>`, plus `model.model.node_attributes` and
  `model.model.*_graph_provider` for the trainable graph parameters.
- **Real gridded observations** have NaN where nothing was observed. `main` checks that a dataset has no NaNs
  after its processors, so they need an imputer, and the loss needs a mask from the gaps of the *target* time
  (anemoi's `InputImputer` takes them from the first input time).

## Richer batches

The notebook ends with what changes with the richer batches of anemoi-core (`feat/richer-batch`): tabular
observations with a time `window`, graphs built per batch, `target_template` for decoding at arbitrary
locations, and config changes.
