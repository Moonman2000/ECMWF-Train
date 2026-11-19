# Using observations in an Anemoi model

Notebook: [`use_observations.ipynb`](use_observations.ipynb)

We take the deterministic O48 model of session 5 and add a second dataset, the way we would add observations,
in two ways:

| Setup | Config | The second dataset (`obs`) is | Starts from |
|---|---|---|---|
| 1. Decode to observations | `finetune_obs_decoder.yaml` | target: its variables are diagnostic, predicted by the obs decoder | pretrained model, frozen |
| 2. Encode observations | `encode_obs.yaml` | input: its variables are prognostic, the obs loss has weight 0 | pretrained model |

The pretrained model is `pretrain_o48.yaml` (run `4438386d8b894aa59beb57bb1c7e0b85`, set as `training.fork_run_id`
in the configs that start from it).

For simplicity, `obs` is **ERA5 on the O96 grid** (complete fields, no gaps), with variables the pretrained
model never had (wind, humidity). The notebook explains what changes for real (sparse) observations.

## Requirements

- **anemoi-training 0.9.0, anemoi-models 0.12.0**: the environment of session 5.
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
| `configs/no_obs_control.yaml` | control run for setup 2: the same training without `obs` |
| `configs/obs/era5_o96.yaml` | the dataset `obs`: variables, files, dates, loss. Copy it for another dataset and select it with `obs=<name>`. |
| `configs/graph/{pretrain_o48,finetune_obs_decoder,encode_obs}.yaml` | the graph of each case. In anemoi-training 0.9 it is applied to each dataset (one graph per dataset, sharing the hidden nodes). |
| `configs/data/toy.yaml`, `configs/dataloader/toy_o48.yaml`, `configs/system/course.yaml` | ERA5 O48 toy dataset, periods, paths |
| `configs/{diagnostics,model,system/{hardware,input,output},training}/...` | copies of the anemoi-training 0.9.0 defaults the configs use, unchanged. They take precedence over the installed package, so `configs/` holds every setting of every run. |
| `helpers.py` | config loading, training from the notebook, MLflow, running a model on one date |

## Training in a terminal

From this folder. The periods of the setups must be the exact first and last dates of the O96 toy dataset in
2020 (training) and 2021 (validation); the notebook prints them (`DATES`).

```bash
export ANEMOI_BASE_SEED=42
D="system.input.dataset=<era5 O48 zarr> system.input.obs_dataset=<era5 O96 zarr>"
DATES="dataloader.training.datasets.obs.start=<first 2020 date> dataloader.training.datasets.obs.end=<last 2020 date> \
  dataloader.validation.datasets.obs.start=<first 2021 date> dataloader.validation.datasets.obs.end=<last 2021 date>"

anemoi-training train --config-path=$PWD/configs --config-name=pretrain_o48 $D
anemoi-training train --config-path=$PWD/configs --config-name=finetune_obs_decoder $D $DATES
anemoi-training train --config-path=$PWD/configs --config-name=encode_obs $D $DATES
anemoi-training train --config-path=$PWD/configs --config-name=no_obs_control $D ${DATES//.obs./.data.}
```

To start from another pretrained run, add `training.fork_run_id=<run id>`.

## Things to know (anemoi-training 0.9)

- **Dates are matched by position.** The samples of the datasets are matched by their position in the
  selected period, not by date. All datasets must have the same frequency and exactly the same dates. The O96
  toy dataset holds only a few days, so `configs/obs/era5_o96.yaml` sets the periods of `data` from those of
  `obs`, and the notebook reads them from the dataset (and checks they have no gaps).
- **Few training samples.** With the O96 toy dataset, the setups train on ~10 days: 1000 steps are many
  epochs. Good for a demo, and for discussing overfitting.
- **Every dataset has an encoder, a decoder and a loss**, named after the dataset (`encoder.obs`, `decoder.obs`),
  all of the kind of `model.encoder` / `model.decoder`, with the same `model.trainable_parameters`. In setup 1
  the obs encoder reads only the forcings of `obs`; in setup 2 the obs loss has weight 0.
- **One graph per dataset.** The graph config is applied to each dataset, with its grid points as the `data`
  nodes, and saved as `<system.input.graph>_<dataset>.pt`.
- **Freezing** (`training.submodules_to_freeze`) matches module *names* anywhere in the model: `data` freezes
  `encoder.data`, `decoder.data`, their graph edges and the trainable parameters of the `data` nodes.
- **Real gridded observations** have NaN where nothing was observed: they need an imputer, and the loss needs a
  mask from the gaps of the *target* time (anemoi's `InputImputer` takes them from the first input time).

## Looking ahead

The notebook ends with what changes in newer versions of anemoi-core: encoders and decoders per dataset
(input-only and output-only datasets), tabular observations with a time `window`, graphs built per batch, and
decoding at arbitrary locations.
