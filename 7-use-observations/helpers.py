"""Helpers of the observations hands-on (anemoi-core `main`)."""

import codecs
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SESSION_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SESSION_DIR / "configs"


# ---------------------------------------------------------------------------------------------------------
# Configs and trainings
# ---------------------------------------------------------------------------------------------------------
def load_config(config_name, overrides=(), validate=True):
    """Build a training config the way anemoi-training does (Hydra), resolve all ${...} references and,
    if `validate` and the config has config_validation: True, check it against the config schema of
    anemoi-training (as `anemoi-training train` does)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.resolve(config)
    if validate and config.get("config_validation", True):
        from anemoi.training.schemas.base_schema import BaseSchema

        BaseSchema(**config)
    return config


def config_diff(config_a, config_b, skip=("diagnostics", "system")):
    """List the (key, value in a, value in b) of all settings that differ between two configs."""
    from omegaconf import OmegaConf

    def flatten(node, prefix=""):
        if isinstance(node, list) and node and all(isinstance(v, dict) for v in node):
            node = {str(i): v for i, v in enumerate(node)}
        if not isinstance(node, dict):
            return {prefix: node}
        flat = {}
        for key, value in node.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat

    a = flatten(OmegaConf.to_container(config_a))
    b = flatten(OmegaConf.to_container(config_b))
    keys = sorted(k for k in a.keys() | b.keys() if not k.startswith(tuple(skip)))
    return [(k, a.get(k, "-"), b.get(k, "-")) for k in keys if a.get(k, "-") != b.get(k, "-")]


def run_training(config_name, overrides=(), seed=42):
    """Run `anemoi-training train` from the session folder and stream its output into the notebook.

    Same as running this in a terminal (in the folder of this file):
        PYTHONPATH=$PWD ANEMOI_BASE_SEED=42 anemoi-training train --config-path=$PWD/configs --config-name=<config_name> <overrides>
    """
    executable = Path(sys.executable).parent / "anemoi-training"  # the one of this notebook's environment
    command = [str(executable), "train", f"--config-path={CONFIG_DIR}", f"--config-name={config_name}", *overrides]
    print(" ".join(command[1:]), flush=True)
    env = {**os.environ, "ANEMOI_BASE_SEED": str(seed), "PYTHONWARNINGS": "ignore"}
    # the session folder on the path, for custom classes referenced in the configs (e.g. processors)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SESSION_DIR), env.get("PYTHONPATH")]))
    # Read raw bytes so that the progress bar's carriage returns update it in place instead of adding lines
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with subprocess.Popen(command, cwd=SESSION_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        while chunk := process.stdout.read1(4096):
            print(decoder.decode(chunk), end="", flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {process.returncode}), see the output above.")


# ---------------------------------------------------------------------------------------------------------
# MLflow (offline, files in <output root>/logs/mlflow)
# ---------------------------------------------------------------------------------------------------------
def latest_run(client, run_name, experiment_name="training-course-obs"):
    """The most recent MLflow run with this name (diagnostics.log.mlflow.run_name)."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment {experiment_name!r} yet")
    runs = client.search_runs(
        [experiment.experiment_id],
        f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No run called {run_name!r} yet")
    return runs[0]


def metric_history(client, run, metric):
    import pandas as pd

    history = client.get_metric_history(run.info.run_id, metric)
    return pd.Series([m.value for m in history], index=[m.step for m in history], name=metric).sort_index()


def plot_metrics(client, runs, metrics, ylabel="loss"):
    """Plot the history of some metrics of one or more runs, one panel per metric."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for run in runs:
            values = metric_history(client, run, metric)
            if len(values):
                ax.plot(values.index, values.values, ".-" if len(values) < 30 else "-", label=run.info.run_name)
        ax.set(title=metric, xlabel="training step", ylabel=ylabel, yscale="log")
        ax.legend()
    plt.tight_layout()


# ---------------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------
def lon180(lon):
    return (np.asarray(lon) + 180) % 360 - 180


def scatter_map(ax, lat, lon, values, title="", s=4, **kwargs):
    """Scatter plot of point values on a map (NaNs are not drawn)."""
    import cartopy.crs as ccrs

    values = np.asarray(values)
    ok = ~np.isnan(values)
    im = ax.scatter(lon180(lon)[ok], np.asarray(lat)[ok], c=values[ok], s=s, transform=ccrs.PlateCarree(), **kwargs)
    ax.coastlines(linewidth=0.5)
    ax.set_global()
    ax.set_title(title)
    return im


# ---------------------------------------------------------------------------------------------------------
# Running a trained model on one sample, from Python
# ---------------------------------------------------------------------------------------------------------
def open_datasets(dataset_configs):
    """Open each dataset like the training does: {name: anemoi.datasets.open_dataset(dataset_config)}."""
    from anemoi.datasets import open_dataset
    from omegaconf import OmegaConf

    opened = {}
    for name, cfg in dataset_configs.items():
        cfg = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        opened[name] = open_dataset(cfg)
    return opened


def make_input(model, datasets, date, timestep=np.timedelta64(6, "h")):
    """The input of `model` for a forecast from `date`: the states at date - (n_step_input-1)*timestep ... date.

    Returns {name: tensor (batch=1, time, grid, variables)} with the input variables of the model for each
    dataset (all variables except the diagnostic ones), as AnemoiModelInterface.predict_step expects.
    """
    import torch

    date = np.datetime64(date)
    batch = {}
    for name, ds in datasets.items():
        n_step_input = getattr(model, "n_step_input", None) or model.multi_step  # name in newer versions / 0.12
        dates = [date - k * timestep for k in reversed(range(n_step_input))]
        idx = [int(np.where(ds.dates == d)[0][0]) for d in dates]
        x = np.stack([ds[i][:, 0, :] for i in idx])  # (time, variables, grid)
        x = x[:, model.data_indices[name].data.input.full.cpu().numpy()]
        batch[name] = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1)))[None].float()
    return batch


def predict(model, datasets, date):
    """Run `model` for one step from `date`: {dataset: {variable: predicted values on its grid}}."""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        out = model.predict_step({k: v.to(device) for k, v in make_input(model, datasets, date).items()})
    return {
        name: dict(zip(output_variables(model, name), y.squeeze().float().cpu().numpy().T))
        for name, y in out.items()
    }


def target(datasets, name, date, timestep=np.timedelta64(6, "h")):
    """The true values of dataset `name` at date + timestep, as {variable: values on the grid}."""
    ds = datasets[name]
    i = int(np.where(ds.dates == np.datetime64(date) + timestep)[0][0])
    values = ds[i][:, 0, :]
    return {v: values[k] for k, v in enumerate(ds.variables)}


def output_variables(model, name):
    """Names of the output variables of dataset `name` of a model, in the order of the model output."""
    data_indices = model.data_indices[name]
    name_to_index = data_indices.model.output.name_to_index
    return sorted(name_to_index, key=name_to_index.get)
