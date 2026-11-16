"""Helper code for the notebook ``graph_neural_networks_2026_solutions.ipynb``.

The notebook shows the important parts (graphs, message passing, the GNN layers).
This module holds the parts that are not specific to graphs: data loading,
the training loop, the evaluation and the plots.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:  # the notebook also works without cartopy, but the maps have no coastlines
    HAS_CARTOPY = False

# The anemoi dataset: ERA5 on the octahedral reduced Gaussian grid O48 (approximately 2 degrees), every 6 hours.
# You can give the name of the dataset (if your anemoi configuration knows where it is) or its full path.
DATASET = "aifs-ea-an-oper-0001-mars-o48-2010-2022-6h-v1-toy.zarr"
# os.environ.get(
#     "GNN_DATASET", "aifs-ea-an-oper-0001-mars-o48-2010-2022-6h-v1-toy.zarr"
# )
GRAVITY = 9.80665  # m s-2, to change the surface geopotential to a height
STANDARD_LAPSE_RATE = -6.5  # K per km

TEMP_CMAP = "RdYlBu_r"
ERROR_CMAP = "RdBu_r"
OROG_CMAP = "terrain"


def set_seed(seed: int = 42) -> None:
    """Make the random numbers repeatable."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Use the GPU if there is one."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Era5Sample:
    """ERA5 fields on the points of an unstructured grid.

    t2m has the shape (time, point) and the unit kelvin. orography is in metres.
    """

    t2m: np.ndarray
    orography: np.ndarray
    land_sea_mask: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    times: np.ndarray

    @property
    def num_points(self) -> int:
        return self.latitudes.size


def load_era5(start, end, frequency: str = "12h", dataset: str = DATASET, num_threads: int = 16) -> Era5Sample:
    """Read the 2 m temperature, the orography and the land-sea mask with anemoi-datasets.

    ``start`` and ``end`` are years or dates, as in ``anemoi.datasets.open_dataset``.
    The dates are read in parallel (one thread for each date), because each date is one chunk.
    """
    from anemoi.datasets import open_dataset

    ds = open_dataset(dataset, start=start, end=end, frequency=frequency, select=["2t", "z", "lsm"])
    var = {name: i for i, name in enumerate(ds.variables)}
    first = ds[0]  # the orography and the land-sea mask do not change with time
    with ThreadPoolExecutor(num_threads) as pool:
        t2m = np.stack(list(pool.map(lambda i: ds[i][var["2t"], 0], range(len(ds)))))
    longitudes = np.asarray(ds.longitudes, dtype=np.float64)
    return Era5Sample(
        t2m=t2m.astype(np.float32),
        orography=(first[var["z"], 0] / GRAVITY).astype(np.float32),
        land_sea_mask=first[var["lsm"], 0].astype(np.float32),
        latitudes=np.asarray(ds.latitudes, dtype=np.float64),
        longitudes=np.where(longitudes > 180.0, longitudes - 360.0, longitudes),  # from [0, 360) to [-180, 180)
        times=np.asarray(ds.dates).astype("datetime64[h]"),
    )


def octahedral_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes (in degrees) of the octahedral reduced Gaussian grid O<n>.

    The grid has 2n rows of latitude (the Gaussian latitudes). The row number i (from 1 at the pole
    to n at the equator) has 4 i + 16 points, at equal distances in longitude, from longitude 0.
    """
    roots, _ = np.polynomial.legendre.leggauss(2 * n)
    row_lats = np.rad2deg(np.arcsin(roots))[::-1]  # from north to south
    lats, lons = [], []
    for k, lat in enumerate(row_lats):
        i = k + 1 if k < n else 2 * n - k
        num = 4 * i + 16
        lon = np.arange(num) * 360.0 / num
        lats.append(np.full(num, lat))
        lons.append(np.where(lon >= 180.0, lon - 360.0, lon))
    return np.concatenate(lats), np.concatenate(lons)


def to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Points on the unit sphere (Cartesian coordinates) from latitudes and longitudes in degrees."""
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def nearest_coarse_node(fine_lat, fine_lon, coarse_lat, coarse_lon) -> np.ndarray:
    """For each fine point, the index of the nearest coarse point."""
    _, idx = cKDTree(to_xyz(coarse_lat, coarse_lon)).query(to_xyz(fine_lat, fine_lon))
    return idx


def coarsen(field: np.ndarray, assignment: np.ndarray, num_coarse: int) -> np.ndarray:
    """Mean of the fine values in the area of each coarse point. The last axis is the fine point.

    The area of a coarse point contains all the fine points that are nearer to it than to the other
    coarse points. Each value of ``assignment`` gives the coarse point of one fine point.
    """
    counts = np.bincount(assignment, minlength=num_coarse)
    lead = field.shape[:-1]
    flat = field.reshape(-1, field.shape[-1])
    out = np.zeros((flat.shape[0], num_coarse), dtype=np.float64)
    np.add.at(out.T, assignment, flat.T)
    return (out / counts).reshape(*lead, num_coarse).astype(np.float32)


def time_features(times: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Cosine and sine of the local solar time and of the day of the year.

    Returns an array with shape (num_times, num_points, 4).
    """
    hours = (times - times.astype("datetime64[D]")).astype("timedelta64[h]").astype(np.float32)
    day_of_year = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(np.float32)
    local_time = (hours[:, None] + longitudes[None, :] / 15.0) % 24.0
    doy = np.broadcast_to(day_of_year[:, None], local_time.shape)
    return np.stack(
        [
            np.cos(2 * np.pi * local_time / 24.0),
            np.sin(2 * np.pi * local_time / 24.0),
            np.cos(2 * np.pi * doy / 365.25),
            np.sin(2 * np.pi * doy / 365.25),
        ],
        axis=-1,
    ).astype(np.float32)


class DownscalingDataset(Dataset):
    """Pairs of (coarse 2 m temperature, fine 2 m temperature) for a downscaling task.

    The fine grid is the grid of the data (O48). The coarse grid is the octahedral grid O<coarse_n>.
    The coarse fields are the means of the fine fields in the area of each coarse point.

    Each item is a tuple ``(x_src, x_dst, y)``:

    - ``x_src`` (num_coarse_nodes, 3): coarse temperature, coarse orography, coarse land-sea mask.
    - ``x_dst`` (num_fine_nodes, 10): fine orography, fine land-sea mask, sin/cos of latitude and
      longitude, sin/cos of the local solar time and of the day of the year.
    - ``y`` (num_fine_nodes, 1): fine temperature.

    All temperatures are normalised with the mean and standard deviation of the training data.
    The orography is in km.
    """

    src_feature_names = ["t2m (coarse)", "orography (coarse)", "land-sea mask (coarse)"]
    dst_feature_names = [
        "orography (fine)",
        "land-sea mask (fine)",
        "sin(lat)",
        "cos(lat)",
        "sin(lon)",
        "cos(lon)",
        "cos(local time)",
        "sin(local time)",
        "cos(day of year)",
        "sin(day of year)",
    ]

    def __init__(self, sample: Era5Sample, coarse_n: int = 12, t2m_mean: float | None = None, t2m_std: float | None = None):
        self.sample = sample
        self.coarse_n = coarse_n
        self.t2m_mean = float(sample.t2m.mean()) if t2m_mean is None else t2m_mean
        self.t2m_std = float(sample.t2m.std()) if t2m_std is None else t2m_std

        self.fine_lat, self.fine_lon = sample.latitudes, sample.longitudes
        self.coarse_lat, self.coarse_lon = octahedral_grid(coarse_n)
        self.assignment = nearest_coarse_node(self.fine_lat, self.fine_lon, self.coarse_lat, self.coarse_lon)

        # Coarse fields (the "low-resolution model")
        n = self.num_coarse_nodes
        self.t2m_fine = sample.t2m
        self.t2m_coarse = coarsen(sample.t2m, self.assignment, n)
        self.orog_fine = sample.orography / 1000.0
        self.orog_coarse = coarsen(sample.orography, self.assignment, n) / 1000.0
        self.lsm_fine = sample.land_sea_mask
        self.lsm_coarse = coarsen(sample.land_sea_mask, self.assignment, n)

        lat_rad, lon_rad = np.deg2rad(self.fine_lat), np.deg2rad(self.fine_lon)
        self.dst_static = np.stack(
            [self.orog_fine, self.lsm_fine, np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
            axis=-1,
        ).astype(np.float32)
        self.dst_time = time_features(sample.times, self.fine_lon)
        self.src_static = np.stack([self.orog_coarse, self.lsm_coarse], axis=-1).astype(np.float32)

    @property
    def times(self) -> np.ndarray:
        return self.sample.times

    @property
    def num_coarse_nodes(self) -> int:
        return self.coarse_lat.size

    @property
    def num_fine_nodes(self) -> int:
        return self.fine_lat.size

    @property
    def num_src_features(self) -> int:
        return len(self.src_feature_names)

    @property
    def num_dst_features(self) -> int:
        return len(self.dst_feature_names)

    def time_index(self, date: str) -> int:
        return int(np.where(self.times == np.datetime64(date))[0][0])

    def normalise(self, t2m: np.ndarray) -> np.ndarray:
        return (t2m - self.t2m_mean) / self.t2m_std

    def denormalise(self, t2m: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return t2m * self.t2m_std + self.t2m_mean

    def __len__(self) -> int:
        return self.t2m_fine.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_src = np.concatenate([self.normalise(self.t2m_coarse[idx])[:, None], self.src_static], axis=-1)
        x_dst = np.concatenate([self.dst_static, self.dst_time[idx]], axis=-1)
        y = self.normalise(self.t2m_fine[idx])[:, None]
        return torch.from_numpy(x_src.astype(np.float32)), torch.from_numpy(x_dst), torch.from_numpy(y.astype(np.float32))


# ---------------------------------------------------------------------------------------------------------------------
# Model wrapper and training
# ---------------------------------------------------------------------------------------------------------------------


class GraphDownscaler(nn.Module):
    """Apply a bipartite GNN layer to a batch of samples that share the same graph.

    PyTorch Geometric layers expect node features with the shape (num_nodes, num_features).
    To process a batch, this wrapper puts the B copies of the graph side by side in one large
    graph with B disconnected parts. The node indices of copy b move by b * num_nodes.

    Residual learning: if ``interp_weights`` is given, the GNN learns a correction to a baseline,

        output = T_interp + lapse_rate * (z_fine - z_interp) + GNN(...)

    T_interp and z_interp are the coarse temperature and orography, interpolated with
    ``interp_weights``. ``lapse_rate`` is in normalised temperature units per km (use
    STANDARD_LAPSE_RATE / t2m_std). If it is None, the baseline is the interpolation only.
    The wrapper expects the feature order of DownscalingDataset: x_src = [t2m, orography, ...]
    and x_dst = [orography, ...].
    """

    def __init__(
        self,
        gnn: nn.Module,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        interp_weights: torch.Tensor | None = None,
        lapse_rate: float | None = None,
    ):
        super().__init__()
        self.gnn = gnn
        self.register_buffer("edge_index", edge_index.to(torch.int64))
        self.register_buffer("edge_attr", None if edge_attr is None else edge_attr.to(torch.float32))
        self.register_buffer("interp_weights", None if interp_weights is None else interp_weights.reshape(-1).to(torch.float32))
        self.lapse_rate = lapse_rate
        self._cache = {}

    def interpolate(self, x_src: torch.Tensor, num_dst: int) -> torch.Tensor:
        """Weighted sum of the source values on the incoming edges of each target node."""
        messages = x_src[:, self.edge_index[0], :] * self.interp_weights[None, :, None]
        out = torch.zeros(x_src.shape[0], num_dst, x_src.shape[-1], dtype=x_src.dtype, device=x_src.device)
        return out.index_add_(1, self.edge_index[1], messages)

    def batched_edges(self, batch_size: int, num_src: int, num_dst: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        key = (batch_size, num_src, num_dst, self.edge_index.device)
        if key not in self._cache:
            offsets = torch.arange(batch_size, device=self.edge_index.device)[:, None]
            src = (self.edge_index[0][None, :] + offsets * num_src).reshape(-1)
            dst = (self.edge_index[1][None, :] + offsets * num_dst).reshape(-1)
            attr = None if self.edge_attr is None else self.edge_attr.repeat(batch_size, 1)
            self._cache = {key: (torch.stack([src, dst]), attr)}
        return self._cache[key]

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        # x_src: (batch, num_src_nodes, num_src_features), x_dst: (batch, num_dst_nodes, num_dst_features)
        batch_size, num_src, _ = x_src.shape
        num_dst = x_dst.shape[1]
        edge_index, edge_attr = self.batched_edges(batch_size, num_src, num_dst)
        out = self.gnn(
            x_src.reshape(batch_size * num_src, -1),
            x_dst.reshape(batch_size * num_dst, -1),
            edge_index,
            edge_attr,
        )
        out = out.reshape(batch_size, num_dst, -1)
        if self.interp_weights is not None:
            out = out + self.baseline(x_src, x_dst)
        return out

    def baseline(self, x_src: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        """Interpolated coarse temperature, with a lapse-rate correction if lapse_rate is set."""
        interp = self.interpolate(x_src[..., :2], x_dst.shape[1])  # coarse t2m and orography
        base = interp[..., :1]
        if self.lapse_rate is not None:
            base = base + self.lapse_rate * (x_dst[..., :1] - interp[..., 1:2])
        return base


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model: nn.Module, dataset: DownscalingDataset, batch_size: int = 16) -> np.ndarray:
    """Return the predicted fine temperature in kelvin, shape (num_samples, num_fine_nodes)."""
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for x_src, x_dst, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        out = model(x_src.to(device), x_dst.to(device))
        preds.append(out[..., 0].cpu().numpy())
    return dataset.denormalise(np.concatenate(preds, axis=0))


def train_model(
    model: nn.Module,
    train_ds: DownscalingDataset,
    val_ds: DownscalingDataset,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 2e-3,
    max_minutes: float | None = None,
) -> dict[str, list[float]]:
    """Train with the mean squared error on normalised temperature.

    Returns the history of the training loss and of the validation RMSE (in kelvin).
    """
    device = next(model.parameters()).device
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(loader))
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_rmse": [], "step_loss": []}
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_src, x_dst, y in loader:
            x_src, x_dst, y = x_src.to(device), x_dst.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_src, x_dst), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            history["step_loss"].append(loss.item())
        epoch_loss /= len(loader)

        val_rmse = rmse(predict(model, val_ds, batch_size=2 * batch_size), val_ds.t2m_fine)
        history["train_loss"].append(epoch_loss)
        history["val_rmse"].append(val_rmse)
        elapsed = time.time() - t_start
        print(f"Epoch {epoch:2d}/{epochs}  train loss {epoch_loss:.4f}  validation RMSE {val_rmse:.3f} K  ({elapsed:.0f} s)")
        if max_minutes is not None and elapsed > 60 * max_minutes and epoch < epochs:
            print(f"Stop after {epoch} epochs: the time limit of {max_minutes} min is reached.")
            break
    return history


def rmse(pred: np.ndarray, truth: np.ndarray, axis=None) -> float | np.ndarray:
    return np.sqrt(np.mean((pred - truth) ** 2, axis=axis))


# ---------------------------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------------------


def map_axes(fig, nrows: int, ncols: int, index: int, extent=None):
    """Add a map axis (with coastlines if cartopy is available).

    With ``extent=None``, the map shows the globe (Robinson projection). Else, ``extent`` is
    (lon_min, lon_max, lat_min, lat_max) and the map uses a latitude-longitude projection.
    """
    if HAS_CARTOPY:
        projection = ccrs.Robinson() if extent is None else ccrs.PlateCarree()
        ax = fig.add_subplot(nrows, ncols, index, projection=projection)
        if extent is None:
            ax.set_global()
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, color="0.3")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m" if extent is None else "50m"), linewidth=0.5, color="k")
    else:
        extent = extent or (-180, 180, -90, 90)
        ax = fig.add_subplot(nrows, ncols, index)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(np.clip(np.mean(extent[2:]), -60, 60))))
    return ax


def _transform():
    return {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}


def _marker_size(ax, num_points: int, extent) -> float:
    """A marker size (in points^2) that covers the map with ``num_points`` points without large gaps."""
    fig = ax.get_figure()
    bbox = ax.get_position()
    width_pt = bbox.width * fig.get_figwidth() * 72
    if extent is None:
        frac = 1.0  # all the points are in the map
        area_pt = 0.75 * width_pt * width_pt / 2  # the Robinson map is approximately an ellipse, height = width / 2
    else:
        lat0, lat1 = np.deg2rad(extent[2]), np.deg2rad(extent[3])
        frac = (extent[1] - extent[0]) / 360.0 * (np.sin(lat1) - np.sin(lat0)) / 2.0  # part of the sphere
        aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
        area_pt = width_pt * width_pt * aspect
    return 1.3 * area_pt / max(num_points * frac, 1)


def grid_cells(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Corners of the grid cells of a reduced Gaussian grid (rows of constant latitude).

    Each row of latitude has points at equal distances in longitude. The cell of a point goes to
    half the distance to its neighbours. Returns (corners, index): ``corners`` has the shape
    (num_cells, 4, 2) with (lon, lat) in degrees, and ``index`` gives the grid point of each cell.
    A cell that crosses the longitude 180° is split into two cells. Returns None if the grid has
    a different structure.
    """
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    rows, row_of_point, count = np.unique(-lat, return_inverse=True, return_counts=True)
    rows = -rows  # from north to south
    if rows.size > 2000:
        return None
    bounds = np.concatenate([[90.0], 0.5 * (rows[1:] + rows[:-1]), [-90.0]])
    dlon = 360.0 / count[row_of_point]
    north, south = bounds[row_of_point], bounds[row_of_point + 1]
    west, east = lon - dlon / 2, lon + dlon / 2
    index = np.arange(lat.size)
    # Split the cells that cross 180 degrees east or west
    cross_e, cross_w = east > 180.0, west < -180.0
    west = np.concatenate([west, np.full(cross_e.sum(), -180.0), west[cross_w] + 360.0])
    east = np.concatenate([np.minimum(east, 180.0), east[cross_e] - 360.0, np.full(cross_w.sum(), 180.0)])
    west[: lat.size] = np.maximum(west[: lat.size], -180.0)
    north = np.concatenate([north, north[cross_e], north[cross_w]])
    south = np.concatenate([south, south[cross_e], south[cross_w]])
    index = np.concatenate([index, index[cross_e], index[cross_w]])
    corners = np.stack(
        [np.stack([west, south], -1), np.stack([east, south], -1), np.stack([east, north], -1), np.stack([west, north], -1)],
        axis=1,
    )
    return corners, index


def plot_field(ax, lat, lon, values, cmap=TEMP_CMAP, extent=None, vmin=None, vmax=None, **kwargs):
    """Plot a field on the points of a reduced Gaussian grid, as coloured grid cells.

    For other grids, the function plots coloured markers.
    """
    from matplotlib.collections import PolyCollection

    values = np.asarray(values)
    cells = grid_cells(lat, lon)
    if cells is None:
        s = _marker_size(ax, np.size(lat), extent)
        return ax.scatter(lon, lat, c=values, s=s, cmap=cmap, marker="s", linewidths=0, vmin=vmin, vmax=vmax,
                          **_transform(), **kwargs)
    corners, index = cells
    if extent is not None:  # keep only the cells near the map, to make the plot faster
        lo, la = corners[..., 0].mean(1), corners[..., 1].mean(1)
        keep = (lo > extent[0] - 10) & (lo < extent[1] + 10) & (la > extent[2] - 10) & (la < extent[3] + 10)
        corners, index = corners[keep], index[keep]
    if HAS_CARTOPY:
        xy = ax.projection.transform_points(ccrs.PlateCarree(), corners[..., 0].ravel(), corners[..., 1].ravel())
        corners = xy[:, :2].reshape(corners.shape)
    coll = PolyCollection(corners, array=values[index], cmap=cmap, edgecolors="face", linewidths=0.4, **kwargs)
    coll.set_clim(vmin, vmax)
    ax.add_collection(coll)
    if not HAS_CARTOPY:
        ax.autoscale_view()
    return coll


def plot_points(ax, lat, lon, values=None, **kwargs):
    if values is None:
        return ax.scatter(lon, lat, **_transform(), **kwargs)
    return ax.scatter(lon, lat, c=values, **_transform(), **kwargs)


def plot_edges(ax, src_lat, src_lon, dst_lat, dst_lon, edge_index, values=None, cmap="viridis", **kwargs):
    """Draw edges as straight lines from source to target nodes (in degrees)."""
    edge_index = np.asarray(edge_index)
    segments = np.stack(
        [
            np.stack([src_lon[edge_index[0]], src_lat[edge_index[0]]], axis=-1),
            np.stack([dst_lon[edge_index[1]], dst_lat[edge_index[1]]], axis=-1),
        ],
        axis=1,
    )
    lc = LineCollection(segments, cmap=cmap if values is not None else None, **_transform(), **kwargs)
    if values is not None:
        lc.set_array(np.asarray(values))
    ax.add_collection(lc)
    return lc


def plot_loss_curves(histories: dict[str, dict[str, list[float]]], baselines: dict[str, float] | None = None):
    """Plot the training loss and the validation RMSE of one or more models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name, hist in histories.items():
        steps = np.arange(1, len(hist["step_loss"]) + 1)
        steps_per_epoch = len(hist["step_loss"]) / len(hist["train_loss"])
        ax1.plot(steps / steps_per_epoch, hist["step_loss"], alpha=0.3, lw=0.8)
        ax1.plot(np.arange(1, len(hist["train_loss"]) + 1), hist["train_loss"], "o-", label=name, color=ax1.lines[-1].get_color())
        ax2.plot(np.arange(1, len(hist["val_rmse"]) + 1), hist["val_rmse"], "o-", label=name)
    for i, (name, value) in enumerate((baselines or {}).items()):
        ax2.axhline(value, ls="--", color=f"C{i + len(histories)}", label=name)
    ax1.set(xlabel="epoch", ylabel="MSE (normalised units)", yscale="log", title="Training loss")
    ax2.set(xlabel="epoch", ylabel="RMSE [K]", title="Validation RMSE (2021)")
    ax1.legend()
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _in_extent(lat, lon, extent):
    if extent is None:
        return np.ones(np.shape(lat), dtype=bool)
    return (lon >= extent[0]) & (lon <= extent[1]) & (lat >= extent[2]) & (lat <= extent[3])


def plot_downscaling_example(ds: DownscalingDataset, idx: int, predictions: dict[str, np.ndarray], extent=None):
    """Show the coarse input, the truth, the predictions and their errors for one time step.

    Top row: the temperature. Bottom row: the coarse and fine orography, then the errors.
    """
    fine_in = _in_extent(ds.fine_lat, ds.fine_lon, extent)
    truth = ds.t2m_fine[idx] - 273.15
    coarse = ds.t2m_coarse[idx] - 273.15
    vmin, vmax = np.percentile(truth[fine_in], [1, 99])
    ncols = 2 + len(predictions)
    panel_w = 4.6
    if extent is None:
        panel_h = panel_w / 2
    else:
        panel_h = panel_w * (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig = plt.figure(figsize=(panel_w * ncols, 2 * panel_h + 2.0), layout="constrained")

    def field_panel(k, lat, lon, values, title, **kw):
        ax = map_axes(fig, 2, ncols, k, extent)
        im = plot_field(ax, lat, lon, values, extent=extent, **kw)
        ax.set_title(title)
        return ax, im

    field_panel(1, ds.coarse_lat, ds.coarse_lon, coarse, f"Input: temperature (O{ds.coarse_n})", vmin=vmin, vmax=vmax)
    date = np.datetime_as_string(ds.times[idx], unit="h").replace("T", " ")
    _, im = field_panel(2, ds.fine_lat, ds.fine_lon, truth, f"Truth: ERA5 (O48), {date} UTC", vmin=vmin, vmax=vmax)
    for i, (name, pred) in enumerate(predictions.items()):
        field_panel(3 + i, ds.fine_lat, ds.fine_lon, pred[idx] - 273.15, name, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=fig.axes[:ncols], orientation="horizontal", fraction=0.05, pad=0.04, aspect=50, label="2 m temperature [°C]")

    orog_axes = []
    for k, (lat, lon, orog, title) in enumerate(
        [
            (ds.coarse_lat, ds.coarse_lon, ds.orog_coarse * 1000, f"Input: orography (O{ds.coarse_n})"),
            (ds.fine_lat, ds.fine_lon, ds.orog_fine * 1000, "Input: orography (O48)"),
        ]
    ):
        ax, im_orog = field_panel(ncols + 1 + k, lat, lon, orog, title, cmap=OROG_CMAP, vmin=-500, vmax=5000)
        orog_axes.append(ax)
    fig.colorbar(im_orog, ax=orog_axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=30, label="height [m]")

    err_axes = []
    for i, (name, pred) in enumerate(predictions.items()):
        err = pred[idx] - ds.t2m_fine[idx]
        ax, im_err = field_panel(
            ncols + 3 + i, ds.fine_lat, ds.fine_lon, err,
            f"Error: {name}\nRMSE = {rmse(pred[idx][fine_in], ds.t2m_fine[idx][fine_in]):.2f} K",
            cmap=ERROR_CMAP, vmin=-5, vmax=5,
        )
        err_axes.append(ax)
    if err_axes:
        fig.colorbar(im_err, ax=err_axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=30, label="prediction − truth [K]")
    return fig


def plot_rmse_maps(ds: DownscalingDataset, predictions: dict[str, np.ndarray], reference: str | None = None, extent=None):
    """Map of the RMSE over all validation times for each prediction.

    If ``reference`` is the name of one prediction, the last panel shows the RMSE of the last
    prediction minus the RMSE of the reference (blue: the last prediction is better).
    """
    names = list(predictions)
    ncols = len(names) + (reference is not None)
    fig = plt.figure(figsize=(5.2 * ncols, 3.6), layout="constrained")
    node_rmse = {name: rmse(pred, ds.t2m_fine, axis=0) for name, pred in predictions.items()}
    axes = []
    for i, name in enumerate(names):
        ax = map_axes(fig, 1, ncols, i + 1, extent)
        im = plot_field(ax, ds.fine_lat, ds.fine_lon, node_rmse[name], cmap="magma_r", vmin=0, vmax=4, extent=extent)
        ax.set_title(f"{name}\nmean RMSE = {rmse(predictions[name], ds.t2m_fine):.2f} K")
        axes.append(ax)
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.06, pad=0.06, label="RMSE [K]")
    if reference is not None:
        ax = map_axes(fig, 1, ncols, ncols, extent)
        diff = node_rmse[names[-1]] - node_rmse[reference]
        im = plot_field(ax, ds.fine_lat, ds.fine_lon, diff, cmap=ERROR_CMAP, vmin=-1.5, vmax=1.5, extent=extent)
        ax.set_title(f"RMSE({names[-1]}) − RMSE({reference})")
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.06, label="[K]")
    return fig


def plot_attention(
    ds: DownscalingDataset,
    edge_index: np.ndarray,
    attention: np.ndarray,
    target_nodes: dict[str, int],
    title: str = "",
):
    """Draw the incoming edges of some target nodes, each in a small map.

    The width and the colour of an edge show its attention weight.
    """
    edge_index = np.asarray(edge_index)
    fig = plt.figure(figsize=(4.2 * len(target_nodes), 4.4), layout="constrained")
    lc = None
    for k, (name, node) in enumerate(target_nodes.items()):
        edges = np.where(edge_index[1] == node)[0]
        sources = edge_index[0, edges]
        lons = np.concatenate([[ds.fine_lon[node]], ds.coarse_lon[sources]])
        lats = np.concatenate([[ds.fine_lat[node]], ds.coarse_lat[sources]])
        half = max(np.ptp(lons) / 2, np.ptp(lats) / 2) + 3.0
        extent = (lons.mean() - half, lons.mean() + half, lats.mean() - half, lats.mean() + half)
        ax = map_axes(fig, 1, len(target_nodes), k + 1, extent)
        plot_field(ax, ds.fine_lat, ds.fine_lon, ds.orog_fine * 1000, cmap="Greys", vmin=-1500, vmax=6000, extent=extent)
        plot_points(ax, ds.coarse_lat, ds.coarse_lon, s=40, c="k", marker="s", zorder=3)
        lc = plot_edges(
            ax, ds.coarse_lat, ds.coarse_lon, ds.fine_lat, ds.fine_lon, edge_index[:, edges],
            values=attention[edges], cmap="plasma", linewidths=1 + 8 * attention[edges], zorder=4,
        )
        lc.set_clim(0, 1)
        plot_points(ax, ds.fine_lat[node], ds.fine_lon[node], s=90, c="cyan", edgecolors="k", zorder=5)
        ax.set_title(name)
    if lc is not None:
        fig.colorbar(lc, ax=fig.axes, orientation="horizontal", fraction=0.05, aspect=40,
                     label="attention weight (mean over the heads)")
    fig.suptitle(title)
    return fig


def plot_reduced_gaussian_grids(grids: dict[str, tuple[np.ndarray, np.ndarray]], min_lat: float = 40.0):
    """Show reduced Gaussian grids around the North Pole, and the number of points on each circle of latitude.

    ``grids`` maps a name (for example "O12") to the arrays (latitudes, longitudes) in degrees. The maps
    show the area north of ``min_lat``. The colour of a point shows its quarter of the globe (90 degrees
    of longitude). On an octahedral grid, each quarter is a triangle: a face of the octahedron.
    """
    import matplotlib.path as mpath
    from matplotlib.collections import PolyCollection

    ncols = len(grids) + 1
    fig = plt.figure(figsize=(5.0 * ncols, 5.2), layout="constrained")
    for k, (name, (lat, lon)) in enumerate(grids.items()):
        quarter = (np.mod(lon, 360.0) // 90).astype(int)
        keep = lat >= min_lat
        if HAS_CARTOPY:
            ax = fig.add_subplot(1, ncols, k + 1, projection=ccrs.NorthPolarStereo())
            ax.set_extent([-180, 180, min_lat - 2, 90], crs=ccrs.PlateCarree())
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.set_boundary(mpath.Path(np.stack([np.sin(theta), np.cos(theta)], -1) * 0.5 + 0.5), transform=ax.transAxes)
            ax.add_feature(cfeature.LAND, color="0.92")
            ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.4, color="0.6")
            corners, _ = grid_cells(lat, lon)
            cap = corners[:, 0, 1] >= min_lat - 10.0
            xy = ax.projection.transform_points(ccrs.PlateCarree(), corners[cap, :, 0].ravel(), corners[cap, :, 1].ravel())
            ax.add_collection(PolyCollection(xy[:, :2].reshape(-1, 4, 2), facecolors="none", edgecolors="0.6", linewidths=0.4))
            ax.scatter(lon[keep], lat[keep], s=max(4.0, 30000 / lat.size), c=[f"C{q}" for q in quarter[keep]],
                       zorder=3, transform=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(1, ncols, k + 1)
            ax.scatter(lon[keep], lat[keep], s=4, c=[f"C{q}" for q in quarter[keep]])
        rows = np.unique(lat).size
        ax.set_title(f"{name}: {rows} circles of latitude, {lat.size} points\n(north of {min_lat:.0f}°N; colour: quarter of the globe)")

    ax = fig.add_subplot(1, ncols, ncols)
    for k, (name, (lat, lon)) in enumerate(grids.items()):
        row_lats, counts = np.unique(lat, return_counts=True)
        ax.plot(row_lats, counts, "o-", ms=3, color=f"C{k + 4}", label=f"{name} (reduced)")
        ax.plot(row_lats, np.full(row_lats.size, counts.max()), "--", color=f"C{k + 4}", label=f"{name} without reduction")
    ax.set(xlabel="latitude [°]", ylabel="number of points on the circle of latitude", ylim=(0, 270),
           title="Points on each circle of latitude")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def show_images(paths: dict[str, str], width: float = 6.5):
    """Show some PNG files side by side. ``paths`` maps a title to a file name."""
    fig, axes = plt.subplots(1, len(paths), figsize=(width * len(paths), width), squeeze=False)
    for ax, (title, path) in zip(axes[0], paths.items()):
        ax.imshow(plt.imread(path))
        ax.set_title(title, fontsize=13, weight="bold")
        ax.axis("off")
    fig.tight_layout()
    return fig


def show_html(path: str, height: int = 600):
    """Show an HTML file (for example a plot of ``anemoi-graphs inspect``) in the notebook.

    The HTML files of ``anemoi-graphs inspect`` contain a full copy of the plotly.js library
    (approximately 5 MB). To keep the notebook small, we replace this copy by a link to the same
    version of plotly.js on a public server (CDN). Your web browser must have internet access.
    """
    import html
    import re
    from pathlib import Path

    from IPython.display import HTML

    text = Path(path).read_text()
    version = re.search(r"plotly\.js v(\d+\.\d+\.\d+)", text)
    if version:
        header = text.find("plotly.js v")
        start = text.rfind("<script", 0, header)
        end = text.find("</script>", header) + len("</script>")
        cdn = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{version.group(1)}/plotly.min.js"
        text = text[:start] + f'<script src="{cdn}"></script>' + text[end:]
    return HTML(
        f'<iframe srcdoc="{html.escape(text, quote=True)}" width="100%" height="{height}" '
        'style="border: 1px solid #ccc;"></iframe>'
    )


def nearest_node(lat: np.ndarray, lon: np.ndarray, point_lat: float, point_lon: float) -> int:
    """Index of the node nearest to a point (in degrees)."""
    return int(np.argmin(great_circle_km(lat, lon, point_lat, point_lon)))


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between points given in degrees (haversine formula)."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
