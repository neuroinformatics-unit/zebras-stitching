"""Compute metrics of herd behaviour
====================================
This notebook computes metrics of herd behaviour from the unwrapped
and cleaned zebra tracks, so it should be executed after
"clean_unwrapped_tracks.py".

It computes the following metrics:
- Herd polarisation
- Average speed
- Inter-zebra distances

"""

# %%
# Imports
# -------
from pathlib import Path
import os

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from movement.io import load_poses
from movement.kinematics import compute_pairwise_distances, compute_speed
from movement.transforms import scale
from movement.utils.vector import compute_norm, convert_to_unit

# %%
# Print the version of movement that is being used (for reproducibility)
os.system("movement info")

# %%
# Load unwrapped and cleaned tracks
# ---------------------------------
# These come from the output of the "clean_unwrapped_tracks.py" notebook.

repo_root = Path(__file__).parent
data_dir = repo_root / "data"
video_dir = repo_root / "videos"
assert data_dir.exists()
assert video_dir.exists()

filename = "20250325_2228_id_unwrapped_clean_sleap.h5"
file_path = data_dir / filename
video_path = video_dir / "21Jan_007.mp4"
background_path = video_dir / "21Jan_007_unwrapped_bacground.png"
for path in [file_path, video_path, background_path]:
    assert path.exists()

# %%
# Now, let's load the data
ds = load_poses.from_file(file_path, source_software="SLEAP", fps=30)
print(ds)

# %%
# Compute and visualise body vectors
# ----------------------------------
# We define the body vector as the vector originating at keypoint "T" (tail)
# and ending at keypoint "H" (head).

body_vector = ds.position.sel(keypoints="H") - ds.position.sel(keypoints="T")
# Select body vectors for which norm is outside mean +- 2 std
body_length = compute_norm(body_vector)
body_length_std = body_length.std()
body_length_mean = body_length.mean()
body_length_median = body_length.median()

# %% 
# Plot body length histogram, coloured by individuals

fig, ax = plt.subplots()
counts, bins, _ = body_length.plot.hist(bins=100)
ax.vlines(
    body_length_mean,
    ymin=0,
    ymax=np.max(counts),
    color="red",
    linestyle="-",
    label="mean body length",
)
lower_bound = body_length_mean - 2 * body_length_std
upper_bound = body_length_mean + 2 * body_length_std
for bound in [lower_bound, upper_bound]:
    ax.vlines(
        bound,
        ymin=0,
        ymax=np.max(counts),
        color="red",
        linestyle="--",
        label="mean +- 2 std",
    )
ax.set_ylim(0, np.max(counts))
ax.set_xlabel("body length (pixels)")
ax.set_ylabel("counts")
ax.legend()

# %%
# We keep only the body vectors that are within 2 std of the mean
# (this is a bit arbitrary, but we want to remove outliers).

body_vector_filtered = body_vector.where(
    np.logical_and(
        body_length <= body_length_mean + 2 * body_length_std,
        body_length >= body_length_mean - 2 * body_length_std,
    )
)

# Compute average body vector per frame
body_vector_avg = body_vector_filtered.mean("individuals")
print(body_vector_avg.shape)

# %%
# Compute the herd's polarisation
# -------------------------------
# We define polarisation as the mean resultant length of the body vectors:
# 1. convert body length vectors to unit vectors
# 2. compute the mean of the unit vectors
# 3. compute the norm of the mean resultant unit vector as the polarisation

# Compute average **unit** body vector per frame
# (if unit, average is the same as resultant vector)
body_vector_unit_avg = convert_to_unit(body_vector_filtered).mean("individuals")
polarisation = compute_norm(body_vector_unit_avg)
polarisation.name = "Herd polarisation"


# %%
# Compute average speed and compare with polarisation
# ---------------------------------------------------
# First let's scale the data to body length units
# (this is not necessary, but it makes the plots easier to interpret)

position_scaled = scale(
    ds.position,
    factor=1/body_length_median.item(),
    space_unit="body_length"
)

# %%
# Compute speed of each zebra's centroid, and then average across individuals

centroid = position_scaled.mean("keypoints")
speed_avg = compute_speed(centroid).mean("individuals")
speed_avg.name = "Average speed (body lengths/s)"
log10_speed_avg = np.log10(speed_avg)
log10_speed_avg.name = "log10 Average speed (body lengths/s)"

# %%
# plot polarisation and color by log of mean centroid-speed

fig, ax = plt.subplots()
sc = ax.scatter(
    x=polarisation.time,
    y=polarisation,
    c=log10_speed_avg,
    s=5,
    cmap="turbo",
    # rescale color map to 1st and 99th percentiles
    vmin=log10_speed_avg.quantile(0.01).item(),
    vmax=log10_speed_avg.quantile(0.99).item(),
)
cbar = plt.colorbar(sc)
cbar.set_label(log10_speed_avg.name)

ax.set_xlabel("Time (s)")
ax.set_ylabel(polarisation.name)
ax.set_title("Herd polarisation and speed")


# %%
# Compute pairwise distances
# --------------------------

# Generate a dict mapping from pair names to data arrays containing
# inter-centroid distances over time for that pair of individuals
distances_dict = compute_pairwise_distances(
    centroid,
    dim="individuals",
    pairs="all",
    metric="euclidean",
)

# %%
# Let's stack the dict of distances into a single data array
distances = xr.concat(
    distances_dict.values(),
    dim="id_pair",
)
distances = distances.assign_coords(
    id_pair=list(distances_dict.keys())
)
distances.name = "Distance (body lengths)"
print(distances)


# %%
# Plot all pair-wise distances across time as a heatmap

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
distances.plot(ax=ax)
# Don't show the y labels
ax.set_yticklabels([])
ax.set_title("Pair-wise distances between zebras")
ax.set_xlabel("Time (s)")


# %%
# Plot mean and max distances between pairs of zebras across time

fig, ax = plt.subplots(1, 1)
distances.mean(dim="id_pair").plot(label="mean", ax=ax)
distances.max(dim="id_pair").plot(label="max", ax=ax)
ax.legend()
ax.set_title("Pair-wise distances between zebras")
ax.set_xlabel("Time (s)")
ax.set_ylabel(distances.name)

# %%
# Combine polarisation, speed, and distances into a signle plot

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot polarisation over time, coloured by log10 speed
sc = ax[0].scatter(
    x=polarisation.time,
    y=polarisation,
    c=log10_speed_avg,
    s=5,
    cmap="turbo",
    # rescale color map to 1st and 99th percentiles
    vmin=log10_speed_avg.quantile(0.01).item(),
    vmax=log10_speed_avg.quantile(0.99).item(),
)
ax[0].set_ylabel(polarisation.name)

# Make a separate axis for the colorbar
cbar_ax = fig.add_axes([0.15, 0.92, 0.8, 0.02])
cbar = plt.colorbar(sc, orientation="horizontal", cax=cbar_ax)
cbar.set_label(log10_speed_avg.name)

# Plot mean and max distances between pairs of zebras across time
distances.mean(dim="id_pair").plot(label="mean", ax=ax[1])
distances.max(dim="id_pair").plot(label="max (herd extent)", ax=ax[1])
ax[1].legend()
ax[1].set_ylabel("Inter-zebra distance\n(body lengths)")

ax[1].set_xlabel("Time (s)")
ax[1].set_xlim(polarisation.time.min(), polarisation.time.max())

fig.subplots_adjust(
    top=0.8,
    bottom=0.1,
    left=0.15,
    right=0.95,
    hspace=0.3,
)

# %%
