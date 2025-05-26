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
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from movement.io import load_poses
from movement.kinematics import compute_pairwise_distances, compute_speed
from movement.transforms import scale
from movement.utils.vector import compute_norm, convert_to_unit

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
# %%
# Print the version of movement that is being used (for reproducibility)
os.system("movement info")

# %%
# Load unwrapped and cleaned tracks
# ---------------------------------
# These come from the output of the "clean_unwrapped_tracks.py" notebook.

repo_root = Path(__file__).parent
data_dir = repo_root / "data"
assert data_dir.exists()

approach_to_path = {  # paths are relative to data_dir
    "itk-all": (
        "approach-itk-all/20250325_2228_id_unwrapped_20250403_161408_clean.h5"
    ),
    "sfm-interp": (
        "approach-sfm-interp/20250325_2228_id_sfm_interp_PCS_2d_20250516_155745_clean.h5" 
    ),
    "sfm-itk-interp": (
        "approach-sfm-itk-interp/20250325_2228_id_sfm_itk_interp_PCS_2d_20250517_230807_clean.h5"
    ),
}

# Select which approach to use
approach = "sfm-interp"  # can be either "itk-all", "sfm-interp" or "sfm-itk-interp"
file_path_relative = approach_to_path[approach]
file_path = data_dir / Path(file_path_relative)
assert file_path.exists()

# %%
# Now, let's load the data
ds = load_poses.from_file(file_path, source_software="SLEAP", fps=30)
print(ds)

# %%
# Compute body length per individual
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
# Compute the alignment of each individual with the average body orientation
# across time
# -----------------------------------------------------------------------------

# Compute average **unit** body vector across all individuals per frame
# (if vectors are unit, their average is the same as the resultant vector)
body_vector_unit_avg = convert_to_unit(body_vector_filtered).mean("individuals")
print(body_vector_unit_avg.shape)


# Compute dot product between each individual's unit body vector and
# the average unit body vector
body_vector_filtered_unit = convert_to_unit(body_vector_filtered)
cos_body_vector = xr.dot(
    body_vector_filtered_unit,
    body_vector_unit_avg,
    dims=["space"],
)  # the dot product is the cosine of the angle between the two unit vectors


# Plot the alignment of each individual with the average unit body vector
# across time
fig, ax = plt.subplots()
im = ax.matshow(
    cos_body_vector,
    aspect="auto",
    cmap="coolwarm",
)
cbar = plt.colorbar(im)
cbar.set_label("alignment with average unit body vector")
ax.get_images()[0].set_clim(-1, 1)
ax.set_xlabel("individuals")
ax.set_ylabel("frame")

# %%
# Compute the herd's polarisation
# -------------------------------
# We define polarisation as the norm of the resultant unit body vector per frame.
# The resultant unit vector is the average of the unit body vectors across individuals.
# 1. convert body length vectors to unit vectors
# 2. compute the resultant of the unit vectors
# 3. compute the norm of the resultant unit vector as the polarisation

polarisation = compute_norm(body_vector_unit_avg)
polarisation.name = "Herd polarisation"


# %%
# Compute average speed and compare with polarisation
# ---------------------------------------------------
# First let's scale the data to body length units
# (this is not necessary, but it makes the plots easier to interpret)

position_scaled = scale(
    ds.position, factor=1 / body_length_median.item(), space_unit="body_length"
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
distances = distances.assign_coords(id_pair=list(distances_dict.keys()))
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
# Combine polarisation, speed, and distances into a single plot
# -------------------------------------------------------------

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
# Nearest neighbors
# -----------------

# Creat an array of NaN with shape (time, individuals)
distances_nn = xr.DataArray(
    data=np.nan,
    dims=["time", "individuals"],
    coords=dict(
        time=position_scaled.time,
        individuals=position_scaled.individuals,
    ),
)

distances_nn.name = "Distance (body lengths)"

# Compute the nearest neighbor distance for each individual
for id in position_scaled.individuals.values:
    pairs_with_id = [pair for pair in distances_dict.keys() if id in pair]
    distances_nn.loc[dict(individuals=id)] = distances.sel(
        id_pair=pairs_with_id).min(
            dim="id_pair", skipna=True
        )

print(distances_nn)

# %%
# Plot the nearest neighbor distances across time
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
distances_nn.transpose().plot(ax=ax)
# Don't show the y labels
ax.set_yticklabels([])
ax.set_title("Distance to nearest neighbor")
ax.set_xlabel("Time (s)")


# %%
# Plot mean, mix, and max nearest neighbor distances across time
fig, ax = plt.subplots(1, 1)
distances_nn.mean(dim="individuals").plot(label="mean", ax=ax)
distances_nn.min(dim="individuals").plot(label="min", ax=ax)
distances_nn.max(dim="individuals").plot(label="max", ax=ax)
ax.legend()
ax.set_title("Distance to nearest neighbor")
ax.set_xlabel("Time (s)")

# %%