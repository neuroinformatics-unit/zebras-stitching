"""Compute metrics of herd behaviour
====================================
This notebook computes metrics of herd behaviour from the unwrapped
and cleaned zebra tracks, so it should be executed after
"clean_unwrapped_tracks.py".

"""

# %%
# Imports
# -------
from datetime import datetime
from functools import partial
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sleap_io as sio
import xarray as xr
from movement.io import load_poses
from movement.kinematics import compute_pairwise_distances, compute_speed
from movement.transforms import scale
from movement.utils.vector import compute_norm, convert_to_unit

# from PIL import Image

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
# %%
# Figure settings for the preprint
# --------------------------------
# These apply to every figure created below, so exported figures share the
# same typography. Text is kept as editable text in SVG exports (rather than
# converted to outlines), so labels can be adjusted in Inkscape.
mpl.rcParams.update(
    {
        "svg.fonttype": "none",
        "axes.labelsize": 12,
    }
)

# %%
# Print the version of movement that is being used (for reproducibility)
os.system("movement info")

# %%
# Load unwrapped and cleaned tracks
# ---------------------------------
# These come from the output of the "clean_unwrapped_tracks.py" notebook.
video_path = Path(__file__).parents[2] / "video" / "21Jan_007.mp4"

repo_root = Path(__file__).parents[1]
data_dir = repo_root / "data"
assert data_dir.exists()

approach_to_path = {  # paths are relative to data_dir
    "itk-all": ("approach-itk-all/20250325_2228_id_unwrapped_20250403_161408_clean.h5"),
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
ds = load_poses.from_file(file_path, source_software="SLEAP", fps=29.97)
print(ds)

# %%
# Compute body length per individual
# ----------------------------------
# We define the body vector as the vector originating at keypoint "T" (tail)
# and ending at keypoint "H" (head).

body_vector = ds.position.sel(keypoints="H") - ds.position.sel(keypoints="T")

# Compute body length stats
body_length = compute_norm(body_vector)
body_length_std = body_length.std()
body_length_mean = body_length.mean()
body_length_median = body_length.median()

# %%
# Plot body length histogram, coloured by individuals
# Mark limits for mean +- 2 std
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

# Compute average body vector per frame after filtering
body_vector_avg = body_vector_filtered.mean("individuals")
print(body_vector_avg.shape)

# %%
# Compute the polarisation vector
# -----------------------------------------------------------------------------

# Compute average **unit** body vector across all individuals per frame
# This is the polarisation vector
body_vector_filtered_unit = convert_to_unit(body_vector_filtered)
polarisation_vector = body_vector_filtered_unit.mean("individuals")
print(polarisation_vector.shape)

# The polarisation vector is not a unit vector!
# Its norm is the "polarisation", i.e.
# norm(body_vector_filtered_unit_avg) == polarisation
polarisation = compute_norm(polarisation_vector)
polarisation.name = "Herd polarisation"

print(polarisation)  # norm is not unit

# %%
# Plot the evolution of the polarisation vector

# Visualise for a selected frame
# For reference, the frame index with max polarisation: 3160
frame_index = 3160
video_array = sio.load_video(video_path)

fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, (3, 4))


# plot polarisation norm over time
ax1.plot(polarisation, zorder=1)
ax1.scatter(
    frame_index,
    polarisation.isel(time=frame_index),
    color="black",
    marker="o",
    s=50,
    zorder=2,
)
ax1.vlines(
    frame_index,
    ymin=0,
    ymax=1.1,
    color="k",
    linestyle="--",
)
ax1.set_ylim(0, 1.1)
ax1.set_xlabel("frame")
ax1.set_ylabel("polarisation")

# plot individual unit body vectors
ax2.quiver(
    np.zeros((len(body_vector_filtered_unit.individuals), 1)),
    np.zeros((len(body_vector_filtered_unit.individuals), 1)),
    body_vector_filtered_unit.isel(time=frame_index, space=0).values,
    body_vector_filtered_unit.isel(time=frame_index, space=1).values,
    angles="xy",
    scale=1,
    scale_units="xy",
    zorder=1,
)

# plot polarisation vector
ax2.quiver(
    np.zeros((len(polarisation_vector), 1)),
    np.zeros((len(polarisation_vector), 1)),
    polarisation_vector.isel(time=frame_index, space=0).values,
    polarisation_vector.isel(time=frame_index, space=1).values,
    angles="xy",
    scale=1,
    scale_units="xy",
    color="red",
    linewidths=1.5,
    edgecolors="r",
    zorder=3,
)

# plot unit polarisation vector
ax2.quiver(
    np.zeros((len(polarisation_vector), 1)),
    np.zeros((len(polarisation_vector), 1)),
    convert_to_unit(polarisation_vector).isel(time=frame_index, space=0).values,
    convert_to_unit(polarisation_vector).isel(time=frame_index, space=1).values,
    angles="xy",
    scale=1,
    scale_units="xy",
    edgecolor=(0.3, 0, 0.7),
    linewidths=0.5,
    headlength=0,
    zorder=2,
)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_xticks([-1, 0, 1])
ax2.set_yticks([-1, 0, 1])
ax2.invert_yaxis()  # to match image coordinate system in video
ax2.set_aspect("equal")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"Unit body vectors at frame {frame_index}", fontsize=10)


# # Load image from file
# image_path = (
#     repo_root.parent / "video-frames" / f"{frame_index:05d}.png"
# )  # Replace with actual path
# image = Image.open(image_path)

# # Convert to RGB if needed (in case it's RGBA or other format)
# if image.mode != "RGB":
#     image = image.convert("RGB")


ax3.matshow(video_array[frame_index][:, :, ::-1])  # bgr -> rgb
ax3.set_xlabel("x (pixels)")
ax3.set_ylabel("y (pixels)")
ax3.set_title(f"Frame {frame_index}", fontsize=10)

plt.tight_layout(w_pad=2)

# %%
# Inspect the alignment of each individual with the unit polarisation vector
# -----------------------------------------------------------------------------

# Compute the dot product between each individual's unit body vector and
# the unit polarisation vector
# the dot product is the cosine of the angle between the two unit vectors
cos_body_vector = xr.dot(
    body_vector_filtered_unit,
    convert_to_unit(polarisation_vector),
    dims=["space"],
)

# plot
fig, ax = plt.subplots(1, 1)
# plot alignment with unit polarisation vector per individual
# We use a diverging colormap with three distinct colors for -1, 0, and 1
# (red - white - blue). Empty values (i.e. frames in which the body vector is
# nans) are shown in grey, so they are distinguishable from the white midpoint.
cmap = mpl.colormaps["RdBu"].with_extremes(bad="0.6")

im = ax.matshow(
    cos_body_vector.transpose("individuals", "time"),
    aspect="auto",
    cmap=cmap,
    vmin=-1,
    vmax=1,
)
cbar = plt.colorbar(im)
cbar.set_label("alignment with average unit body vector")

# convert frames to seconds in x-axis (one tick every 50 seconds)
time_ticks_step = int(round(50 * ds.fps))
time_ticks = np.arange(0, len(cos_body_vector.time), time_ticks_step)
time_labels = [f"{t:.0f}" for t in cos_body_vector.time.values[0:-1:time_ticks_step]]
ax.set_xticks(time_ticks)
ax.set_xticklabels(time_labels)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

ax.set_xlabel("time (s)")
ax.set_ylabel("individuals")

plt.tight_layout(w_pad=2.5)

# Save the figure as SVG (text stays editable, see figure settings above)
fig.savefig(
    repo_root / f"alignment_with_polarisation_{approach}.svg",
    format="svg",
    bbox_inches="tight",
)


# %%
# Compute average speed and compare with polarisation
# ---------------------------------------------------
# First let's scale the data to body length units

position_scaled = scale(
    ds.position, factor=1 / body_length_median.item(), space_unit="body_length"
)

# %%
# Compute speed of each zebra's centroid, and then average across individuals

centroid = position_scaled.mean("keypoints")
speed = compute_speed(centroid)
speed_avg = speed.mean("individuals")
speed_avg.name = "Average speed (body lengths/s)"
log10_speed_avg = np.log10(speed_avg)
log10_speed_avg.name = "log10 Average speed (body lengths/s)"


# %%
# Plot speed per individual across time
# --------------------------------------
fig, ax = plt.subplots()
im = ax.matshow(
    speed,
    aspect="auto",
    cmap="viridis",
)

# convert frames to seconds in y-axis (one tick every 50 seconds)
time_ticks_step = int(round(50 * ds.fps))
time_ticks = np.arange(0, len(speed.time), time_ticks_step)
time_labels = [f"{t:.0f}" for t in speed.time.values[0:-1:time_ticks_step]]
ax.set_yticks(time_ticks)
ax.set_yticklabels(time_labels)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

ax.set_xlabel("individual")
ax.set_ylabel("time (s)")

# add colorbar
cbar = plt.colorbar(im)
cbar.set_label("speed (BL/s)")

# All values above the 99th percentile are shown in yellow in the colorbar
speed_99th_percentile = np.round(speed.quantile(0.99)).item()
ax.get_images()[0].set_clim(0, speed_99th_percentile)
ax.set_title("Speed per individual across time")
# %%
# Plot polarisation and color by log of mean centroid-speed

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
    distances_nn.loc[dict(individuals=id)] = distances.sel(id_pair=pairs_with_id).min(
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
# Compute polarisation animation
# -----------------------------------


# Define a function to plot the evolution of the polarisation vector
# across time
def polarisation_plot_single_frame(
    frame_index,
    polarisation_norm,
    polarisation_vector,
    body_vector_unit,
    video_array,
    list_axes,
):
    # clear pre-existing plots
    for ax in list_axes:
        ax.clear()

    ax1, ax2, ax3 = list_axes

    # plot polarisation norm over time
    ax1.plot(polarisation_norm, color="tab:blue", zorder=1)
    ax1.scatter(
        frame_index,
        polarisation_norm.isel(time=frame_index),
        color="black",
        marker="o",
        s=50,
        zorder=2,
    )
    ax1.vlines(
        frame_index,
        ymin=0,
        ymax=1.1,
        color="k",
        linestyle="--",
    )
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("polarisation")

    # plot individual unit body vectors
    ax2.quiver(
        np.zeros((len(body_vector_unit.individuals), 1)),
        np.zeros((len(body_vector_unit.individuals), 1)),
        body_vector_unit.isel(time=frame_index, space=0).values,
        body_vector_unit.isel(time=frame_index, space=1).values,
        angles="xy",
        scale=1,
        scale_units="xy",
        zorder=1,
    )

    # plot polarisation vector
    ax2.quiver(
        np.zeros((len(polarisation_vector), 1)),
        np.zeros((len(polarisation_vector), 1)),
        polarisation_vector.isel(time=frame_index, space=0).values,
        polarisation_vector.isel(time=frame_index, space=1).values,
        angles="xy",
        scale=1,
        scale_units="xy",
        color="red",
        linewidths=1.5,
        edgecolors="r",
        zorder=3,
    )

    # plot unit polarisation vector
    ax2.quiver(
        np.zeros((len(polarisation_vector), 1)),
        np.zeros((len(polarisation_vector), 1)),
        convert_to_unit(polarisation_vector).isel(time=frame_index, space=0).values,
        convert_to_unit(polarisation_vector).isel(time=frame_index, space=1).values,
        angles="xy",
        scale=1,
        scale_units="xy",
        edgecolor=(0.3, 0, 0.7),
        linewidths=0.5,
        headlength=0,
        zorder=2,
    )
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_xticks([-1, 0, 1])
    ax2.set_yticks([-1, 0, 1])
    ax2.invert_yaxis()  # to match image coordinate system in video
    ax2.set_aspect("equal")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"Unit body vectors at frame {frame_index}", fontsize=10)

    # plot video frame
    ax3.matshow(video_array[frame_index][:, :, ::-1])  # bgr -> rgb
    ax3.set_xlabel("x (pixels)")
    ax3.set_ylabel("y (pixels)")
    ax3.set_title(f"Frame {frame_index}", fontsize=10)

    return ax


# Plot for first frame
frame_index = 0
fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, (3, 4))

ax1.clear()
ax2.clear()
ax3.clear()
polarisation_plot_single_frame(frame_index)
plt.tight_layout(w_pad=2)

start_frame = 0
end_frame = len(polarisation) - 1
frame_step = 1

# Create the animation
anim = animation.FuncAnimation(
    fig,
    partial(
        polarisation_plot_single_frame,
        polarisation_norm=polarisation,
        polarisation_vector=polarisation_vector,
        body_vector_unit=body_vector_filtered_unit,
        video_array=video_array,
        list_axes=[ax1, ax2, ax3],
    ),
    frames=range(start_frame, end_frame, frame_step),
    interval=np.round((1 / ds.fps) * 1000, 2),  # delay between frames in ms
    repeat=False,
    blit=True,  # True for better performance if possible
)

plt.show()

# %%
# Save the animation (can be very slow)
# by default, the fps is 1/interval in the animation object
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save as GIF
# anim.save(f"polarisation_animation_{timestamp}.gif", writer="pillow")


# To save as MP4 (requires ffmpeg):
anim.save(f"polarisation_animation_{timestamp}.mp4", writer='ffmpeg')


# %%
