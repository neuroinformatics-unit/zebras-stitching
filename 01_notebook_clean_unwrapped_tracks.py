"""Clean noisy zebra tracks
===========================
The following steps are performed:

- Load the unwrapped zebra tracks from file.
- Use movement to filter out low-confidence keypoints and keypoints that
  are too far from the previous position.
- Save the cleaned data to file.
"""

# %%
# Imports
# -------
from pathlib import Path
import os

import xarray as xr
from matplotlib import pyplot as plt

from movement.filtering import filter_by_confidence
from movement.io import load_poses, save_poses
from movement.kinematics import compute_displacement
from movement.plots import plot_centroid_trajectory
from movement.utils.vector import compute_norm

# %%
# Print the version of movement that is being used (for reproducibility)
os.system("movement info")


# %%
# Load unwrapped pose tracks
# --------------------------
# First let's define paths

repo_root = Path(__file__).parent
data_dir = repo_root / "data"
video_dir = repo_root / "videos"
assert data_dir.exists()
assert video_dir.exists()

filename = "20250325_2228_id_unwrapped.h5"
file_path = data_dir / filename
video_path = video_dir / "21Jan_007.mp4"
background_path = video_dir / "21Jan_007_unwrapped_background.png"
for path in [file_path, video_path, background_path]:
    assert path.exists()

# %%
# Now, let's load the unwrapped poses
ds = load_poses.from_file(file_path, source_software="SLEAP", fps=30)

# %%
# Visualise the data
# --------------------
# Let' overlay the trajectories on the unwrapped background image

fig, ax = plt.subplots(1, 1)
ax.imshow(plt.imread(background_path))

# Plot centroid trajectories of the individuals
individuals = ds.individuals.values
# Cycle through colors
cmap = ["r", "g", "b", "c", "m", "y"]
colors = [
    cmap[i % len(cmap)] for i in range(len(individuals))
]

for id, color in zip(individuals, colors):
    plot_centroid_trajectory(
        ds.position,
        individual=id,
        ax=ax,
        linestyle="-",
        marker=".",
        s=0.1,
        c=color,
        alpha=0.2,
        label=id,
    )
ax.set_title("Individual trajectories")

# %%
# Let's visualise the x, y coordinates of the keypoints across time

ds.position.plot(
    x="time", col="keypoints", row="space",
    hue="individuals", aspect=1.5, size=2.5, add_legend=False
)
plt.xlabel("Time (s)")

# %%
# Let's compute and plot the frame-to-frame distance for each keypoint
# (i.e. the distance between the position of the keypoint at time t and
# the position at time t-1). This will help us identify implausible
# jumps in the data (sicne zebras can't teleport!).

displacement = compute_displacement(ds.position)
distance = compute_norm(displacement)
distance.name = "Distance (pixels)"

distance.plot(
    x="time", col="keypoints",
    hue="individuals", aspect=1.5, size=2.5, add_legend=False
)
plt.xlabel("Time (s)")

# We see some implausible jumps that are likely due to
# erroneous predictions. Let's try to filter them out.

# %%
# Filter by confidence
# --------------------

# %%
# Plot a histogram of confidence values
colors = ["green", "purple"]

for i, keypoint in enumerate(ds.keypoints.values):
    ds.confidence.sel(keypoints=keypoint).plot.hist(
        bins=50, histtype="step", color=colors[i], label=keypoint
    )
    plt.legend(title="Keypoint")
    plt.title("Confidence histogram")

# %%
# Filter out low-confidence keypoints

conf_thresh = 0.9
position_filt = filter_by_confidence(
    ds.position, ds.confidence, threshold=conf_thresh,
)


# %%
# Plot the frame-to-frame distance for each keypoint before and after
# filtering.

# Before filtering
distance.plot(
    x="time", col="keypoints",
    hue="individuals", aspect=1.5, size=2.5, add_legend=False
)
y_max = plt.gca().get_ylim()[1]
plt.ylim(0, y_max)
plt.xlabel("Time (s)")
plt.suptitle("Before filtering")

# After filtering
displacement_filt = compute_displacement(position_filt)
distance_filt = compute_norm(displacement_filt)
distance_filt.name = distance.name
distance_filt.plot(
    x="time", col="keypoints",
    hue="individuals", aspect=1.5, size=2.5, add_legend=False
)
plt.ylim(0, y_max)  # enforce same y-axis limits for a fair comparison
plt.xlabel("Time (s)")
plt.suptitle(f"Confidence > {conf_thresh}")


# %%
# We got rid of some but not all of the implausible jumps.

# %%
# Filter by distance
# -------------------
# This time we will directly filter out the implausible jumps by setting
# the position to NaN if the distance is greater than a threshold.
# We will base the threshold on the average zebra body length.

# Define the body length of the zebras in pixels
body_axis = ds.position.sel(keypoints="T") - ds.position.sel(keypoints="H")
body_length = compute_norm(body_axis)
# Compute the median body length (across time and individuals)
median_body_length = body_length.median(skipna=True).item()
print(f"Median body length: {median_body_length:.2f} pixels")

dist_thresh = 2 * median_body_length  # 2 body lengths
print(f"Distance threshold: {dist_thresh:.2f} pixels (2x body length)")

# Apply the distance threshold to confidence-filtered data
position_clean = position_filt.where(distance < dist_thresh)

# Plot the frame-to-frame distance for each keypoint after distance filtering
displacement_clean = compute_displacement(position_clean)
distance_clean = compute_norm(displacement_clean)
distance_clean.name = distance.name
distance_clean.plot(
    x="time", col="keypoints",
    hue="individuals", aspect=1.5, size=2.5, add_legend=False
)
plt.ylim(0, y_max)  # enforce same y-axis limit as before for a fair comparison
plt.xlabel("Time (s)")
plt.suptitle(f"Distance < {dist_thresh:.2f} pixels")

# %%
# Plot the clean trajectories on the background image
fig, ax = plt.subplots(1, 1)
ax.imshow(plt.imread(background_path))
# Plot centroid trajectories of the individuals
for id, color in zip(individuals, colors):
    plot_centroid_trajectory(
        position_clean,
        individual=id,
        ax=ax,
        linestyle="-",
        marker=".",
        s=0.1,
        c=color,
        alpha=0.2,
        label=id,
    )
ax.set_title("Individual trajectories")

# %%
# Count missing values
# -------------------
# Let's see how many point we lost int the filtering process.
# We first define a function for computing the number of NaN values in the
# data array.


def count_nan_percent(data: xr.DataArray) -> float:
    """
    Compute the percentage of missing values in the data array.

    A values is considered missing if at least one of its spatial
    coordinates is NaN.
    """
    n_nan = data.isnull().any(["space"]).sum().item()
    n_total = data.size / data.sizes["space"]
    return n_nan / n_total * 100


# %%
# Count the percentage of missing values in the original and after
# each filtering step.

missing_in_original = count_nan_percent(ds.position)
missing_in_filtered = count_nan_percent(position_filt)
missing_in_clean = count_nan_percent(position_clean)
print(
    f"Missing values in original data: {missing_in_original:.2f}%"
)
print(
    f"Missing values in confidence-filtered data: {missing_in_filtered:.2f}%"
)
print(
    f"Missing values in distance-filtered data: {missing_in_clean:.2f}%"
)

# %%
# Save the cleaned data
# ---------------------
# We will use the SLEAP analysis file format to save the cleaned data.
# We will keep the confidence values from the original dataset

ds_clean = xr.Dataset(
    {
        "position": position_clean,
        "confidence": ds.confidence,
    }
)
ds_clean.attrs = ds.attrs.copy()


save_poses.to_sleap_analysis_file(
    ds_clean, data_dir / filename.replace(".h5", "_clean_sleap.h5"),
)
