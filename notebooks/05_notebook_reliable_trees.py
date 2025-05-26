# %%
# Imports

from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from movement.io import load_bboxes, save_poses
from movement.kinematics import compute_path_length, compute_speed
from pathlib import Path

# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

data_dir = Path(__file__).parent / "data"
filename = data_dir / "21Jan_007_tracked_trees_20250505_100631.csv"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read VIA-tracks file as a movement dataset

ds = load_bboxes.from_via_tracks_file(
    file_path=filename, use_frame_numbers_from_file=False
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise the tree trajectories

n_trees = len(ds.individuals)
print(f"Number of total trees: {n_trees}")

ds.position.plot.line(
    x="time",
    row="space",
    hue="individuals",
    add_legend=False,
    aspect=2,
    size=2.5,
)
plt.xlim(0, ds.time.size - 1);


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Measure number of valid points for each tree trajectory
# (ie number of samples that are not nan)

valid = ds.position.notnull().all(["space"]).sum(["time"])
valid.plot.hist(bins=100);
plt.title("Histogram ofvalid samples per tree")
plt.xlabel("Number of valid samples")
plt.ylabel("Count")
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# keep only trees with > 400 valid points

long_trees = ds.where(valid > 400, drop=True)
print(
    f"Number of trees detected in >400 frames: {len(long_trees.individuals)}"
)

long_trees.position.plot.line(
    x="time",
    row="space",
    hue="individuals",
    add_legend=False,
    aspect=2,
    size=2.5,
)
plt.xlim(0, long_trees.time.size - 1);


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Construct a new poses dataset with the "surviving" trees

long_trees_ds = xr.Dataset(
    {
        "position": long_trees.position,
        "confidence": long_trees.confidence,
    }
).expand_dims(
    {"keypoints": ["centroid"]},
    axis=-2,
)

print(long_trees_ds)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise the speed of "surviving" trees over time

speed = compute_speed(long_trees.position)

speed.plot.line(
    x="time",
    hue="individuals",
    add_legend=False,
    aspect=3,
    size=3,
);

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Drop trees with implausibly high speeds

speed_limit = 10  # pixels/frame
# Drop trees whose max speed is > speed_limit
long_slow_trees_ds = long_trees_ds.where(
    (speed.max("time") < speed_limit), drop=True
)
print(f"Number of reliable trees: {len(long_slow_trees_ds.individuals)}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise the position of the reliable trees over time

long_slow_trees_ds.position.sel(keypoints="centroid").plot.line(
    x="time",
    row="space",
    hue="individuals",
    add_legend=False,
    aspect=2,
    size=2.5,
)
plt.xlim(0, long_slow_trees_ds.time.size - 1);

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save the reliable trees to a new file
# (reliable means: detected in >400 frames and max speed < 10 pixels/frame)

save_poses.to_sleap_analysis_file(
    long_slow_trees_ds,
    file_path=data_dir / "21Jan_007_tracked_trees_reliable_sleap.h5",
)

# %%
