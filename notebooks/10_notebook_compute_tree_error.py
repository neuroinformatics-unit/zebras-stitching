# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
from movement.io import load_poses
from movement.utils.vector import compute_norm
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

# for interactive plots
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
repo_root = Path(__file__).parent
data_dir = repo_root / "data"
assert data_dir.exists()


# Paths to subdirectories per method
# paths relative to data_dir
data_subdir = {
    "itk-all": "approach-itk-all",
    "sfm-interp": "approach-sfm-interp",
    "sfm-itk-interp": "approach-sfm-itk-interp",
}

# Filenames of unwrapped tree data per method
# (we don't need to clean the tree data, since it is already "reliable")
# paths relative to relevant subdir in data_dir
filename_tree_data = {  
    "itk-all": "21Jan_007_tracked_trees_reliable_sleap_unwrapped_20250516_154821.h5",
    "sfm-interp": "21Jan_007_tracked_trees_reliable_sleap_sfm_interp_PCS_2d_20250516_160103.h5",
    "sfm-itk-interp": "21Jan_007_tracked_trees_reliable_sleap_sfm_itk_interp_PCS_2d_20250517_231433.h5",
}

# Filenames of clean zebra data per method
# (we use it to get the median body length)
# paths relative to relevant subdir in data_dir
filename_zebra_data = {
    "itk-all": "20250325_2228_id_unwrapped_20250403_161408_clean.h5",
    "sfm-interp": "20250325_2228_id_sfm_interp_PCS_2d_20250516_155745_clean.h5",
    "sfm-itk-interp": "20250325_2228_id_sfm_itk_interp_PCS_2d_20250517_230807_clean.h5",
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select method used to unwrap trajectories to retrieve relevant data

approach = "sfm-interp"  # can be either "itk-all" or "sfm-interp" or "sfm-itk-interp"
path_tree_data = data_dir / data_subdir[approach] / filename_tree_data[approach]
path_zebra_data = data_dir / data_subdir[approach] / filename_zebra_data[approach]

print(f"Using approach: {approach}")
print(f"Tree data path: {path_tree_data.parent.name}/{path_tree_data.name}")
print(f"Zebra data path: {path_zebra_data.parent.name}/{path_zebra_data.name}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute median zebra body length

ds_zebras = load_poses.from_file(path_zebra_data, source_software="SLEAP")

body_vector = ds_zebras.position.sel(keypoints="H") - ds_zebras.position.sel(
    keypoints="T"
)

body_length = compute_norm(body_vector)

# Compute stats for all individuals across all frames
body_length_std = body_length.std().values
body_length_mean = body_length.mean().values
body_length_median = body_length.median().values

print(f"Body length std: {body_length_std} (a.u.)")
print(f"Body length mean: {body_length_mean} (a.u.)")
print(f"Body length median: {body_length_median} (a.u.)")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load tree data and plot in 2d plane

ds_trees = load_poses.from_file(path_tree_data, source_software="SLEAP")

colormap = plt.cm.turbo
colors = colormap(np.linspace(0, 1, len(ds_trees.individuals.values)))
fig, ax = plt.subplots(1, 1)
for i, tree_id in enumerate(ds_trees.individuals.values):
    ax.scatter(
        ds_trees.position.sel(individuals=tree_id).values[:, 0],
        ds_trees.position.sel(individuals=tree_id).values[:, 1],
        c=colors[i],
    )
ax.set_aspect("equal")
# reverse y-axis
ax.invert_yaxis()
ax.set_xlabel("x (a.u.)")
ax.set_ylabel("y (a.u.)")
ax.set_title(f" Tree positions - {approach}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute standard deviation of each tree x,y-coordinate
std_per_tree = ds_trees.position.std(dim=["time", "keypoints"])
std_per_tree_normalized = std_per_tree / body_length_median

fig, ax = plt.subplots(1, 1)
std_per_tree_normalized.sel(space="x").plot(ax=ax)
std_per_tree_normalized.sel(space="y").plot(ax=ax)
ax.set_xlabel("Tree ID")
ax.set_ylabel("std / median zebra body length")
ax.legend()
ax.set_title(f"Tree coordinates std - {approach}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute treedispersion as average distance to each tree centroid
# normalize by zebra median body length

# Compute tree centroids
tree_centroids = ds_trees.mean(dim=["time", "keypoints"])

# Compute mean distance to each tree centroid
distance_to_centroid_per_tree = {}
for tree_id in ds_trees.individuals.values:
    dist_to_centroid = cdist(
        tree_centroids.position.sel(individuals=tree_id).values.reshape(
            -1, 2
        ),  # (1, 2)
        ds_trees.sel(individuals=tree_id).position.values.reshape(
            -1, 2
        ),  # (n_frames, 2)
        metric="euclidean",
    )
    distance_to_centroid_per_tree[tree_id] = {
        "mean": np.nanmean(dist_to_centroid),
        "max": np.nanmax(dist_to_centroid),
        "min": np.nanmin(dist_to_centroid),
        "std": np.nanstd(dist_to_centroid),
        "n_samples": np.sum(~np.isnan(dist_to_centroid))
    }


# Normalize by zebra median body length
dist_to_centroid_normalized = {
    "mean": {
        tree_id: distance_to_centroid_per_tree[tree_id]["mean"] / body_length_median
        for tree_id in ds_trees.individuals.values
    },
    "max": {
        tree_id: distance_to_centroid_per_tree[tree_id]["max"] / body_length_median
        for tree_id in ds_trees.individuals.values
    },
    "min": {
        tree_id: distance_to_centroid_per_tree[tree_id]["min"] / body_length_median
        for tree_id in ds_trees.individuals.values
    },
    "std": {
        tree_id: distance_to_centroid_per_tree[tree_id]["std"] / body_length_median
        for tree_id in ds_trees.individuals.values
    },
    "n_samples": {
        tree_id: distance_to_centroid_per_tree[tree_id]["n_samples"]
        for tree_id in ds_trees.individuals.values
    },
}

# Print weighted mean across all trees
list_mean_per_tree = list(dist_to_centroid_normalized['mean'].values())
list_n_samples_per_tree = list(dist_to_centroid_normalized['n_samples'].values())
list_weights = list_n_samples_per_tree / np.sum(list_n_samples_per_tree)

weighted_mean = np.sum(list_mean_per_tree * list_weights)

print(
    "Weighted mean normalized distance to centroid across all trees: "
    f"{weighted_mean}"
)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot
fig, ax = plt.subplots(1, 1)
ax.plot(
    dist_to_centroid_normalized["mean"].keys(),
    dist_to_centroid_normalized["mean"].values(),
)
ax.set_xlabel("Tree ID")
ax.set_ylabel("d tree-centroid normalized by median zebra body length")


# Plot in 2D, larger error as a larger marker
fig, ax = plt.subplots(1, 1)
for i, tree_id in enumerate(ds_trees.individuals.values):
    ax.scatter(
        np.nanmean(ds_trees.position.sel(individuals=tree_id).values[:, 0]),
        np.nanmean(ds_trees.position.sel(individuals=tree_id).values[:, 1]),
        edgecolor=colors[i],
        s=dist_to_centroid_normalized["mean"][tree_id] * 100,
        facecolors="none",
    )
ax.set_aspect("equal")
# reverse y-axis
ax.invert_yaxis()
ax.set_xlabel("x (a.u.)")
ax.set_ylabel("y (a.u.)")
ax.set_title(f" Tree positions - {approach}")


# Plot in 2D, color representing error?
# fig, ax = plt.subplots(1, 1)
# # colormap = plt.cm.turbo
# # color_array_error
# for i, tree_id in enumerate(ds_trees.individuals.values):
#     ax.scatter(
#         np.nanmean(ds_trees.position.sel(individuals=tree_id).values[:, 0]),
#         np.nanmean(ds_trees.position.sel(individuals=tree_id).values[:, 1]),
#         c=dist_to_centroid_normalized["mean"][tree_id],
#         cmap="turbo",
#     )
# ax.set_aspect("equal")
# # reverse y-axis
# ax.invert_yaxis()
# ax.set_xlabel("x (a.u.)")
# ax.set_ylabel("y (a.u.)")
# ax.set_title(f" Tree positions - {approach}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export results as latex table


# Define dataframe
df = pd.DataFrame(
    data=[
        dist_to_centroid_normalized["mean"],
        dist_to_centroid_normalized["max"],
        dist_to_centroid_normalized["min"],
        dist_to_centroid_normalized["std"],
        dist_to_centroid_normalized["n_samples"],
    ],
    index=[
        "mean",
        "max",
        "min",
        "std",
        "n_samples",
    ],
).T


df["n_samples"] = df["n_samples"].astype(int)

print(df.head())

# Export to latex
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_latex(
    data_dir
    / data_subdir[approach]
    / f"trees_dist_to_centroid_normalized_{approach}_{timestamp}.tex",
    index=True,
    float_format="%.3f",  
    caption=(
        f"Distance to centroid normalized by median zebra body length. Approach: {approach}. "
        f"Weighted normalised mean: {weighted_mean:.3f} (body length units)"
    ),
)

 # %%
