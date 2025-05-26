"""A notebook to express keypoints in a world coordinate system"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio
import xarray as xr
from movement.io import load_poses, save_poses
from movement.plots import plot_centroid_trajectory
from datetime import datetime

import skimage as ski
from skimage.transform import warp

# Uncomment the following line for interactive plotting
# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data paths

repo_root = Path(__file__).parents[0]
data_dir = repo_root / "data"
transforms_dir = data_dir / "elastix"

# Wrapped data
filename_zebras = Path("20250325_2228_id.slp")
filename_trees = Path("21Jan_007_tracked_trees_reliable_sleap.h5")
filename = filename_trees  # choose the file you want to unwrap
file_path = data_dir / filename

# Elastix transforms
transforms_file = transforms_dir / "out_euler_frame_masked.csv"

# Video file
video_file = repo_root / "videos" / "21Jan_007.mp4"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read video
video = sio.load_video(video_file)
print(video.shape)

n_frames = video.shape[0]
frame_shape = video.shape[1:]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read trajectory data as a movement dataset

ds = load_poses.from_file(file_path, source_software="SLEAP")

# get position array
position_array = ds.position

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read transforms from elastix
#
# Transforms are Euler (i.e., 2D rot+translation only)
# They are expressed:
# - as a rotation around the center of the image
# - and a translation vector
# The rotation is expressed as an angle in radians
# The translation is expressed as a vector in pixels
# The transform given for frame f is the transform required to
# go from frame f to frame f-1

# itk-elastix computes transformations that map points
# from the fixed image (f) domain to the moving image (f-1) domain.
# See: https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example08_SimpleTransformix.ipynb

# read as pandas dataframe
transforms_df = pd.read_csv(transforms_file)

# Add row to dataframe with transform for first frame
# For f=0, the transform from the current to the previous frame
# is the identity (i.e., no rotation, no translation)
transforms_df = pd.concat(
    [pd.DataFrame({"theta": 0, "tx": 0, "ty": 0}, index=[0]), transforms_df],
    ignore_index=True,
)


# Check as many transforms as frames
assert transforms_df.shape[0] == n_frames
print(f"Number of transforms: {transforms_df.shape}")
print(f"Number of frames: {n_frames}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute the position array in homogeneous coordinates
# (x, y, 1) instead of (x, y)
# I use "h" for the third homog coord instead of "z" for clarity

position_array_homogeneous = xr.concat(
    [
        ds["position"],
        xr.full_like(ds["position"].sel(space="x"), 1).expand_dims(space=["h"]),
    ],
    dim="space",
)

print(position_array_homogeneous)  # shape (time, 3, keypoints, individuals)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute change of basis matrix:

# ICS = image coordinate system. Its origin is the top-left corner of the image.

# Q_corner_to_centre is a change of basis matrix
# that transforms coordinates from the ICS (corner) to the ICS (centre)

# When applied from the left to a (column) vector,
# it transforms its coordinates, to express them in
# a a coordinate system that is parallel to the ICS but with
# its origin in the image centre (ICS_centre)
v_corner_to_centre = np.array(tuple(s // 2 for s in frame_shape[:2]))
Q_corner_to_centre = np.eye(3)
Q_corner_to_centre[:2, 2] = -v_corner_to_centre


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&&&&&&&
# Compute array of (accum) rotation matrices per frame --- ok?

# ICS0_centre is the coordinate system fixed to the ground,
# and parallel to the ICS at frame 0. Its origin is the projection
# of the image centre in the ground at frame 0.

# The rotation is accumulated because we compute it as the
# cumulative sum of the rotations from f to f-1


def compute_rotation_matrix(theta):
    """Compute homogeneous rotation matrix for a given angle theta (in radians).
    The rotation is around the z-axis (i.e., in the xy-plane).
    Theta is positive going from x to y."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# Compute rotation matrix for every theta value
compute_rotation_matrix_vec = np.vectorize(
    compute_rotation_matrix, signature="()->(3, 3)"
)
rotation_to_ICS0_centre_array = compute_rotation_matrix_vec(
    transforms_df["theta"].cumsum().values,  # we take the cumulative sum of theta
)

print(rotation_to_ICS0_centre_array.shape)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute array of accumulated translation per frame


def compute_translation_matrix(tx, ty):
    """Compute the translation matrix in homog coordinates
    for a given translation vector (tx, ty)."""
    return np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ]
    )


compute_translation_matrix_vec = np.vectorize(
    compute_translation_matrix, signature="(),()->(3, 3)"
)
translation_to_ICS0_centre_array = compute_translation_matrix_vec(
    transforms_df["tx"].cumsum().values, transforms_df["ty"].cumsum().values
)

print(translation_to_ICS0_centre_array.shape)

# %%
translation_norm = np.linalg.norm(
    np.array(
        [transforms_df["tx"].cumsum().values, transforms_df["ty"].cumsum().values]
    ),
    axis=0,
)
fig, ax = plt.subplots()
ax.plot(translation_norm)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Apply full transform to position_array_homogeneous

# Following the matrix multiplication from Right to Left:
# 1- Transform coordinates to ICS_centre
# 3- Apply transform from f to f=0: translation (first) + rotation (second)
# 4- Transform coordinates back to ICS_corner

position_array_ICS0 = (
    np.linalg.inv(Q_corner_to_centre)  # (3, 3)
    @ rotation_to_ICS0_centre_array  # (time, 3, 3)
    @ translation_to_ICS0_centre_array  # (time, 3, 3)
    @ Q_corner_to_centre  # (3, 3)
    @ np.expand_dims(
        np.moveaxis(position_array_homogeneous.values, [0, 1], [2, 3]), axis=-1
    )  # (keypoints, individuals, time, 3, 1)
    # we move the array axes to the end as per numpy.matmul convention
    # https://numpy.org/doc/2.0/reference/generated/numpy.matmul.html --> Notes
)
print(position_array_ICS0.shape)

# %%
# Remove the last expanded axis, returning (keypoints, individuals, time, 3)
position_array_ICS0 = position_array_ICS0.squeeze(axis=-1)
# Undo the reordering dimensions required for broadcasting,
# returning (time, 3, keypoints, individuals)
position_array_ICS0 = np.moveaxis(position_array_ICS0, [0, 1], [-2, -1])
print(position_array_ICS0.shape)


# %%
# format the result as xarray
position_array_ICS0 = xr.DataArray(
    position_array_ICS0,
    dims=["time", "space", "keypoints", "individuals"],
    coords={
        "time": ds["time"],
        "space": ["x", "y", "h"],
        "keypoints": ds["keypoints"],
        "individuals": ds["individuals"],
    },
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot centroid trajectories per individual

# Get the "tab20" colormap
cmap = matplotlib.colormaps["tab20"]
values = np.linspace(0, 1, len(position_array_ICS0.individuals))
colors = cmap(values)

fig, ax = plt.subplots()
for i, ind in enumerate(position_array_ICS0.individuals):
    ax.scatter(
        position_array_ICS0.mean("keypoints").sel(space="x", individuals=ind).values,
        position_array_ICS0.mean("keypoints").sel(space="y", individuals=ind).values,
        s=1,
        c=colors[i, :].reshape(1, -1),
    )
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# %% 
# Plot trajectories per individual but colored by time
fig, ax = plt.subplots()
c = position_array_ICS0.time.values
for i, ind in enumerate(position_array_ICS0.individuals):
    plot_centroid_trajectory(
        position_array_ICS0,
        individual=ind,
        ax=ax,
        c=c,
        s=1,
    )

ax.set_aspect("equal")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot keypoints for one individual
ind = 0  # the index of the individual to plot
fig, ax = plt.subplots()

# Get all keypoint names
keypoints = position_array_ICS0.keypoints.values

for f in range(position_array_ICS0.shape[0]):  # iterate over time
    # Plot each keypoint's position
    for kp in keypoints:
        ax.plot(
            position_array_ICS0.isel(individuals=ind).sel(
                time=f, keypoints=kp).sel(space="x"),
            position_array_ICS0.isel(individuals=ind).sel(
                time=f, keypoints=kp).sel(space="y"),
            "go",
        )

ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute blended image applying MAX of one very n frames
blend_step = 10
list_frames = list(range(position_array_ICS0.values.shape[0]))
list_frames_to_plot = list_frames[0:-1:blend_step]

# Shape of output (blended) image
# output_shape = [int(x * 5) for x in frame_shape[:2]]
# Note: it effectively crops anything above the y=0 axis of the first frame
output_shape = [1400, 5500]
blended_warped_img_max = np.zeros(output_shape + [3])


for f_i, f in enumerate(list_frames_to_plot):
    img_warped = warp(
        video[f],
        np.linalg.inv(
            np.linalg.inv(Q_corner_to_centre)
            @ rotation_to_ICS0_centre_array[list_frames.index(f), :, :]
            @ translation_to_ICS0_centre_array[list_frames.index(f), :, :]
            @ Q_corner_to_centre
        ),
        # we do inverse outside because skimage's warp expects
        # the transform from output image to input image
        output_shape=output_shape,
    )

    # Compute running max pixel
    blended_warped_img_max = np.maximum(blended_warped_img_max, img_warped)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute blended image by computing MEAN of every n frames
blend_step = 500
list_frames = list(range(position_array_ICS0.values.shape[0]))
list_frames_to_plot = list_frames[0:-1:blend_step]

# Shape of output (blended) image
# output_shape = [int(x * 5) for x in frame_shape[:2]]
output_shape = [1400, 5500]
blended_warped_img_mean = np.zeros(output_shape + [3])


for f_i, f in enumerate(list_frames_to_plot):
    img_warped = warp(
        video[f],
        np.linalg.inv(
            np.linalg.inv(Q_corner_to_centre)
            @ rotation_to_ICS0_centre_array[list_frames.index(f), :, :]
            @ translation_to_ICS0_centre_array[list_frames.index(f), :, :]
            @ Q_corner_to_centre
        ),
        # we do inverse outside because skimage's warp expects
        # the transform from output image to input image
        output_shape=output_shape,
    )

    # Compute blend
    blended_warped_img_mean = ski.util.compare_images(
        blended_warped_img_mean, img_warped, method="blend"
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute blended image by OVERLAYING one every n frames,
# taking the EARLIEST pixel value

blend_step = 300
list_frames = list(range(position_array_ICS0.values.shape[0]))
list_frames_to_plot = list_frames[0:-1:blend_step]

# Shape of output (blended) image
# output_shape = [int(x * 5) for x in frame_shape[:2]]
# Note: it effectively crops anything above the y=0 axis of the first frame
output_shape = [1650, int(max(translation_norm) + frame_shape[1])]
blended_warped_img = np.zeros(output_shape + [3])

video_warped = np.zeros([video.shape[0], *output_shape, 3])

# TODO: vectorize this now that we use the full set of frames
for f in list_frames_to_plot:
    img_warped = warp(
        video[f],
        np.linalg.inv(
            np.linalg.inv(Q_corner_to_centre)
            @ rotation_to_ICS0_centre_array[f, :, :]
            @ translation_to_ICS0_centre_array[f, :, :]
            @ Q_corner_to_centre,
        ),
        # we do inverse outside because skimage's warp expects
        # the transform from output image to input image
        output_shape=output_shape,
    )

    # Add to array
    video_warped[f] = img_warped

    # Define a mask that is positive around img_warped
    mask = np.all(
        [img_warped[:, :, i] != 0 for i in range(img_warped.shape[-1])], axis=0
    )

    # Assign value to blended_warped_img only if not already assigned
    blended_warped_img = np.where(
        blended_warped_img == 0, img_warped, blended_warped_img
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Save blended image

# get string timestamp of today in yyyymmdd_hhmmss
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

matplotlib.image.imsave(
    Path("figures") / f"Figure_blended_n{blend_step}_{timestamp}.png",
    blended_warped_img,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot trajectories over blended image
fig, ax = plt.subplots()
ax.imshow(blended_warped_img)
for ind in position_array_ICS0.individuals:
    ax.scatter(
        position_array_ICS0.mean("keypoints").sel(space="x", individuals=ind).values,
        position_array_ICS0.mean("keypoints").sel(space="y", individuals=ind).values,
        s=1,
        cmap="tab20",
    )
ax.set_aspect("equal")

ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export data

# Remove homgeneous coordinate in position_array_ICS0
position_array_ICS0_non_homog = position_array_ICS0.drop_sel(space="h")

ds_export = load_poses.from_numpy(
    position_array=position_array_ICS0_non_homog.values,
    confidence_array=ds.confidence.values,
    individual_names=ds.individuals.values,
    keypoint_names=ds.keypoints.values,
    source_software="manual",
)

ds_export.attrs["source_file"] = ""

# get string timestamp of  today in yyyymmdd_hhmmss
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

export_dir = data_dir / "approach-itk-all"
slp_file = save_poses.to_sleap_analysis_file(
    ds_export,
    export_dir / f"{filename.stem}_unwrapped_{timestamp}.h5",
)

# %%
