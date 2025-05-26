"""
Apply the interpolated SfM transforms to the 2D trajectories and export as sleap file.

To use with movement 0.5.1 (i.e., not inside container)
"""

# %%
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
import trimesh
from movement.io import load_poses, save_poses
from scipy.spatial.transform import Rotation as R
from utils import (
    compute_H_norm_to_pixel_coords,
    compute_plane_normal_and_center,
    position_array_to_homogeneous,
    ray_plane_intersection,
    get_camera_intrinsic_matrix,
    get_orthophoto_corners_in_3d,
    compute_Q_world2plane,
)

import matplotlib.pyplot as plt

# Uncomment for interactive plots
# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

data_dir = Path(__file__).parent / "data"

# Camera poses
sfm_interpolated_file_dict = {
    "sfm-interp": data_dir
    / "sfm_keyframes_transforms_20250514_212616_interp_20250514_223104.csv",
    "sfm-itk-interp": data_dir
    / "sfm_keyframes_transforms_20250514_212616_ITK_interp_20250517_225813.csv",
}

# 2D data dictionary
points_2d_file_dict = {
    "zebras": data_dir / "20250325_2228_id.slp",
    "trees": data_dir / "21Jan_007_tracked_trees_reliable_sleap.h5",
}

# ODM data
# odm_dataset_dir = Path(__file__).parents[1] / "datasets/project"
mesh_path = (
    data_dir / "odm_data" / "odm_25dmesh.ply"
)  # odm_dataset_dir / "odm_meshing/odm_25dmesh.ply"
orthophoto_corners_file = (
    data_dir / "odm_data" / "odm_orthophoto_corners.txt"
)  # odm_dataset_dir / "odm_orthophoto/odm_orthophoto_corners.txt"
camera_intrinsics = (
    data_dir / "odm_data" / "cameras.json"
)  # odm_dataset_dir / "cameras.json"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select poses to use

# select file with interpolated poses
approach_ref_name = (
    "sfm-itk-interp"  # "approach-sfm-interp" or "approach-sfm-itk-interp"
)
sfm_interpolated_file = sfm_interpolated_file_dict[approach_ref_name]
print(f"Using poses from '{approach_ref_name}'")
print(f"File: {sfm_interpolated_file.name}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select data to transform
data_ref_name = "zebras"  # can be either "zebras" or "trees"
points_2d_file = points_2d_file_dict[data_ref_name]
print(f"Using 2d data file: {points_2d_file.name}")
print(f"File: {points_2d_file.name}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read the transforms file
df_input = pd.read_csv(sfm_interpolated_file)

# Read rotations as dict with frame index as key
rots_cam_to_world_interp = {
    frame_idx: R.from_rotvec(rotvec_xyz)
    for frame_idx, rotvec_xyz in zip(
        df_input["frame_index"].values,
        df_input[
            [
                "R_cam_to_world_as_rotvec_x",
                "R_cam_to_world_as_rotvec_y",
                "R_cam_to_world_as_rotvec_z",
            ]
        ].values,
    )
}

# Read translations as dict with frame index as key
t_cam_to_world_interp = {
    frame_idx: t_xyz
    for frame_idx, t_xyz in zip(
        df_input["frame_index"].values,
        df_input[["t_cam_to_world_x", "t_cam_to_world_y", "t_cam_to_world_z"]].values,
    )
}

print(len(rots_cam_to_world_interp))
print(len(t_cam_to_world_interp))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read mesh and fit a plane to it
mesh = trimesh.load(mesh_path)

plane_normal, plane_center = compute_plane_normal_and_center(mesh)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read 2D trajectories

ds = load_poses.from_sleap_file(points_2d_file)

position = ds.position  # as xarray data array, time in frames
position_homogeneous = position_array_to_homogeneous(position)
position_homogeneous_shape = (
    position_homogeneous.shape
)  # time, space (homogeneous), kpts, individuals

print(position_homogeneous)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Camera parameters

# Get camera intrinsic matrix in pixel coordinates
K_camera_intrinsic = get_camera_intrinsic_matrix(
    camera_intrinsics, in_pixel_coords=True
)

# Get camera width and height
with open(camera_intrinsics, "r") as f:
    cameras = json.load(f)
image_width = cameras["  1920 1080 brown 0.85"]["width"]
image_height = cameras["  1920 1080 brown 0.85"]["height"]

# Get H_pixel2norm, conversion matrix from pixel to normalised coordinates
H_pixel2norm = np.linalg.inv(compute_H_norm_to_pixel_coords(image_width, image_height))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply SfM interpolated transforms to 2D points
list_3D_points_per_frame = []
shape_with_space_last = position_homogeneous.sel(time=0).T.shape  # space dim last

for frame_idx in df_input["frame_index"].values:
    # Get 2D (unnormalied) pixel **homogeneous** coordinates of points at this frame
    # (M,3 array, with M = total number of points)
    pt2D_pixels_homogeneous = position_homogeneous.sel(time=frame_idx).values.T.reshape(
        -1, 3
    )  # transpose to get space dim last

    # Compute *normalised* coordinates in camera coordinate system (aka bearings)
    # K is the camera intrinsic matrix, transforms from CCS -> ICS
    # inv(K) transforms from ICS -> CCS
    pt2D_bearings_homogeneous_ccs = (
        np.linalg.inv(K_camera_intrinsic)  # TODO: should be in pixel coords!
        @ pt2D_pixels_homogeneous.T
    ).T  # returns normalised coordinates!

    # Compute depth from intersection of bearings with plane
    # 1- compute free bearing vectors in world coordinates
    bearings_rotated_to_wcs = (
        rots_cam_to_world_interp[frame_idx].as_matrix()
        @ pt2D_bearings_homogeneous_ccs.T
    ).T
    bearings_rotated_to_wcs = bearings_rotated_to_wcs / np.linalg.norm(
        bearings_rotated_to_wcs, axis=1, keepdims=True
    )

    # 2- compute intersection with plane
    # exclude nans
    pt3D_world_w_nans = np.nan * np.ones_like(pt2D_pixels_homogeneous)
    slc_nan_bearings = np.isnan(bearings_rotated_to_wcs).any(axis=1)

    pt3D_world_normalized = ray_plane_intersection(
        ray_origins=np.tile(
            t_cam_to_world_interp[frame_idx], (sum(~slc_nan_bearings), 1)
        ),
        ray_directions_unit=bearings_rotated_to_wcs[~slc_nan_bearings, :],
        plane_normal=plane_normal,
        plane_point=plane_center,
    )

    # Reshape to original shape
    pt3D_world_w_nans[~slc_nan_bearings, :] = pt3D_world_normalized
    pt3D_world_w_nans = pt3D_world_w_nans.reshape(shape_with_space_last).T

    # append to list per frame
    list_3D_points_per_frame.append(pt3D_world_w_nans)

# Concatenate all 3D points
pt3D_world_all = np.stack(list_3D_points_per_frame, axis=0)

print(pt3D_world_all.shape)  # frame, space, kpts, individuals


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute change of basis to plane basis

# The plane coordinate system:
# origin at corner_xmax_ymin
# x-axis parallel to vector from corner_xmax_ymin to corner_xmin_ymin
# z-axis parallel to plane normal
Q_world2plane = compute_Q_world2plane(
    orthophoto_corners_file, plane_normal, plane_center
)

# Check: express orthophoto corners in the plane basis
# z-coord should be 0 and last point should be (0,0)
orthophoto_corners_3d = get_orthophoto_corners_in_3d(
    orthophoto_corners_file, plane_normal, plane_center
)
corner_xmax_ymin = orthophoto_corners_3d[-1, :]
orthophoto_corners_3d_plane = (
    Q_world2plane @ (orthophoto_corners_3d - corner_xmax_ymin).T
).T
print(orthophoto_corners_3d_plane)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Transform 3D points to plane basis
pt3D_plane_all = (
    Q_world2plane  # (3,3)
    @ (np.moveaxis(pt3D_world_all, 1, -1) - corner_xmax_ymin)[
        ..., None
    ]  # (6293, 2, 44, 3, 1)
    # we move the array axes to the end as per numpy.matmul convention
    # https://numpy.org/doc/2.0/reference/generated/numpy.matmul.html --> Notes
).squeeze(-1)

# Reorder axes to (time, space, kpts, individuals)
pt3D_plane_all = np.moveaxis(pt3D_plane_all, -1, 1)

print(np.nanmax(pt3D_plane_all[:, 2, :, :]))  # should be almost 0
print(np.nanmin(pt3D_plane_all[:, 2, :, :]))  # should be almost 0


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save 3D points in WCS - (original arbitrary units)

# get string timestamp of  today in yyyymmdd_hhmmss
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# These arbitrary units should match the mesh units
# Note that we don't apply any scaling factor here
# since we don't expect to visualize these in napari
ds_3d_wcs = load_poses.from_numpy(
    position_array=pt3D_world_all,
    confidence_array=ds.confidence.values,
    individual_names=ds.individuals.values,
    keypoint_names=ds.keypoints.values,
    fps=None,
    source_software="sfm-interpolated-wcs-3d",
)
ds_3d_wcs.attrs["source_file"] = ""
ds_3d_wcs.attrs["units"] = "pixels"

slp_file = save_poses.to_sleap_analysis_file(
    ds_3d_wcs,
    (
        data_dir
        / f"approach-{approach_ref_name}"
        / f"{points_2d_file.stem}_{approach_ref_name.replace('-', '_')}_WCS_3d_{timestamp}.h5"
    ),
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save 2D points in (scaled) plane basis as movement dataset
# 2D points should be visualizable in napari

# Apply scaling factor before saving
# Note: the world coordinates from sfm are in arbitrary units, since the
# input data is not georeferenced. We scale by the max of the image
# dimensions for easier visualization in napari, but the units continue
# to be arbitrary and have no physical meaning. However note that the relative
# positions of the points are correct.
pt3D_plane_all_scaled = pt3D_plane_all * max(image_width, image_height)

ds_2d_plane = load_poses.from_numpy(
    position_array=pt3D_plane_all_scaled[:, :2, :, :],  # remove z-coordinates
    confidence_array=ds.confidence.values,
    individual_names=ds.individuals.values,
    keypoint_names=ds.keypoints.values,
    fps=None,
    source_software="sfm-interpolated-pcs-2d",
)
ds_2d_plane.attrs["source_file"] = ""
ds_2d_plane.attrs["units"] = "pixels"

slp_file = save_poses.to_sleap_analysis_file(
    ds_2d_plane,
    data_dir
    / f"approach-{approach_ref_name}"
    / f"{points_2d_file.stem}_{approach_ref_name.replace('-', '_')}_PCS_2d_{timestamp}.h5",
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save 2D points in (scaled) plane coordinates projected to z=0 as movement dataset
# 2D points should be visualizable in napari

# Apply scaling factor before saving
# Note: the world coordinates from sfm are in arbitrary units, since the
# input data is not georeferenced. We scale by the max of the image
# dimensions for easier visualization in napari, but the units continue
# to be arbitrary and have no physical meaning. However note that the relative
# positions of the points are correct.
pt3D_world_all_scaled = pt3D_world_all * max(image_width, image_height)

ds_2d_z0 = load_poses.from_numpy(
    position_array=pt3D_world_all_scaled[:, :2, :, :],  # remove z-coordinates
    confidence_array=ds.confidence.values,
    individual_names=ds.individuals.values,
    keypoint_names=ds.keypoints.values,
    fps=None,
    source_software="sfm-interpolated-wcs-2d-z0",
)
ds_2d_z0.attrs["source_file"] = ""
ds_2d_z0.attrs["units"] = "pixels"

slp_file = save_poses.to_sleap_analysis_file(
    ds_2d_z0,
    data_dir
    / f"approach-{approach_ref_name}"
    / f"{points_2d_file.stem}_{approach_ref_name.replace('-', '_')}_WCS_2d_z0_{timestamp}.h5",
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot 3D points

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
color_per_frame = plt.cm.viridis(np.linspace(0, 1, pt3D_world_all.shape[0]))
color_per_individual = plt.cm.viridis(np.linspace(0, 1, pt3D_world_all.shape[1]))
for individual_idx in range(pt3D_world_all.shape[-1]):
    for frame_idx in range(pt3D_world_all.shape[0]):
        # line
        ax.plot(
            pt3D_world_all[frame_idx, 0, :, individual_idx],
            pt3D_world_all[frame_idx, 1, :, individual_idx],
            pt3D_world_all[frame_idx, 2, :, individual_idx],
            "-",
            label=f"frame {frame_idx}",
            c=color_per_frame[frame_idx],
        )
        # # head
        # ax.scatter(
        #     pt3D_world_all[frame_idx, 0, 0, individual_idx],
        #     pt3D_world_all[frame_idx, 1, 0, individual_idx],
        #     pt3D_world_all[frame_idx, 2, 0, individual_idx],
        #     'o',
        #     c=color_per_individual[individual_idx]
        # )
        # # tail
        # ax.scatter(
        #     pt3D_world_all[frame_idx, 0, 1, individual_idx],
        #     pt3D_world_all[frame_idx, 1, 1, individual_idx],
        #     pt3D_world_all[frame_idx, 2, 1, individual_idx],
        #     '*',
        #     c=color_per_individual[individual_idx]
        # )

# # plot orthophoto corners -- ATT: these are in normalised coordinates!
# ax.scatter(
#     orthophoto_corners_3d[:, 0],
#     orthophoto_corners_3d[:, 1],
#     orthophoto_corners_3d[:, 2],
#     'o',
#     c='red'
# )
# plt.legend()
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# %%
