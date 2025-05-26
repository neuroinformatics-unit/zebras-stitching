"""
Notebook to export 3D trajectories as CSV files for each individual.

We use these CSV files to import the trajectories into Blender.
"""

# %%
from movement.io import load_poses
from pathlib import Path
import xarray as xr
import numpy as np
from utils import compute_Q_world2plane, get_orthophoto_corners_in_3d

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

data_dir = Path("data")

# "uncleaned" 3D trajectories zebras
# zebras_input_file = data_dir / "approach-sfm-interp" / "20250325_2228_id_sfm_interp_WCS_3d_20250516_155745.h5"

# or "cleaned" 2D trajectories zebras or reliable trees
zebras_input_file = (
    data_dir
    / "approach-sfm-interp"
    # / "21Jan_007_tracked_trees_reliable_sleap_sfm_interp_PCS_2d_20250516_160103.h5"  # trees
    / "20250325_2228_id_sfm_interp_PCS_2d_20250516_155745_clean.h5" # zebras
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create output directory
blender_csv_dir = data_dir / f"blender-csv-{zebras_input_file.stem}"
blender_csv_dir.mkdir(exist_ok=False)  # do not overwrite existing files


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get additional data required for PCS to WCS transformation
# The plane coordinate system:
# origin at corner_xmax_ymin
# x-axis parallel to vector from corner_xmax_ymin to corner_xmin_ymin
# z-axis parallel to plane normal

plane_normal = np.array([-0.08907009, 0.08261507, -0.9925932])
plane_center = np.array([1.24176788, 0.15539519, -1.00927566])

image_width = 1920
image_height = 1080
scale_factor = max(image_width, image_height)

orthophoto_corners_file = data_dir / "odm_data" / "odm_orthophoto_corners.txt"
orthophoto_corners_3d = get_orthophoto_corners_in_3d(
    orthophoto_corners_file, plane_normal, plane_center
)
corner_xmax_ymin = orthophoto_corners_3d[-1, :]  # origin of PCS


Q_world2plane = compute_Q_world2plane(
    orthophoto_corners_file, plane_normal, plane_center
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read 3D data as a movement dataset
ds = load_poses.from_sleap_file(zebras_input_file)


# if input file is 2D in PCS, add z-coordinate with z=0 and transform to WCS

if ds.position.shape[1] == 2:
    # remove scaling factor applied to 2D data
    position_array = ds.position
    position_array = position_array / scale_factor

    # compute new position array in PCS
    new_position_array_PCS = xr.concat(
        [
            position_array,
            xr.full_like(
                position_array.sel(space="x"),
                fill_value=0,  # add z-coordinate with z=0
            ).expand_dims(space=["z"]),
        ],
        dim="space",
    ).values

    # transform to WCS
    new_position_array_WCS = (
        np.linalg.inv(Q_world2plane)  # (3,3)
        @ (np.moveaxis(new_position_array_PCS, 1, -1))[..., None]  # (6293, 2, 44, 3, 1)
        # we move the array axes to the end as per numpy.matmul convention
        # https://numpy.org/doc/2.0/reference/generated/numpy.matmul.html --> Notes
    ).squeeze(-1)

    # Reorder axes to (time, space, kpts, individuals)
    new_position_array_WCS = np.moveaxis(new_position_array_WCS, -1, 1)

    # add translation to origin of PCS?
    new_position_array_WCS += corner_xmax_ymin[None, :, None, None]

    # redefined dataset with new position array
    ds = xr.Dataset(
        data_vars=dict(
            position=(
                ["time", "space", "keypoints", "individuals"],
                new_position_array_WCS,
            ),
            confidence=ds.confidence,
        ),
        coords={
            "time": ds.time,
            "space": ["x", "y", "z"],
            "keypoints": ds.keypoints,
            "individuals": ds.individuals,
        },
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export each individual's 3D trajectory as a CSV file

centroid_da = ds.position.mean(dim="keypoints")
for individual in ds.individuals.data:
    filepath = blender_csv_dir / f"{individual}_3d_traj.csv"

    centroid_individual = centroid_da.sel(
        individuals=individual
    ).to_pandas()  # for 2D data

    # remove rows with nans before exporting
    centroid_individual = centroid_individual.dropna()

    # write to csv
    centroid_individual.to_csv(filepath, index=False)


# %%
