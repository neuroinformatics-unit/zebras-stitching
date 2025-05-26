"""Extract transforms in keyframes from the SfM output.

Requires launching this notebook in the opendm container.
"""

# %%
from datetime import datetime
from pathlib import Path
from opensfm import dataset 
import csv

from scipy.spatial.transform import Rotation
from utils import get_camera_intrinsic_matrix

import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
opensfm_dir = Path("/workspace/datasets/project/opensfm")
odm_dataset_dir = Path(__file__).parents[1] / "datasets/project"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read reconstruction data
data = dataset.DataSet(opensfm_dir)
recs = data.load_reconstruction()[0]

# Get frame filenames
list_frame_filenames = list(recs.shots.keys())
list_frame_filenames = sorted(list_frame_filenames)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check K matrix in pixel coordinates
cam = recs.shots[list_frame_filenames[0]].camera  # camera in first frame
image_width = cam.width
image_height = cam.height

# Get K matrix in pixel coordinates
K_in_pixel_coords = cam.get_K_in_pixel_coordinates(image_width, image_height)
print(K_in_pixel_coords)

# Compare to K matrix computed "manually"
K_camera_intrinsic = get_camera_intrinsic_matrix(
    odm_dataset_dir / "cameras.json", 
    in_pixel_coords=True
)
print(K_camera_intrinsic)


# Check if they are the same
# The difference of 0.5 pixels in cx,cy
# is due to opensfm missing -1 when transforming K to pixel coordinates
# See https://opensfm.org/docs/geometry.html
# I stick to my implementation with -1 as described in their docs
print(K_in_pixel_coords - K_camera_intrinsic)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract keyframe transforms per frame
list_frame_idx = []
list_R_cam_to_world = []
list_t_cam_to_world = []

for frame_filename in list_frame_filenames:
    # Get frame index
    frame_idx = int(frame_filename.split(".")[0])
    list_frame_idx.append(frame_idx)

    # Get intrinsic and extrinsic camera parameters for this frame
    shot = recs.shots[frame_filename]
    pose = shot.pose  # holds extrinsic parameters

    # Get rotation and translation matrices 
    # they are applied to vectors in normalised coordinates
    list_R_cam_to_world.append(pose.get_R_cam_to_world())
    list_t_cam_to_world.append(pose.get_t_cam_to_world())


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export transforms per keyframe as csv file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_dir = Path(__file__).parent / "data"
csv_path = data_dir / f"sfm_keyframes_transforms_{timestamp}.csv"

# Write csv
with open(csv_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "frame_index", 
            "R_cam_to_world_as_rotvec_x", 
            "R_cam_to_world_as_rotvec_y", 
            "R_cam_to_world_as_rotvec_z", 
            "t_cam_to_world_x", 
            "t_cam_to_world_y", 
            "t_cam_to_world_z"
        ]
    )

    for f, R, t in zip(list_frame_idx, list_R_cam_to_world, list_t_cam_to_world):
        # Convert rotation matrix to quaternion using scipy
        # quat = Rotation.from_matrix(R).as_quat()  # returns [x, y, z, w]
        rotvec = Rotation.from_matrix(R).as_rotvec()
        writer.writerow([f, *rotvec, *t])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot keyframe poses
