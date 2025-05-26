"""
Interpolate SfM transforms using slerp for rotations
and linear interpolation for translations.
"""

# %%
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
data_dir = Path(__file__).parent / "data"
sfm_keyframe_transforms_file = data_dir / "sfm_keyframes_transforms_20250514_212616.csv"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read the transformscsv file
df_input = pd.read_csv(sfm_keyframe_transforms_file)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Interpolate rotations

keyframe_times = df_input["frame_index"].values
keyframe_rots = R.from_rotvec(
    df_input[
        [
            "R_cam_to_world_as_rotvec_x", 
            "R_cam_to_world_as_rotvec_y", 
            "R_cam_to_world_as_rotvec_z"
        ]
    ].values
)

# interpolator object
slerp = Slerp(keyframe_times, keyframe_rots)

# interpolate at required times
times_to_interpolate = np.arange(keyframe_times[0], keyframe_times[-1]+1, 1)
rots_interpolated = slerp(times_to_interpolate)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Interpolate translations

keyframe_trans = df_input[["t_cam_to_world_x", "t_cam_to_world_y", "t_cam_to_world_z"]].values

trans_interpolated = np.zeros((len(times_to_interpolate), 3))
trans_interpolated[:, 0] = np.interp(times_to_interpolate, keyframe_times, keyframe_trans[:, 0])
trans_interpolated[:, 1] = np.interp(times_to_interpolate, keyframe_times, keyframe_trans[:, 1])
trans_interpolated[:, 2] = np.interp(times_to_interpolate, keyframe_times, keyframe_trans[:, 2])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export interpolated transforms as csv

df_interpolated = pd.DataFrame(
    {
        "frame_index": times_to_interpolate,
        "R_cam_to_world_as_rotvec_x": rots_interpolated.as_rotvec()[:, 0],
        "R_cam_to_world_as_rotvec_y": rots_interpolated.as_rotvec()[:, 1],
        "R_cam_to_world_as_rotvec_z": rots_interpolated.as_rotvec()[:, 2],
        "t_cam_to_world_x": trans_interpolated[:, 0],
        "t_cam_to_world_y": trans_interpolated[:, 1],
        "t_cam_to_world_z": trans_interpolated[:, 2],
    }
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df_interpolated.to_csv(
    data_dir / f"{sfm_keyframe_transforms_file.stem}_interp_{timestamp}.csv", index=False)

# %%
