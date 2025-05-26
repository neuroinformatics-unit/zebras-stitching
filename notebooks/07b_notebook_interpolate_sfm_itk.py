# %%
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
data_dir = Path(__file__).parent / "data"

sfm_keyframe_transforms_file = data_dir / "sfm_keyframes_transforms_20250514_212616.csv"
itk_batched_transforms_dir = data_dir / "elastix" / "batched"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read the transforms keyframes csv file
df_input = pd.read_csv(sfm_keyframe_transforms_file)

# list of batch transform files
list_batch_transform_files = sorted(list(itk_batched_transforms_dir.glob("*.csv")))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Interpolate rotations

# camera pose at keyframes
# (rotation from world to camera)
keyframe_times = df_input["frame_index"].values

rotation_columns = [
    "R_cam_to_world_as_rotvec_x",
    "R_cam_to_world_as_rotvec_y",
    "R_cam_to_world_as_rotvec_z",
]

camera_pose_per_keyframe = {
    kf: R.from_rotvec(
        df_input.loc[df_input["frame_index"] == kf][rotation_columns].values
    )
    for kf in keyframe_times
}

print(len(camera_pose_per_keyframe))

# %%


rots_interpolated = []
for batch_file in list_batch_transform_files:

    # Get keyframe from batch filename
    start_keyframe = int(batch_file.stem.split("_")[-1].split("-")[0])

    # Read batch transform file
    # contains theta rotation around camera z-axis
    # positive theta is camera x-axis to camera y-axis
    df_batch = pd.read_csv(batch_file)
    df_batch.drop(columns=["tx", "ty"], inplace=True)
    # df_batch["frame_index"] = np.arange(
    #     start_keyframe + 1, start_keyframe + len(df_batch) + 1
    # )

    # Get rotation from keyframe to each frame in batch
    # (this rotation is in local axes, camera coordinate system)
    rot_from_keyframe_to_frame_f = R.from_rotvec(
        np.r_[
            [[0, 0, 0]],  # identity rotation for the keyframe
            df_batch["theta"].values.reshape(-1, 1) * np.array([[0, 0, 1]])
        ]
    )  # len =  20

    # Get camera pose at start keyframe
    camera_pose_at_keyframe = camera_pose_per_keyframe[start_keyframe]

    # compute camera pose at each frame in batch
    # (rotation in local coordinates is applied first)
    camera_pose_batch = camera_pose_at_keyframe * rot_from_keyframe_to_frame_f

    # store
    rots_interpolated.append(camera_pose_batch.as_rotvec())

rots_interpolated = np.concatenate(rots_interpolated)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Interpolate translations
# use itk instead? -- its in pixels, would have to transform to ccs, then wcs


keyframe_trans = df_input[
    ["t_cam_to_world_x", "t_cam_to_world_y", "t_cam_to_world_z"]
].values


times_to_interpolate = np.arange(keyframe_times[0], keyframe_times[-1] + 1, 1)

trans_interpolated = np.zeros((len(times_to_interpolate), 3))
trans_interpolated[:, 0] = np.interp(
    times_to_interpolate, keyframe_times, keyframe_trans[:, 0]
)
trans_interpolated[:, 1] = np.interp(
    times_to_interpolate, keyframe_times, keyframe_trans[:, 1]
)
trans_interpolated[:, 2] = np.interp(
    times_to_interpolate, keyframe_times, keyframe_trans[:, 2]
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export interpolated transforms as csv

df_interpolated = pd.DataFrame(
    {
        "frame_index": times_to_interpolate,
        "R_cam_to_world_as_rotvec_x": rots_interpolated[:, 0],  # rotation in rotvec format
        "R_cam_to_world_as_rotvec_y": rots_interpolated[:, 1],  # rotation in rotvec format
        "R_cam_to_world_as_rotvec_z": rots_interpolated[:, 2],  # rotation in rotvec format
        "t_cam_to_world_x": trans_interpolated[:, 0],
        "t_cam_to_world_y": trans_interpolated[:, 1],
        "t_cam_to_world_z": trans_interpolated[:, 2],
    }
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df_interpolated.to_csv(
    data_dir / f"{sfm_keyframe_transforms_file.stem}_ITK_interp_{timestamp}.csv", index=False)


# %%
