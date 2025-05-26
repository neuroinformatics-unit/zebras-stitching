# %%
from pathlib import Path

import cv2
import numpy as np
import re
import pandas as pd
import sleap_io as sio
import tifffile

from movement.io import load_poses
from utils import load_video_to_numpy, run_registration
from tqdm import tqdm

# %%

# Load video into a numpy array with shape (num_frames, height, width)
# The video is converted to grayscale and stored as uint8.

video_path = "21Jan_007.mp4"

# Lazy load video data using sio
video_data = sio.load_video(video_path, plugin="opencv")

# Non-lazy load video data using cv2
# video_data = load_video_to_numpy(video_path, frame_size=(1080, 1920))

# %%
# Compute masks
ds = load_poses.from_sleap_file("data/20250325_2228_id.slp", fps=30)

position_numpy = ds.position.to_numpy()
num_animals = position_numpy.shape[-1]

max_x = np.max(position_numpy[:, 0, :, :], axis=1).round().astype(int)
max_y = np.max(position_numpy[:, 1, :, :], axis=1).round().astype(int)
min_x = np.min(position_numpy[:, 0, :, :], axis=1).round().astype(int)
min_y = np.min(position_numpy[:, 1, :, :], axis=1).round().astype(int)
min_y.shape, max_y.shape, min_x.shape, max_x.shape

#%%
def register_batch(start_index, output_file, skip = 20, buffer = 10):
    moving_index = start_index

    with open(output_file, "w") as f:
        f.write("theta,tx,ty\n")

    for i in range(moving_index + 1, moving_index + skip):
        # Track to avoid mixing up the frames
        fixed_index = i

        fixed = video_data[fixed_index]
        moving = video_data[moving_index]

        # Convert to grayscale
        fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
        moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
        fixed_mask = np.ones_like(fixed_gray, dtype=np.uint8)
        moving_mask = np.ones_like(moving_gray, dtype=np.uint8)

        for j in range(num_animals):
            # After casting to int, nan values become int64 min value
            if (
                    np.any(min_y[fixed_index, j] < 0)
                    or np.any(min_x[fixed_index, j] < 0)
                    or np.any(max_y[fixed_index, j] < 0)
                    or np.any(max_x[fixed_index, j] < 0)
            ):
                continue

            if (
                    np.any(min_y[moving_index, j] < 0)
                    or np.any(min_x[moving_index, j] < 0)
                    or np.any(max_y[moving_index, j] < 0)
                    or np.any(max_x[moving_index, j] < 0)
            ):
                continue

            # Mask out the area around the animal
            fixed_mask[
            min_y[fixed_index, j] - buffer: max_y[fixed_index, j] + buffer,
            min_x[fixed_index, j] - buffer: max_x[fixed_index, j] + buffer,
            ] = 0
            moving_mask[
            min_y[moving_index, j] - buffer: max_y[moving_index, j] + buffer,
            min_x[moving_index, j] - buffer: max_x[moving_index, j] + buffer,
            ] = 0

        parameters = run_registration(
            moving_gray,
            fixed_gray,
            param_path,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
        )

        # Regular expression to find the TransformParameters line
        pattern = r"\(TransformParameters ([\d\.\-e ]+)\)"
        input_string = str(parameters)
        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        if match:
            # Extract the numbers and convert them to floats
            transform_parameters = list(map(float, match.group(1).split()))
            with open(output_file, "a") as f:
                f.write(",".join(map(str, transform_parameters)))
                f.write("\n")
        else:
            print("TransformParameters not found")

# %%
param_path = Path("./data/elastix/registration_params.txt")
output_file_stub = "data/elastix/out_euler_{}.csv"
frame_skip = 20
final_frame = video_data.shape[0]

for k in tqdm(range(0, final_frame, frame_skip)):
    frame_skip = min(frame_skip, final_frame - k)
    moving_index = k
    curr_output = output_file_stub.format(f"{k:04}-{k+frame_skip-1:04}")
    register_batch(moving_index, curr_output, skip=frame_skip)
