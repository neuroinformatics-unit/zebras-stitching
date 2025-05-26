"""
Detect and track trees in a video using DeepForest

"""

# %%
from datetime import datetime
from pathlib import Path

import sleap_io as sio
import torch
from boxmot import BotSort
from deepforest import main
from movement.io import load_bboxes
import csv
# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data
video_path = Path(__file__).parent / "videos" / "21Jan_007.mp4"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters
confidence_threshold = 0.0


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set default device: CUDA if available, otherwise mps, otherwise CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper function to write tracked detections to a csv file
def write_tracked_detections_to_csv(
    csv_file_path: str,
    tracked_bboxes_dict: dict,
    frame_name_regexp: str = "frame_{frame_idx:08d}.png",
    all_frames_size: int = 8888,
):
    """Write tracked detections to a csv file."""
    # Initialise csv file
    csv_file = open(csv_file_path, "w")
    csv_writer = csv.writer(csv_file)

    # write header following VIA convention
    # https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html
    csv_writer.writerow(
        (
            "filename",
            "file_size",
            "file_attributes",
            "region_count",
            "region_id",
            "region_shape_attributes",
            "region_attributes",
        )
    )

    # write detections
    # loop thru frames
    for frame_idx in tracked_bboxes_dict:
        # loop thru all boxes in frame
        for bbox, id, pred_score in zip(
            tracked_bboxes_dict[frame_idx]["tracked_boxes"],
            tracked_bboxes_dict[frame_idx]["ids"],
            tracked_bboxes_dict[frame_idx]["scores"],
            strict=False,
        ):
            # extract shape
            xmin, ymin, xmax, ymax = bbox
            width_box = int(xmax - xmin)
            height_box = int(ymax - ymin)

            # Add to csv
            csv_writer.writerow(
                (
                    frame_name_regexp.format(
                        frame_idx=frame_idx
                    ),  # f"frame_{frame_idx:08d}.png",  # frame index!
                    all_frames_size,  # frame size
                    '{{"clip":{}}}'.format("123"),
                    1,
                    0,
                    f'{{"name":"rect","x":{xmin},"y":{ymin},"width":{width_box},"height":{height_box}}}',
                    f'{{"track":"{int(id)}", "confidence":"{pred_score}"}}',
                )
            )



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialize the deepforest model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

# Place model on device
model.to(device)

print(f"Model device: {model.device}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialize the tracker
tracker = BotSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),  # Path to ReID model
    device="0",  # why not device? why is this in GPU if we then copy to CPU?
    half=False,
    track_high_thresh=confidence_threshold,
    track_low_thresh=confidence_threshold,
    new_track_thresh=confidence_threshold,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read video as array
video_array = sio.load_video(video_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Detect and track trees in the video

# Initialise dict to store tracked bboxes
tracked_detections_all_frames = {}

# Loop over frames
for frame_idx, frame in enumerate(video_array):
    # Place image as tensor on device
    # image_tensor = torch.from_numpy(frame).to(device)[None]

    # Run detection
    df = model.predict_image(image=frame.astype("float32"))

    # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
    df.loc[df["label"] == "Tree", "label_int"] = 1
    detections_array = df.loc[
        :, ["xmin", "ymin", "xmax", "ymax", "score", "label_int"]
    ].to_numpy()

    # Consider only the detections with a confidence greater than the threshold
    detections_array = detections_array[detections_array[:, 4] >= confidence_threshold]

    if detections_array.size == 0:
        continue
    else:
        # Run tracker on detections
        # Update the tracker
        tracked_boxes_array = tracker.update(
            detections_array, frame
        )  # --> M X (x, y, x, y, id, conf, cls, ind)
        # ind is the index of the corresponding detection in the detections_array

        # Add data to dict; key is frame index (0-based) for input clip
        tracked_detections_all_frames[frame_idx] = {
            "tracked_boxes": tracked_boxes_array[:, :4],  # :-1],  # (x, y, x, y)
            "ids": tracked_boxes_array[:, 4],  # -1],  # IDs are the last(5th) column
            "scores": detections_array[:, 4],
        }


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Write tracked detections as VIA-tracks file
# to inspect in napari

# timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_dir = Path(__file__).parent / "data"
filename = data_dir / f"{video_path.stem}_tracked_trees_{timestamp}.csv"

write_tracked_detections_to_csv(
    csv_file_path=filename,
    tracked_bboxes_dict=tracked_detections_all_frames,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read VIA-tracks file as a movement dataset

ds = load_bboxes.from_via_tracks_file(
    file_path=filename, use_frame_numbers_from_file=False
)

# %%
