## Notebooks
The notebooks are organized as follows:
- `00_notebook_elastix_transforms.py`: extracts the transforms from the image registration algorithm for approach 1.
- `01_notebook_elastix_unwrap_keypoints.py`: uses the transforms from notebook `00` to unwrap the input 2D trajectories (zebras or trees).
- `02_notebook_clean_unwrapped_tracks.py`: uses `movement` to clean the unwrapped trajectories of zebras.
- `03_notebook_compute_behaviour_metrics.py`: computes collective behaviour metrics of interest from the cleaned unwrapped zebra trajectories.
- `04_notebook_deepforest.py`: uses `deepforest` to detect trees in the image coordinate system, and the BotSort tracking algorithm (from `boxmot`) to track them.
- `05_notebook_reliable_trees.py`: uses  `movement` to select a subset of tree trajectories that have no ID swaps and low jitter, and which span the full area covered by the drone.
- `06_notebook_extract_sfm_keyframes.py`: it uses the output from the SfM algorithm (ran on the data using `opendronemap`) to extract the estimated camera intrinsic matrix, and the camera poses at the selected keyframes (one every 20 frames, plus the first and last frames). This notebook uses the `opensfm` library, which we install via the `opendronemap` container. However, the extracted transforms are exported to a csv file that can be used in subsequent notebooks.
- `07_notebook_interpolate_sfm.py`: it defines the camera poses at every frame by interpolating the camera poses at the keyframes using slerp for rotations and linear interpolation for translations.
- `08_notebook_apply_interpolated_sfm.py`: it applies the interpolated camera poses (either linearly or using `itk-elastix`'s image registration algorithm) to unwrap the trajectories of the zebras or trees.
- `09_notebook_export_3d_trajs_csv.py`: it exports the 3D trajectories of each invidual zebra as a csv file. These files are meant to be imported in Blender for visualisation along the OpenDroneMap generated mesh.
- `10_notebook_compute_tree_error.py`: it computes the distance of each tree trajectory to its centroid in the world coordinate system, and the weighted mean across all trees (weighted by the number of samples).
- `compute_orthophoto_blender.py`: is a script that is meant to be run using Blender's Python interpreter. It loads the OpenDroneMap generated mesh and the 3D trajectories of the animals into Blender, and generates a orthophoto of the scene with the virtual camera's z-axis parallel to the normal of the mesh best-fitting plane.

## OpenDroneMap pipeline
We used OpenDroneMap's command-line interface ([ODM](https://github.com/OpenDroneMap/ODM)) to run the SfM pipeline and generate the mesh. We used the GPU-compatible docker image and ran the pipeline on the 314 keyframes of the video (one every 20 frames, plus the first and last frames). In each of the keyframes, we provided masks around the individual zebras to prevent those pixels from being used in the reconstruction. We computed these masks as padded bounding boxes around each zebra's keypoints.

We use the default setting except for the following custom flags: `--dem-resolution 1.0 --orthophoto-resolution 1.0 --pc-quality high`
