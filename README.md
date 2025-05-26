# zebras-stitching
Exploring how to stitch drone imagery and animal trajectories 🦓🦓🦓

## Installation

From the root of the repository:

```bash
conda create -n zebras-env python=3.11 -y
conda activate zebras-env
pip install -r requirements.txt
```


## Overview

This repository explores possible pipelines for "unwrapping" trajectories of moving animals recorded with a camera drone, using open-source freely available software and off-the-shelf models. We use "unwrapping" to refer to the process of transforming the trajectories of the animals from a moving image coordinate system to a world coordinate system that is fixed to the ground. As a proof-of-concept, we focus on its application to a single video of 44 zebras in escape behavior.

The code is not meant as a fully-fledged package, but rather as a collection of notebooks to explore three different approaches and their trade-offs. We hope these notebooks can serve as a starting point for researchers to get familiar with the problem and the existing tools, or for further development of a more robust pipeline.

The input data consists on a set of 44 trajectories of zebras in escape behaviour, tracked in a moving image coordinate system (associated to the camera drone). These data was provided by the researchers as a SLEAP file (`20250325_2228_id.slp`). The three approaches we explore to compute the unwrapped trajectory are:

1. Using an image registration algorithm (via `itk-elastix`) to compute the transform from each frame to the previous frame.
2. Using a Structure-from-Motion (SfM) algorithm (via `opendronemap` and `opensfm`) to compute the camera poses at selected keyframes, and then interpolating the camera poses in the missing frames using slerp for rotations and linear interpolation for translations.
3. Using an SfM algorithm (via `opendronemap` and `opensfm`) to compute the camera poses at selected keyframes, and then interpolating the camera poses in the missing frames using `itk-elastix`'s image registration algorithm for rotations, and linear interpolation for translations.

To compare the performance of these three prototype approaches, we detect and track trees (using `deepforest` and `boxmot`) in the moving image coordinate system, and then unwrap the trajectories following each of the methods described above. Trees are useful since they are distinctive elements of the scene and they should be static relative to the ground. We evaluate the performance of each approach by computing the mean distance of each tree trajectory to its centroid in the world coordinate system.

Once the trajectories of the animals are "unwrapped" (i.e., in the world coordinate system), we demonstrate the applicability of these methods by computing common metrics from collective behaviour. We use  `movement` to clean the trajectories and programmatically correct ID swaps first, and then to compute these metrics of interest. 

We use Blender to generate custom orthophotos of the scene and overlay the generated 3D trajectories of the animals.


## Toolkit
We would like to acknowledge the following cool open-source tools and libraries that we used in this project:

- [itk-elastix](https://github.com/InsightSoftwareConsortium/ITKElastix)
- [elastix](https://elastix.dev/)
- [opendronemap](https://github.com/OpenDroneMap)
- [opensfm](https://github.com/mapillary/OpenSfM)
- [deepforest](https://github.com/weecology/DeepForest)
- [boxmot](https://github.com/mikel-brostrom/boxmot)
- [movement](https://github.com/neuroinformatics-unit/movement)
- [Blender](https://www.blender.org/)
- [SLEAP](https://sleap.ai/)
