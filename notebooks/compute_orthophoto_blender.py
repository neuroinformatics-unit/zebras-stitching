"""
Script to render a custom orthophoto using Blender.

To run this script using Blender's Python interpreter, run the following command:

blender --background --python path/to/compute_orthophoto_blender.py

Alternatively, you can copy and paste the content into Blender's scripting editor
and run it from there.

Loads the mesh .obj file and represents the trajectories of each individual
as separate polylines. Creates a virtual camera and orients its z-axis parallel
to the normal of the mesh's best-fitting plane. It then renders the scene from
the camera position.

Blender version: 4.4.3

"""

from datetime import datetime
import bpy
import csv
from pathlib import Path

import numpy as np
import mathutils

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
data_dir = Path(__file__).parent / "data"
reference_trajectory_file = "20250325_2228_id_sfm_interp_PCS_2d_20250516_155745_clean"
# "20250325_2228_id_sfm_interp_WCS_3d_20250516_155745" for "uncleaned" zebra trajectories
# "20250325_2228_id_sfm_interp_PCS_2d_20250516_155745_clean" for "cleaned" zebra trajectories
# "21Jan_007_tracked_trees_reliable_sleap_sfm_interp_PCS_2d_20250516_160103" for reliable trees
csv_dir = data_dir / f"blender-csv-{reference_trajectory_file}"

# Path to .obj mesh file output from OpenDroneMap
opendronemap_dir = (
    Path(__file__).parents[1]
    / "datasets" #_step20_03_0indexing_masking"
    / "project" #_no_images"
)
mesh_file = opendronemap_dir / "odm_texturing" / "odm_textured_model_geo.obj"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Other input parameters

# offset added to the trajectories for visibility
# (the trajectories are defined in the best-fitting plane)
z_offset = 0.05


plane_normal = np.array([-0.08907009, 0.08261507, -0.9925932])
plane_center = np.array([1.24176788, 0.15539519, -1.00927566])

# the camera is placed at the plane centre plus this offset
camera_position_offset = np.array([0.40, -0.40, 4.75])

# Define colors array
# blender python doesn't have matplotlib so we define the 
# 50 amples from the turbo colormap here.
# To reproduce:
# >>> import matplotlib.pyplot as plt
# >>> cmap = plt.get_cmap("turbo")
# >>> colors_array = cmap(np.linspace(0,1,50))
colors_array = np.array(
    [
       [0.18995, 0.07176, 0.23217, 1.     ],
       [0.21291, 0.12947, 0.37314, 1.     ],
       [0.23236, 0.18603, 0.50004, 1.     ],
       [0.2483 , 0.24143, 0.61286, 1.     ],
       [0.26074, 0.29568, 0.71162, 1.     ],
       [0.27103, 0.35926, 0.81156, 1.     ],
       [0.27576, 0.41097, 0.87936, 1.     ],
       [0.27698, 0.46153, 0.93309, 1.     ],
       [0.27469, 0.51094, 0.97275, 1.     ],
       [0.26252, 0.56967, 0.99773, 1.     ],
       [0.23874, 0.61931, 0.99485, 1.     ],
       [0.20708, 0.66866, 0.97423, 1.     ],
       [0.17223, 0.7168 , 0.93981, 1.     ],
       [0.13886, 0.76279, 0.8955 , 1.     ],
       [0.10738, 0.81381, 0.83484, 1.     ],
       [0.09377, 0.85175, 0.78264, 1.     ],
       [0.09662, 0.88454, 0.73316, 1.     ],
       [0.12014, 0.91193, 0.6866 , 1.     ],
       [0.17377, 0.94053, 0.61938, 1.     ],
       [0.23449, 0.96065, 0.55614, 1.     ],
       [0.30513, 0.97697, 0.48987, 1.     ],
       [0.38127, 0.98909, 0.42386, 1.     ],
       [0.45854, 0.99663, 0.3614 , 1.     ],
       [0.54658, 0.99907, 0.29581, 1.     ],
       [0.61088, 0.99514, 0.2528 , 1.     ],
       [0.66428, 0.98524, 0.2237 , 1.     ],
       [0.71577, 0.96875, 0.20815, 1.     ],
       [0.77591, 0.94113, 0.2031 , 1.     ],
       [0.82333, 0.91253, 0.20663, 1.     ],
       [0.86709, 0.87968, 0.21391, 1.     ],
       [0.90605, 0.84337, 0.22188, 1.     ],
       [0.93909, 0.80439, 0.22744, 1.     ],
       [0.96931, 0.75519, 0.22663, 1.     ],
       [0.98549, 0.7125 , 0.2165 , 1.     ],
       [0.99438, 0.66386, 0.19971, 1.     ],
       [0.99672, 0.60977, 0.17842, 1.     ],
       [0.99153, 0.54036, 0.1491 , 1.     ],
       [0.98108, 0.48104, 0.12332, 1.     ],
       [0.96555, 0.42241, 0.09798, 1.     ],
       [0.94538, 0.36638, 0.07461, 1.     ],
       [0.92105, 0.31489, 0.05475, 1.     ],
       [0.88691, 0.26152, 0.03753, 1.     ],
       [0.8538 , 0.2217 , 0.02677, 1.     ],
       [0.81608, 0.18462, 0.01809, 1.     ],
       [0.77377, 0.15028, 0.01148, 1.     ],
       [0.71692, 0.11268, 0.00629, 1.     ],
       [0.66449, 0.08436, 0.00424, 1.     ],
       [0.60746, 0.05878, 0.00427, 1.     ],
       [0.54583, 0.03593, 0.00638, 1.     ],
       [0.4796 , 0.01583, 0.01055, 1.     ]
    ]
)


def main():
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Clear pre-existing objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load mesh file
    bpy.ops.wm.obj_import(
        filepath=str(mesh_file),
        directory=str(Path(mesh_file).parent),
        files=[
            {"name": "odm_textured_model_geo.obj", "name": "odm_textured_model_geo.obj"}
        ],
        forward_axis="Y",
        up_axis="Z",
    )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Add sun light
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 0))

    sun_light = bpy.data.lights["Sun"]
    sun_light.energy = 1.0
    sun_light.use_shadow = False

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Get all CSV files in the directory
    list_csv_filepaths = list(Path(csv_dir).glob("*.csv"))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create a curve for each CSV file
    for csv_i, csv_filepath in enumerate(list_csv_filepaths):
        # Read coordinates from the CSV file
        with open(csv_filepath, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            points = [(float(x), float(y), float(z) + z_offset) for x, y, z in reader]

        # Create a new curve data object
        curve_data = bpy.data.curves.new(
            name=f"polyline-{csv_filepath.stem}", type="CURVE"
        )
        curve_data.dimensions = "3D"
        curve_data.resolution_u = 2

        # Create a new curve object
        curve_object = bpy.data.objects.new(
            f"polylineObject-{csv_filepath.stem}", curve_data
        )
        bpy.context.collection.objects.link(curve_object)

        # Create a polyline from the points
        polyline = curve_data.splines.new(type="POLY")
        polyline.points.add(len(points) - 1)  # Add points to the polyline
        for p_i, (x, y, z) in enumerate(points):
            polyline.points[p_i].co = (
                x,
                y,
                z,
                1,
            )  # The last value is the weight of the data point

        # Create a new material
        material = bpy.data.materials.new(name=f"material-{csv_filepath.stem}")
        material.use_nodes = True  # Enable nodes for the material

        # Set the base color of the material
        principled_bsdf = material.node_tree.nodes.get("Principled BSDF")
        principled_bsdf.inputs["Base Color"].default_value = tuple(
            colors_array[csv_i, :]
        )

        # Assign the material to the curve object
        if curve_object.data.materials:
            curve_object.data.materials[0] = material
        else:
            curve_object.data.materials.append(material)

        # Set thickness
        curve_object.data.bevel_depth = 0.0015

        # Disable shadow casting for the curves
        curve_object.display.show_shadows = False

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Add a camera at a specific position
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("CameraObject", camera_data)
    bpy.context.collection.objects.link(camera_object)

    # Set the camera position
    camera_object.location = tuple(plane_center + camera_position_offset)

    # Create a rotation that maps the camera's local z-axis to the negative plane normal
    target_vector = mathutils.Vector(-plane_normal).normalized()
    rot_quat = target_vector.to_track_quat("Z", "X")  # Align Z to vector, keep X as up

    # Create a quaternion representing a 10° rotation around local Z
    z_rot_quat = mathutils.Quaternion((0, 0, 1), np.deg2rad(55))

    # Apply the rotations
    # To apply the rotation in z in the camera's local space, we need to apply the
    # rotation to the "world" coordinate system first, and then apply the camera pose to that.
    camera_object.rotation_mode = "QUATERNION"
    camera_object.rotation_quaternion = rot_quat @ z_rot_quat

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Render view from the camera with default camera settings
    # Set the camera as the active camera
    bpy.context.scene.camera = camera_object

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(
        data_dir / f"orthophoto_{reference_trajectory_file}_{timestamp}.png"
    )

    # Set the resolution for orthophoto
    bpy.context.scene.render.resolution_x = 1920  # width
    bpy.context.scene.render.resolution_y = 1080  # height
    bpy.context.scene.render.resolution_percentage = 100  # (100% for full resolution)
    # transparent background
    bpy.context.scene.render.film_transparent = True

    # Render the scene
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
