import json
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import itk
import numpy as np
import numpy.typing as npt
import xarray as xr


def run_registration(
    moving_image: npt.NDArray,
    fixed_image: npt.NDArray,
    registration_parameter_path: Path,
    moving_mask: Optional[npt.NDArray] = None,
    fixed_mask: Optional[npt.NDArray] = None,
) -> itk.ParameterObject:
    """
    Run the ITK Elastix registration process on the given images.

    Parameters
    ----------
    moving_image : npt.NDArray
        The fixed image.
    fixed_image : npt.NDArray
        The moving image.
    registration_parameter_path : Path
        The path to the registration parameter file.
    moving_mask : Optional[npt.NDArray], optional
        The atlas mask, by default None. If provided, the mask will be used
        to exclude regions from the registration.
    fixed_mask : Optional[npt.NDArray], optional
        The moving mask, by default None. If provided, the mask will be used
        to exclude regions from the registration.

    Returns
    -------
    itk.ParameterObject
        The result transform parameters.
    """
    # convert to ITK, view only
    moving_image = itk.GetImageViewFromArray(moving_image)
    fixed_image = itk.GetImageViewFromArray(fixed_image)

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)

    if moving_mask is not None:
        moving_mask = itk.GetImageViewFromArray(moving_mask)
        elastix_object.SetMovingMask(moving_mask)

    if fixed_mask is not None:
        fixed_mask = itk.GetImageViewFromArray(fixed_mask)
        elastix_object.SetFixedMask(fixed_mask)

    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(str(registration_parameter_path))

    elastix_object.SetParameterObject(parameter_object)

    # Run the registration
    elastix_object.UpdateLargestPossibleRegion()

    result_transform_parameters = elastix_object.GetTransformParameterObject()

    return result_transform_parameters


def load_video_to_numpy(video_path, frame_size=(1080, 1920)):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = np.zeros((num_frames, *frame_size, 3), dtype=np.uint8)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames[frame_index] = frame
        frame_index += 1

    cap.release()

    return frames


def position_array_to_homogeneous(position_array: np.ndarray) -> np.ndarray:
    """
    Convert a position array to a homogeneous coordinate array.

    (x, y, 1) instead of (x, y)
    I use "h" for the third homog coord instead of "z" for clarity
    """
    return xr.concat(
        [
            position_array,
            xr.full_like(position_array.sel(space="x"), 1).expand_dims(space=["h"]),
        ],
        dim="space",
    )


def compute_plane_normal_and_center(mesh):
    # Fit a plane to the mesh vertices
    vertices = mesh.vertices
    center = vertices.mean(axis=0)  # a point on the plane

    # Get the covariance matrix
    cov = np.cov(vertices.T)
    # Get the eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal = eigenvecs[:, 0]

    return normal, center


def compute_H_norm_to_pixel_coords(w, h):
    """
    Convert normalized coordinates to pixel coordinates

    https://opensfm.org/docs/geometry.html
    """
    s = max(w, h)
    return np.array([[s, 0, 0.5 * (w - 1)], [0, s, 0.5 * (h - 1)], [0, 0, 1]])


def ray_plane_intersection(ray_origins, ray_directions_unit, plane_normal, plane_point):
    """
    Compute the intersection points of an array of rays with a plane.

    Parameters:
    ray_origins: (N, 3) array of ray origins
    ray_directions: (N, 3) array of ray directions (should be unit vectors)
    plane_normal: (3,) normal vector to the plane (should be unit vector)
    plane_point: (3,) point contained in the plane

    Returns:
    intersections: (N, 3) array of intersection points (NaN where no intersection)

    The mathematical solution uses the parametric form of the ray equation and
    the point-normal form of the plane equation to find the intersection point.
    The intersection point p fullfils the following two equations:
    - The ray equation:
        p = ray_origin + t * ray_direction
    - The plane equation:
        (p - plane_point) · plane_normal = 0
    We can combine these two equations to solve for the parameter t,
    which we then use to find the intersection point p.

    The function also includes checks for:
    - Rays parallel to the plane (denominator close to zero)
    - Intersections behind the camera (t ≤ 0)
    - Invalid rays (NaN in input)
    """
    # Initialize array for intersections
    intersections = np.full_like(ray_origins, np.nan)

    # Compute projection of the ray vector onto the plane normal
    ray_projection = np.dot(ray_directions_unit, plane_normal)

    # Find rays that intersect the plane (projection != 0)
    valid_rays = np.abs(ray_projection) > 1e-10

    if np.any(valid_rays):
        # Compute t parameter for valid rays
        t = (
            np.dot(plane_point - ray_origins[valid_rays], plane_normal)
            / ray_projection[valid_rays]
        )

        # Only keep intersections in front of the camera (t > 0)
        valid_t = t > 0
        valid_rays[valid_rays] = valid_t

        # Compute intersection points
        intersections[valid_rays] = (
            ray_origins[valid_rays]
            + t[valid_t, np.newaxis] * ray_directions_unit[valid_rays]
        )

    return intersections


def get_camera_intrinsic_matrix(
    input_json: Union[str, Path],
    in_pixel_coords: bool = False,
    camera_id: str = "  1920 1080 brown 0.85",
) -> np.ndarray:
    """
    Read the cameras.json file and return the camera intrinsic matrix K.

    Parameters:
    -----------
    input_json : str | Path
        The path to the cameras.json file
    in_pixel_coords : bool
        Whether the camera intrinsic matrix is in pixel coordinates.
        Default is False (normalised coordinates).
    camera_id : str
        The ID of the camera in the cameras.json file. Default is "  1920 1080 brown 0.85"

    Returns:
    --------
    K : numpy.ndarray
        The 3x3 camera intrinsic matrix

    Notes
    -----
    - The camera intrinsic matrix is in pixel coordinates if in_pixel_coords is True.
    - The camera intrinsic matrix is in normalised coordinates if in_pixel_coords is False.
    - See https://opensfm.org/docs/geometry.html for more information.
    """
    # Read the cameras.json file
    with open(input_json, "r") as f:
        cameras = json.load(f)

    # Get the camera parameters
    camera = cameras[camera_id]
    fx = camera["focal_x"]
    fy = camera["focal_y"]
    cx = camera["c_x"]
    cy = camera["c_y"]

    if in_pixel_coords:
        scale = np.max([camera["width"], camera["height"]])
        # Convert parameters to pixel coordinates
        # same as cx = H_norm_to_pixel_coords @ cx
        cx = (cx * scale) + 0.5 * (camera["width"] - 1)
        cy = (cy * scale) + 0.5 * (camera["height"] - 1)
        fx = fx * scale
        fy = fy * scale

    # Construct the intrinsic matrix K
    # Following the notation from https://ksimek.github.io/2013/08/13/intrinsic/
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K


def get_camera_distortion_coeffs(
    input_json: Union[str, Path],
    camera_id: str = "  1920 1080 brown 0.85",
) -> Dict[str, float]:
    """
    Read the cameras.json file and return the Brown-Conrady distortion coefficients.

    Parameters:
    -----------
    input_json : str | Path
        The path to the cameras.json file
    camera_id : str
        The ID of the camera in the cameras.json file. Default is "  1920 1080 brown 0.85"

    Returns:
    --------
    distortion : Dict[str, float]
        Dictionary containing the Brown-Conrady distortion coefficients:
        - k1, k2, k3: Radial distortion coefficients
        - p1, p2: Tangential distortion coefficients

    Examples
    --------
    # Use opencv to correct a single point

    # Get the camera intrinsic matrix
    K = get_camera_intrinsic_matrix("cameras.json")

    # Get the camera distortion coefficients
    distortion = get_camera_distortion_coeffs("cameras.json")

    # OpenCV expects coefficients in order: k1, k2, p1, p2, k3
    dist_coeffs = np.array([
        distortion["k1"],
        distortion["k2"],
        distortion["p1"],
        distortion["p2"],
        distortion["k3"]
    ])

    # Point in pixel coordinates
    point = np.array([[100, 200]], dtype=np.float32)

    # Correct the point
    corrected_point = cv2.undistortPoints(
        point,
        K,
        dist_coeffs,
        P=K  # Use same camera matrix for output
    )

    # Example: Correct an entire image
    distorted_image = cv2.imread("distorted.jpg")
    corrected_image = cv2.undistort(
        distorted_image,
        K,
        dist_coeffs
    )
    """
    # Read the cameras.json file
    with open(input_json, "r") as f:
        cameras = json.load(f)

    # Get the camera parameters
    camera = cameras[camera_id]

    # Extract Brown-Conrady distortion coefficients
    distortion = {
        "k1": camera["k1"],  # Radial distortion coefficient
        "k2": camera["k2"],  # Radial distortion coefficient
        "k3": camera["k3"],  # Radial distortion coefficient
        "p1": camera["p1"],  # Tangential distortion coefficient
        "p2": camera["p2"],  # Tangential distortion coefficient
    }

    return distortion


def get_orthophoto_corners_in_3d(
    orthophoto_corners_path: Union[str, Path],
    plane_normal: np.ndarray,
    plane_center: np.ndarray,
) -> np.ndarray:
    """
    Get the corners of the orthophoto in 3D.
    """
    # Read 2d coordinates
    orthophoto_corners_2d = np.loadtxt(orthophoto_corners_path).reshape(-1, 2)

    # add missing corners
    orthophoto_corners_2d = np.vstack(
        [
            orthophoto_corners_2d,
            np.diag(orthophoto_corners_2d),
            np.array(orthophoto_corners_2d[[1, 0], [0, 1]]),
        ]
    )

    # compute projection of orthophoto corners onto the best fitting plane
    # to the mesh
    orthophoto_corners_3d = np.zeros((orthophoto_corners_2d.shape[0], 3))
    orthophoto_corners_3d[:, 0] = orthophoto_corners_2d[:, 0]
    orthophoto_corners_3d[:, 1] = orthophoto_corners_2d[:, 1]
    orthophoto_corners_3d[:, 2] = (
        -np.dot(plane_normal[:2], (orthophoto_corners_2d - plane_center[:2]).T)
        / plane_normal[2]
    ) + plane_center[2]

    return orthophoto_corners_3d


def compute_Q_world2plane(orthophoto_corners_file, plane_normal, plane_center):
    """
    Compute the Q matrix to transform world coordinates to plane coordinates.

    The origin of the plane coordinate system is the xmax, ymin corner of the orthophoto.
    The x-axis is parallel to the vector from (xmax, ymin) to (xmin, ymin)
    The z-axis is parallel to the plane normal.
    The y-axis is the cross product of the z-axis and the x-axis.
    """
    # origin of the plane coordinate system: corner_xmax_ymin
    orthophoto_corners_3d = get_orthophoto_corners_in_3d(
        orthophoto_corners_file, plane_normal, plane_center
    )
    corner_xmax_ymin = orthophoto_corners_3d[-1, :]
    corner_xmin_ymin = orthophoto_corners_3d[0, :]

    # versors
    x_versor = corner_xmin_ymin - corner_xmax_ymin
    x_versor = x_versor / np.linalg.norm(x_versor)

    y_versor = np.cross(plane_normal, x_versor)
    y_versor = y_versor / np.linalg.norm(y_versor)

    z_versor = plane_normal

    # Q matrix:versors as rows
    Q_world2plane = np.vstack([x_versor, y_versor, z_versor])

    return Q_world2plane
