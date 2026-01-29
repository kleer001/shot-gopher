"""Matrix and transform utilities.

Provides reusable components for:
- Matrix decomposition (translation, rotation, scale)
- Quaternion operations (to/from rotation matrix, SLERP)
- Euler angle conversion with multiple rotation orders
- Coordinate system conversions (OpenCV <-> OpenGL)
- Interpolation (linear, Bezier, SLERP)
"""

from typing import Optional, Union
import numpy as np


def decompose_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose 4x4 transformation matrix into translation, rotation, scale.

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        Tuple of (translation vec3, rotation matrix 3x3, scale vec3)
    """
    translation = matrix[:3, 3].copy()

    m = matrix[:3, :3]
    scale = np.array([
        np.linalg.norm(m[:, 0]),
        np.linalg.norm(m[:, 1]),
        np.linalg.norm(m[:, 2])
    ])

    rotation = m.copy()
    for i in range(3):
        if scale[i] != 0:
            rotation[:, i] /= scale[i]

    return translation, rotation, scale


def compose_matrix(
    translation: np.ndarray,
    rotation: np.ndarray,
    scale: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compose 4x4 transformation matrix from components.

    Args:
        translation: 3D translation vector
        rotation: 3x3 rotation matrix
        scale: Optional 3D scale vector (default: [1, 1, 1])

    Returns:
        4x4 transformation matrix
    """
    matrix = np.eye(4)
    matrix[:3, 3] = translation

    if scale is not None:
        scaled_rotation = rotation.copy()
        for i in range(3):
            scaled_rotation[:, i] *= scale[i]
        matrix[:3, :3] = scaled_rotation
    else:
        matrix[:3, :3] = rotation

    return matrix


def quaternion_to_rotation_matrix(quat: Union[list, np.ndarray]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Args:
        quat: Quaternion as [w, x, y, z] (scalar-first convention)

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z].

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as numpy array [w, x, y, z]
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotation_matrix_to_euler(
    rotation: np.ndarray,
    order: str = "xyz"
) -> np.ndarray:
    """Convert 3x3 rotation matrix to Euler angles.

    Args:
        rotation: 3x3 rotation matrix
        order: Euler rotation order - "xyz" (Maya), "zxy" (Nuke), "zyx"

    Returns:
        Euler angles in degrees as [rx, ry, rz]

    Raises:
        ValueError: If unsupported rotation order
    """
    order = order.lower()

    if order == "xyz":
        sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotation[2, 1], rotation[2, 2])
            y = np.arctan2(-rotation[2, 0], sy)
            z = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            x = np.arctan2(-rotation[1, 2], rotation[1, 1])
            y = np.arctan2(-rotation[2, 0], sy)
            z = 0
        return np.degrees(np.array([x, y, z]))

    elif order == "zxy":
        cy = np.sqrt(rotation[0, 0] ** 2 + rotation[2, 0] ** 2)
        singular = cy < 1e-6
        if not singular:
            x = np.arctan2(-rotation[2, 1], rotation[1, 1])
            y = np.arctan2(rotation[2, 0], rotation[2, 2])
            z = np.arctan2(-rotation[0, 1], rotation[1, 1])
        else:
            x = np.arctan2(rotation[1, 2], rotation[1, 1])
            y = 0
            z = np.arctan2(-rotation[0, 1], rotation[0, 0])
        return np.degrees(np.array([x, y, z]))

    elif order == "zyx":
        cy = np.sqrt(rotation[0, 0] ** 2 + rotation[0, 1] ** 2)
        singular = cy < 1e-6
        if not singular:
            x = np.arctan2(rotation[1, 2], rotation[2, 2])
            y = np.arctan2(-rotation[0, 2], cy)
            z = np.arctan2(rotation[0, 1], rotation[0, 0])
        else:
            x = np.arctan2(-rotation[2, 1], rotation[1, 1])
            y = np.arctan2(-rotation[0, 2], cy)
            z = 0
        return np.degrees(np.array([x, y, z]))

    else:
        raise ValueError(f"Unsupported rotation order: {order}. Use 'xyz', 'zxy', or 'zyx'")


def euler_to_rotation_matrix(
    euler: np.ndarray,
    order: str = "xyz"
) -> np.ndarray:
    """Convert Euler angles to 3x3 rotation matrix.

    Args:
        euler: Euler angles in degrees as [rx, ry, rz]
        order: Rotation order - "xyz", "zxy", "zyx"

    Returns:
        3x3 rotation matrix
    """
    rx, ry, rz = np.radians(euler)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    order = order.lower()
    if order == "xyz":
        return Rz @ Ry @ Rx
    elif order == "zxy":
        return Ry @ Rx @ Rz
    elif order == "zyx":
        return Rx @ Ry @ Rz
    else:
        raise ValueError(f"Unsupported rotation order: {order}")


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions.

    Args:
        q1: Start quaternion [w, x, y, z]
        q2: End quaternion [w, x, y, z]
        t: Interpolation factor (0 = q1, 1 = q2)

    Returns:
        Interpolated quaternion [w, x, y, z]
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(np.clip(dot, -1, 1))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    if abs(sin_theta_0) < 1e-10:
        return q1

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two values.

    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0 = a, 1 = b)

    Returns:
        Interpolated value
    """
    return a + t * (b - a)


def cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float
) -> np.ndarray:
    """Cubic Bezier interpolation.

    Args:
        p0: Start point (control point 0)
        p1: Control point 1
        p2: Control point 2
        p3: End point (control point 3)
        t: Parameter (0 to 1)

    Returns:
        Interpolated point
    """
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3


def convert_opencv_to_opengl(matrix: np.ndarray) -> np.ndarray:
    """Convert camera matrix from OpenCV/COLMAP convention to OpenGL/DCC convention.

    OpenCV/COLMAP: X-right, Y-down, Z-forward (camera looks down +Z)
    OpenGL/DCC:    X-right, Y-up,   Z-back    (camera looks down -Z)

    This flips Y and Z axes to match Maya/Houdini/Blender/Nuke conventions.

    Args:
        matrix: 4x4 camera-to-world matrix in OpenCV convention

    Returns:
        4x4 camera-to-world matrix in OpenGL convention
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return matrix @ flip


def convert_opengl_to_opencv(matrix: np.ndarray) -> np.ndarray:
    """Convert camera matrix from OpenGL/DCC convention to OpenCV/COLMAP convention.

    Inverse of convert_opencv_to_opengl.

    Args:
        matrix: 4x4 camera-to-world matrix in OpenGL convention

    Returns:
        4x4 camera-to-world matrix in OpenCV convention
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return matrix @ flip


def matrix_to_alembic_xform(matrix: np.ndarray) -> list[float]:
    """Convert 4x4 numpy matrix to Alembic's column-major 16-element list.

    Alembic uses column-major order, numpy is row-major by default.

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        16-element list in column-major order
    """
    return matrix.T.flatten().tolist()


def compute_fov_from_intrinsics(
    intrinsics: dict,
    image_width: int,
    image_height: int
) -> tuple[float, float]:
    """Compute horizontal and vertical FOV from camera intrinsics.

    Args:
        intrinsics: Dict with fx, fy, cx, cy values
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Tuple of (horizontal_fov, vertical_fov) in degrees
    """
    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    fy = intrinsics.get("fy", intrinsics.get("focal_y", 1000))

    h_fov = 2 * np.degrees(np.arctan(image_width / (2 * fx)))
    v_fov = 2 * np.degrees(np.arctan(image_height / (2 * fy)))

    return h_fov, v_fov


def focal_length_to_fov(focal_length_mm: float, sensor_size_mm: float) -> float:
    """Convert focal length to field of view.

    Args:
        focal_length_mm: Focal length in millimeters
        sensor_size_mm: Sensor dimension in millimeters

    Returns:
        Field of view in degrees
    """
    return 2 * np.degrees(np.arctan(sensor_size_mm / (2 * focal_length_mm)))


def fov_to_focal_length(fov_degrees: float, sensor_size_mm: float) -> float:
    """Convert field of view to focal length.

    Args:
        fov_degrees: Field of view in degrees
        sensor_size_mm: Sensor dimension in millimeters

    Returns:
        Focal length in millimeters
    """
    return (sensor_size_mm / 2) / np.tan(np.radians(fov_degrees / 2))


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(q)
    if norm > 0:
        return q / norm
    return q


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate (inverse for unit quaternions).

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])
