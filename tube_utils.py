"""
Utility functions for pipe and borehole geometry, simulation, and analysis.

Provides core algorithms for:
- Frenet-Serret frame (TNB) computation
- Pipe deflection simulation
- Contact angle calculation
"""

from typing import Tuple, List
import numpy as np
import constants


def compute_tnb_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Frenet-Serret frame (Tangent, Normal, Binormal) for a curve.
    
    Args:
        points: Nx3 array of 3D points along the curve
        
    Returns:
        Tuple of (tangents: Nx3, normals: Nx3, binormals: Nx3) — unit vectors
        
    Raises:
        ValueError: If points array is invalid
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be Nx3 NumPy array")
    
    if len(points) < 2:
        raise ValueError("At least 2 points required")
    
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)

    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)

    arbitrary = np.array([0, 0, 1])
    if np.allclose(tangents[0], arbitrary):
        arbitrary = np.array([1, 0, 0])
    
    normals[0] = np.cross(tangents[0], arbitrary)
    normals[0] /= np.linalg.norm(normals[0]) + 1e-8
    binormals[0] = np.cross(tangents[0], normals[0])

    for i in range(1, len(points)):
        binormals[i] = np.cross(tangents[i], normals[i-1])
        binormals[i] /= np.linalg.norm(binormals[i]) + 1e-8
        normals[i] = np.cross(binormals[i], tangents[i])

    return tangents, normals, binormals



def simulate_pipe_deflection(
    centerline: np.ndarray,
    tube_radius: float,
    bore_radius: float,
    stiffness: float = constants.DEFLECTION_STIFFNESS,
    iterations: int = constants.DEFAULT_ITERATIONS
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Simulate pipe deflection under bore constraints using iterative constraint solver.
    
    Physics model: Iteratively moves interior points toward straightness while enforcing
    contact constraints with the bore boundary.
    
    Args:
        centerline: Nx3 array of original survey centerline
        tube_radius: Inner pipe radius in meters
        bore_radius: Outer bore radius in meters
        stiffness: Deformation stiffness parameter [0.0, 1.0] (default: 0.1)
        iterations: Number of simulation iterations (default: 60)
        
    Returns:
        Tuple of:
            - deflected_centerline: Nx3 array of deformed centerline
            - contact_points: N-length boolean array (True where contact occurs)
            - contact_positions: List of Kx3 arrays (K = number of contact points)
            - contact_forces: N-length float array (normalized [0.0, 1.0])
            
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not isinstance(centerline, np.ndarray) or centerline.ndim != 2 or centerline.shape[1] != 3:
        raise ValueError("Centerline must be Nx3 NumPy array")
    
    if tube_radius <= 0 or bore_radius <= 0:
        raise ValueError("Radii must be positive")
    
    if not (0.0 <= stiffness <= 1.0):
        raise ValueError("Stiffness must be in range [0.0, 1.0]")
    
    if iterations < 1:
        raise ValueError("Iterations must be >= 1")
    
    deflected_centerline = centerline.copy()
    contact_points = np.zeros(len(centerline), dtype=bool)
    contact_forces = np.zeros(len(centerline), dtype=float)
    contact_positions_dict = {}  # Track positions by index

    for _ in range(iterations):
        for i in range(1, len(centerline)-1):
            # Attraction force: move toward straightness
            straight_pos = (deflected_centerline[i-1] + deflected_centerline[i+1]) / 2
            move_vector = straight_pos - deflected_centerline[i]
            deflected_centerline[i] += stiffness * move_vector

            # Contact constraint: enforce bore boundary
            dist_to_center = np.linalg.norm(deflected_centerline[i] - centerline[i])
            penetration = (dist_to_center + tube_radius) - bore_radius
            
            if penetration > 0:
                # Point penetrates bore boundary; push back to boundary
                direction = (deflected_centerline[i] - centerline[i]) / (dist_to_center + 1e-10)
                deflected_centerline[i] = centerline[i] + direction * (bore_radius - tube_radius)
                contact_points[i] = True
                contact_forces[i] = min(1.0, penetration * constants.CONTACT_FORCE_VISUALIZATION_SCALE)
                contact_position = centerline[i] + direction * bore_radius
                contact_positions_dict[i] = contact_position
            else:
                contact_forces[i] = 0.0

    # Convert contact_positions_dict to list ordered by index
    contact_indices = sorted(contact_positions_dict.keys())
    contact_positions = [contact_positions_dict[i] for i in contact_indices]

    return deflected_centerline, contact_points, contact_positions, contact_forces

def compute_contact_angles(
    centerline: np.ndarray,
    contact_indices: List[int],
    contact_positions: List[np.ndarray]
) -> List[Tuple[float, float]]:
    """
    Compute contact point angles and arc-length distances from centerline start.
    
    For each contact point, computes its position in the local pipe cross-section
    coordinate system (Normal, Binormal), expressed as an angle and distance along
    the centerline.
    
    Args:
        centerline: Nx3 array of centerline points
        contact_indices: List of indices where contact occurs
        contact_positions: List of 3D contact point positions
        
    Returns:
        List of tuples (arc_distance_m: float, angle_deg: float) for each contact
        where angle is in range [-180°, 180°]:
        - 0° = right (Normal direction)
        - 90° = up (Binormal direction)
        - 180° = left
        - -90° = down
        
    Raises:
        ValueError: If inputs are invalid
    """
    if len(contact_indices) != len(contact_positions):
        raise ValueError("contact_indices and contact_positions lengths must match")
    
    results = []
    for i, idx in enumerate(contact_indices):
        # Compute arc-length distance from start to contact point
        segment_distance = np.sum([
            np.linalg.norm(centerline[j] - centerline[j-1])
            for j in range(1, min(idx + 1, len(centerline)))
        ])
        
        # Compute local tangent vector (centerline direction)
        if idx == 0:
            tangent = centerline[1] - centerline[0]
        elif idx == len(centerline) - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[idx+1] - centerline[idx-1]
        
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
        
        # Compute normal and binormal vectors (local frame)
        up = np.array([0, 0, 1])
        normal = np.cross(tangent, up)
        norm_mag = np.linalg.norm(normal)
        
        if norm_mag > 1e-10:
            normal = normal / norm_mag
        else:
            normal = np.array([1, 0, 0])
        
        binormal = np.cross(tangent, normal)
        
        # Project contact position onto local frame
        pos = contact_positions[i]
        center = centerline[idx]
        vec = pos - center
        
        x = np.dot(vec, normal)
        y = np.dot(vec, binormal)
        angle_rad = np.arctan2(y, x)
        angle_deg = np.degrees(angle_rad)
        
        results.append((segment_distance, angle_deg))
    
    return results



