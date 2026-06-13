"""
Utility functions for pipe and borehole geometry, simulation, and analysis.
"""

import numpy as np

def compute_tnb_frame(points):
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





def simulate_pipe_deflection(centerline, tube_radius, bore_radius, stiffness, iterations):
    deflected_centerline = centerline.copy()
    contact_points = np.zeros(len(centerline), dtype=bool)
    contact_forces = np.zeros(len(centerline), dtype=float)
    contact_positions = []

    for _ in range(iterations):
        for i in range(1, len(centerline)-1):
            straight_pos = (deflected_centerline[i-1] + deflected_centerline[i+1]) / 2
            move_vector = straight_pos - deflected_centerline[i]
            deflected_centerline[i] += stiffness * move_vector

            dist_to_center = np.linalg.norm(deflected_centerline[i] - centerline[i])
            penetration = (dist_to_center + tube_radius) - bore_radius
            if penetration > 0:
                direction = (deflected_centerline[i] - centerline[i]) / dist_to_center
                deflected_centerline[i] = centerline[i] + direction * (bore_radius - tube_radius)
                contact_points[i] = True
                contact_forces[i] = min(1.0, penetration * 10)  # scale for visualization
                contact_position = centerline[i] + direction * bore_radius
                contact_positions.append(contact_position)
            else:
                contact_forces[i] = 0.0

    return deflected_centerline, contact_points, contact_positions, contact_forces



def compute_contact_angles(centerline, contact_indices, contact_positions):
    """
    Zwraca listę (dystans od początku, kąt lokalny [deg]) dla każdego punktu styku.
    """
    results = []
    for i, idx in enumerate(contact_indices):
        # Dystans od początku do punktu styku
        segment_distance = np.sum([np.linalg.norm(centerline[j] - centerline[j-1]) for j in range(1, idx+1)])
        
        # Lokalny układ odniesienia (przyjmij, że tangent to kierunek otworu)
        if idx == 0:
            tangent = centerline[1] - centerline[0]
        elif idx == len(centerline) - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[idx+1] - centerline[idx-1]
        tangent = tangent / np.linalg.norm(tangent)
        
        # Zakładamy, że Z to "do góry", wyznacz binormalną
        up = np.array([0,0,1])
        normal = np.cross(tangent, up)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 1e-10 else np.array([1,0,0])
        binormal = np.cross(tangent, normal)
        
        # Pozycja styku względem środka przekroju
        pos = contact_positions[i]
        center = centerline[idx]
        vec = pos - center
        
        # Rzutuj na płaszczyznę przekroju (normal, binormal)
        x = np.dot(vec, normal)
        y = np.dot(vec, binormal)
        angle_rad = np.arctan2(y, x)  # 0° = prawa, 90° = góra, 180° = lewa, -90° = dół
        angle_deg = np.degrees(angle_rad)
        
        results.append((segment_distance, angle_deg))
    return results



