import numpy as np

class TubeGenerator:
    @staticmethod
    def compute_tube_geometry(centerline, radius, n_points):
        n_centerline = len(centerline)
        tangents = np.zeros_like(centerline)
        tangents[1:-1] = centerline[2:] - centerline[:-2]
        tangents[0] = centerline[1] - centerline[0]
        tangents[-1] = centerline[-1] - centerline[-2]
        
        magnitudes = np.linalg.norm(tangents, axis=1, keepdims=True)
        magnitudes[magnitudes < 1e-10] = 1e-10
        tangents = tangents / magnitudes
        
        normals, binormals = TubeGenerator._compute_normals_and_binormals(tangents)
        vertices, faces, vertex_normals = TubeGenerator._generate_tube_mesh(centerline, radius, n_points, tangents, normals, binormals)
        
        return vertices, faces, vertex_normals

    @staticmethod
    def _compute_normals_and_binormals(tangents):
        n_centerline = len(tangents)
        normals = np.zeros_like(tangents)
        binormals = np.zeros_like(tangents)
        
        t0 = tangents[0]
        min_idx = np.argmin(np.abs(t0))
        v = np.zeros(3)
        v[min_idx] = 1.0
        normal0 = v - np.dot(v, t0) * t0
        norm_normal0 = np.linalg.norm(normal0)
        
        if norm_normal0 < 1e-10:
            other_idx = (min_idx + 1) % 3
            v = np.zeros(3)
            v[other_idx] = 1.0
            normal0 = v - np.dot(v, t0) * t0
            norm_normal0 = np.linalg.norm(normal0)
        
        normal0 = normal0 / max(norm_normal0, 1e-10)
        normals[0] = normal0
        binormals[0] = np.cross(tangents[0], normals[0])
        
        for i in range(1, n_centerline):
            t = tangents[i]
            dot_product = np.dot(normals[i-1], t)
            normal = normals[i-1] - dot_product * t
            
            norm_normal = np.linalg.norm(normal)
            if norm_normal < 1e-6:
                min_idx = np.argmin(np.abs(t))
                v = np.zeros(3)
                v[min_idx] = 1.0
                normal = v - np.dot(v, t) * t
                
                if np.linalg.norm(normal) < 1e-6:
                    other_idx = (min_idx + 1) % 3
                    v = np.zeros(3)
                    v[other_idx] = 1.0
                    normal = v - np.dot(v, t) * t
            
            normals[i] = normal / max(np.linalg.norm(normal), 1e-10)
            binormals[i] = np.cross(t, normals[i])
        
        return normals, binormals

    @staticmethod
    def _generate_tube_mesh(centerline, radius, n_points, tangents, normals, binormals):
        n_centerline = len(centerline)
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        
        vertices = np.zeros((n_centerline * n_points, 3))
        vertex_normals = np.zeros((n_centerline * n_points, 3))
        
        # Generate vertices and normals for the tube
        for i in range(n_centerline):
            n = normals[i]
            b = binormals[i]
            
            for j in range(n_points):
                idx = i * n_points + j
                # Calculate vertex position
                vertices[idx] = centerline[i] + circle_x[j] * n + circle_y[j] * b
                # Calculate normal pointing outward from tube center
                normal_dir = circle_x[j] * n + circle_y[j] * b
                vertex_normals[idx] = normal_dir / max(np.linalg.norm(normal_dir), 1e-10)
        
        # Generate faces for the tube
        faces = []
        for i in range(n_centerline - 1):
            for j in range(n_points):
                p0 = i * n_points + j
                p1 = i * n_points + ((j + 1) % n_points)
                p2 = (i + 1) * n_points + ((j + 1) % n_points)
                p3 = (i + 1) * n_points + j
                
                # Create consistent winding order for all faces
                faces.append([p0, p1, p2])  # First triangle
                faces.append([p0, p2, p3])  # Second triangle
        
        return vertices, np.array(faces), vertex_normals 