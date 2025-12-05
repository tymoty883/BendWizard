import numpy as np

def compute_clearance(inner_path, outer_centerline, outer_radii, inner_radius):
    """
    For each point on the inner_path, compute the minimum distance to the outer tube surface.
    Returns an array of clearance values (positive: inside, negative: outside).
    """
    clearance = []
    for i, pt in enumerate(inner_path):
        # Find closest point on outer centerline
        dists = np.linalg.norm(outer_centerline - pt, axis=1)
        idx = np.argmin(dists)
        # Surface-to-surface clearance
        c = outer_radii[idx] - inner_radius - dists[idx]
        clearance.append(c)
    return np.array(clearance)

def optimize_inner_path(outer_centerline, outer_radii, inner_radius, smoothness=0.1, steps=200):
    """
    Simple greedy local optimization: for each point, try to maximize clearance
    while keeping start/end fixed and path smooth.
    """
    path = np.copy(outer_centerline)
    for step in range(steps):
        for i in range(1, len(path)-1):
            orig = path[i].copy()
            best = orig.copy()
            best_clear = -np.inf
            for dx in np.linspace(-smoothness, smoothness, 5):
                for dy in np.linspace(-smoothness, smoothness, 5):
                    for dz in np.linspace(-smoothness, smoothness, 5):
                        candidate = orig + np.array([dx, dy, dz])
                        # Smoothness constraint
                        if np.linalg.norm(candidate-path[i-1]) > 2*smoothness: continue
                        if np.linalg.norm(candidate-path[i+1]) > 2*smoothness: continue
                        # Clearance
                        d = np.linalg.norm(candidate - outer_centerline[i])
                        clear = outer_radii[i] - inner_radius - d
                        if clear > best_clear:
                            best = candidate
                            best_clear = clear
            path[i] = best
    return path
