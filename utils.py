# utils.py
import numpy as np
import math
from typing import Tuple
import constants as const

from constants import GEOMETRY_TOLERANCE, ANGLE_OVERLAP_TOLERANCE

def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalizes a numpy vector."""
    norm = np.linalg.norm(vector)
    if norm < GEOMETRY_TOLERANCE:
        # Return zero vector or handle as error? For now, zero vector.
        return np.zeros_like(vector)
    return vector / norm

def angle_dist(a1: float, a2: float) -> float:
    """Calculates the shortest distance between two angles in radians (-pi to pi)."""
    return (a2 - a1 + math.pi) % (2 * math.pi) - math.pi

def is_angle_within(
    angle: float, start_angle: float, end_angle: float, tolerance: float = ANGLE_OVERLAP_TOLERANCE
) -> bool:
    """
    Checks if angle is within [start_angle, end_angle), handling wrap correctly.
    Angles are in radians. Start and end define the range counterclockwise.
    """
    # Normalize all angles to [0, 2*pi)
    angle_norm = (angle + 4 * math.pi) % (2 * math.pi)
    start_norm = (start_angle + 4 * math.pi) % (2 * math.pi)
    end_norm = (end_angle + 4 * math.pi) % (2 * math.pi)

    # Handle edge case where start and end are effectively the same (full circle)
    if abs(angle_dist(start_norm, end_norm)) < tolerance:
        return True # Covers everything

    # Check based on wrap-around
    if start_norm <= end_norm:
        # Normal case: start ---- angle ---- end
        # Check if angle is within [start, end)
        # Use tolerance carefully: >= start and < end
        return start_norm - tolerance <= angle_norm < end_norm - tolerance
    else:
        # Wrap case: angle --- end --- start --- angle
        # Check if angle is >= start OR angle < end
        return (
            start_norm - tolerance <= angle_norm < 2 * math.pi
            or 0 <= angle_norm < end_norm - tolerance
        )

def angles_overlap(start1: float, end1: float, start2: float, end2: float, tolerance: float = ANGLE_OVERLAP_TOLERANCE) -> bool:
    """
    Checks if two angular ranges [start, end] overlap.
    Assumes angles are in [0, 2pi) initially, end can be > start.
    Handles wrapping correctly.
    """
    # Normalize angles to handle potential negative inputs or values > 2pi
    n_start1 = (start1 + 4 * math.pi) % (2 * math.pi)
    n_end1 = (end1 + 4 * math.pi) % (2 * math.pi)
    n_start2 = (start2 + 4 * math.pi) % (2 * math.pi)
    n_end2 = (end2 + 4 * math.pi) % (2 * math.pi)

    # If the range endpoint is (almost) the same as the start, it covers the full circle
    if abs(angle_dist(n_start1, n_end1)) < tolerance:
        return True # Full circle range 1
    if abs(angle_dist(n_start2, n_end2)) < tolerance:
        return True # Full circle range 2

    # Does range 1 contain range 2's start? OR Does range 2 contain range 1's start?
    range1_contains_start2 = is_angle_within(n_start2, n_start1, n_end1, tolerance)
    range2_contains_start1 = is_angle_within(n_start1, n_start2, n_end2, tolerance)

    return range1_contains_start2 or range2_contains_start1

def map_point_to_hemisphere(
    point_2d: Tuple[float, float], max_maze_radius: float, sphere_radius: float
) -> np.ndarray:
    """Maps a 2D point (polar coords origin) to a hemisphere surface via Azimuthal Equidistant Projection."""
    x, y = point_2d
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x) # Angle remains the same

    if max_maze_radius <= GEOMETRY_TOLERANCE:
        # Avoid division by zero, map origin to north pole
        return np.array([0.0, 0.0, sphere_radius])

    # Clamp the radius to the maze boundary
    r_clamped = min(r, max_maze_radius)

    # Calculate the polar angle (phi) on the sphere (0 at pole, pi/2 at equator)
    # Linear mapping: r=0 -> phi=0, r=max_maze_radius -> phi=pi/2
    phi = (r_clamped / max_maze_radius) * (math.pi / 2.0)

    # Convert spherical (radius, theta, phi) to Cartesian coordinates
    x_sph = sphere_radius * math.sin(phi) * math.cos(theta)
    y_sph = sphere_radius * math.sin(phi) * math.sin(theta)
    z_sph = sphere_radius * math.cos(phi)

    return np.array([x_sph, y_sph, z_sph])

# Optional: Add check for scipy availability if needed by multiple modules
try:
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # print("Warning: scipy not found. Some checks might be less robust.")
    
def map_point_to_lower_hemisphere(
    point_2d: Tuple[float, float], max_maze_radius: float, sphere_radius: float
) -> np.ndarray:
    """Maps a 2D point to a LOWER hemisphere surface via Azimuthal Equidistant Projection."""
    x, y = point_2d
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    if max_maze_radius <= const.GEOMETRY_TOLERANCE:
        return np.array([0.0, 0.0, -sphere_radius]) # South pole
    r_clamped = min(r, max_maze_radius)
    # Phi is angle from the pole (0=pole, pi/2=equator)
    phi = (r_clamped / max_maze_radius) * (math.pi / 2.0)
    # Spherical to Cartesian conversion
    x_sph = sphere_radius * math.sin(phi) * math.cos(theta)
    y_sph = sphere_radius * math.sin(phi) * math.sin(theta)
    # Z becomes negative for lower hemisphere
    z_sph = -sphere_radius * math.cos(phi)
    return np.array([x_sph, y_sph, z_sph])