# mesh_builder.py

import numpy as np
import trimesh
import math
from typing import List, Tuple, Optional, Callable

# --- Ensure this import is present ---
import constants as const

# Import from other project modules
# Assuming grid_core contains CircularGrid
from grid_core import CircularGrid  # Added this import
from utils import (
    map_point_to_hemisphere,
    map_point_to_lower_hemisphere,
    normalize,
    SCIPY_AVAILABLE,
    is_angle_within,
)

# Conditional import for pdist
if SCIPY_AVAILABLE:
    try:
        from scipy.spatial.distance import pdist
    except ImportError:
        pdist = None
        print("Warning: scipy found but pdist couldn't be imported.")
else:
    pdist = None

# Imports needed for base generation/outline/boolean
import trimesh.path
import trimesh.path.entities
import trimesh.geometry
import trimesh.boolean
import trimesh.creation
import trimesh.transformations

# import trimesh.interfaces.blender # Not strictly needed for generation


# --- Helper Functions (_generate_spherical_patch_grid, _triangulate_grid, _is_mesh_degenerate) ---
# (Keep these as they are used by the correct method)
def _generate_spherical_patch_grid(
    corner_verts_2d: List[Tuple[float, float]],
    max_maze_radius: float,
    sphere_radius: float,
    num_subdiv_u: int,
    num_subdiv_v: int,
    map_func: Callable,
) -> np.ndarray:
    """Generates a grid of 3D points on a spherical surface corresponding to the 2D patch."""
    if len(corner_verts_2d) != 4:
        raise ValueError("Need 4 corner vertices for the 2D patch.")
    v0_2d, v1_2d, v2_2d, v3_2d = [np.array(p) for p in corner_verts_2d]
    points_3d = []
    if num_subdiv_u < 1:
        num_subdiv_u = 1
    if num_subdiv_v < 1:
        num_subdiv_v = 1
    for j in range(num_subdiv_v + 1):
        v_ratio = j / num_subdiv_v
        p_start_edge = (1 - v_ratio) * v0_2d + v_ratio * v1_2d  # Edge v0-v1
        p_end_edge = (1 - v_ratio) * v3_2d + v_ratio * v2_2d  # Edge v3-v2
        for i in range(num_subdiv_u + 1):
            u_ratio = i / num_subdiv_u
            current_pt_2d = (1 - u_ratio) * p_start_edge + u_ratio * p_end_edge
            if not np.all(np.isfinite(current_pt_2d)):  # Basic check for safety
                print(
                    f"WARN: Non-finite 2D point generated during grid interpolation. Using average."
                )
                current_pt_2d = np.mean(corner_verts_2d, axis=0)
                if not np.all(np.isfinite(current_pt_2d)):
                    print(f"WARN: Average 2D point is also non-finite. Using origin.")
                    current_pt_2d = np.array([0.0, 0.0])
            point_3d = map_func(tuple(current_pt_2d), max_maze_radius, sphere_radius)
            points_3d.append(point_3d)
    return np.array(points_3d)


def _triangulate_grid(
    num_subdiv_u: int,
    num_subdiv_v: int,
    start_index: int = 0,
    reverse_winding: bool = False,
) -> List[List[int]]:
    """Creates faces (triangles) for a grid defined by subdivisions u and v."""
    faces = []
    if num_subdiv_u < 1:
        num_subdiv_u = 1
    if num_subdiv_v < 1:
        num_subdiv_v = 1
    cols = num_subdiv_u + 1
    for j in range(num_subdiv_v):
        for i in range(num_subdiv_u):
            idx00 = start_index + j * cols + i
            idx10 = start_index + j * cols + (i + 1)
            idx01 = start_index + (j + 1) * cols + i
            idx11 = start_index + (j + 1) * cols + (i + 1)
            if not reverse_winding:
                faces.extend(
                    [[idx00, idx10, idx11], [idx00, idx11, idx01]]
                )  # Standard CCW winding when viewed from outside
            else:
                faces.extend(
                    [[idx00, idx11, idx10], [idx00, idx01, idx11]]
                )  # Reversed CW winding
    return faces


def _is_mesh_degenerate(vertices: np.ndarray) -> bool:
    """Checks if vertices are too close together."""
    if vertices.shape[0] <= 1:
        return False
    min_dist_sq = const.MESH_VERTEX_DISTANCE_TOLERANCE_SQ  # Uses const
    if SCIPY_AVAILABLE and pdist is not None and vertices.shape[0] > 1:
        try:
            finite_verts = vertices[np.all(np.isfinite(vertices), axis=1)]
            if finite_verts.shape[0] <= 1:
                return False
            # pdist calculates pairwise distances; check if any are below tolerance
            return np.any(pdist(finite_verts) ** 2 < min_dist_sq)
        except Exception as e:
            # print(f"DEBUG: pdist check failed: {e}. Falling back.") # Optional debug
            pass  # Fallback to manual check

    # Manual fallback check (slower)
    for i in range(vertices.shape[0]):
        if not np.all(np.isfinite(vertices[i])):
            continue
        for j in range(i + 1, vertices.shape[0]):
            if not np.all(np.isfinite(vertices[j])):
                continue
            if np.sum((vertices[i] - vertices[j]) ** 2) < min_dist_sq:
                # print(f"DEBUG: Degenerate vertices found: {i} and {j}") # Optional debug
                return True
    return False


# --- Remove Prism Helper Functions ---
# These simple extrusion methods are not suitable for the curved walls.
# def _create_extruded_prism_simple_z_centered(base_verts_2d, height): ... REMOVED ...
# def _create_prism_on_equator(base_verts_2d, height: float, hemisphere_type: str): ... REMOVED ...


def create_hemisphere_base_shell(
    outer_shell_radius: float,
    base_thickness: float,
    subdivisions: int,  # Controls smoothness of the revolved arc profile
    hemisphere_type: str = "upper",
) -> Optional[trimesh.Trimesh]:
    """
    Creates a closed, thick hemisphere shell by revolving a 2D profile.
    """
    print(f"\n--- Creating Thick Hemisphere Base Shell (Revolve Profile) ---")
    print(
        f"  Type: {hemisphere_type}, Outer Radius: {outer_shell_radius:.3f}, Thickness: {base_thickness:.3f}"
    )

    inner_radius = outer_shell_radius - base_thickness
    if inner_radius < 0:  # Allow inner radius to be exactly 0
        print(f"WARN: Inner radius ({inner_radius:.3f}) is negative, setting to 0.")
        inner_radius = 0.0
    # Ensure outer radius is strictly larger than inner radius if inner > 0
    if (
        inner_radius > 0
        and inner_radius >= outer_shell_radius - const.GEOMETRY_TOLERANCE
    ):
        print(
            f"ERROR: Inner radius ({inner_radius:.3f}) must be smaller than outer radius ({outer_shell_radius:.3f})."
        )
        return None

    print(f"  Inner Radius: {inner_radius:.3f}, Profile Segments={subdivisions}")

    try:
        # --- Define the 2D profile in the XZ plane (X=radius, Z=height) ---
        profile_points = []
        num_segments = subdivisions  # Number of segments for the quarter-circle arcs

        # 1. Outer Arc points (from Z=0 up/down to pole)
        start_angle = 0
        end_angle = math.pi / 2.0 if hemisphere_type == "upper" else -math.pi / 2.0
        outer_arc_angles = np.linspace(start_angle, end_angle, num_segments + 1)
        # Calculate X (radius) and Z (height) for outer arc
        outer_arc_x = outer_shell_radius * np.cos(outer_arc_angles)
        outer_arc_z = outer_shell_radius * np.sin(outer_arc_angles)
        # Add points from equator towards pole
        for i in range(num_segments + 1):
            profile_points.append([outer_arc_x[i], outer_arc_z[i]])

        # 2. Inner Arc points (from pole down/up to Z=0)
        # Only add if inner_radius > 0
        if inner_radius > const.GEOMETRY_TOLERANCE:
            inner_arc_angles = np.linspace(
                end_angle, start_angle, num_segments + 1
            )  # Reverse direction
            inner_arc_x = inner_radius * np.cos(inner_arc_angles)
            inner_arc_z = inner_radius * np.sin(inner_arc_angles)
            # Add points from pole towards equator (skip first point as it duplicates last outer point at pole)
            for i in range(1, num_segments + 1):
                profile_points.append([inner_arc_x[i], inner_arc_z[i]])
        else:
            # If inner radius is 0, add a point at the origin to close the profile at the pole
            # The last point of the outer arc is at [0, +/-outer_shell_radius]
            # We need to connect this to the origin [0, 0]
            profile_points.append([0.0, 0.0])  # Connect pole to origin

        # 3. Close the polygon loop back to the start point (outer radius at Z=0)
        # The first point added was [outer_shell_radius, 0.0]
        # The last point added was [inner_radius, 0.0] or [0.0, 0.0]
        # No explicit closing needed if revolve handles it, but Path2D needs closed loop.
        # Let's ensure it's explicitly closed for Path2D
        # profile_points.append(profile_points[0]) # This might create duplicate vertex

        profile_polygon = np.array(profile_points)

        # Check for duplicate start/end points
        if (
            np.linalg.norm(profile_polygon[0] - profile_polygon[-1])
            < const.GEOMETRY_TOLERANCE
        ):
            profile_polygon = profile_polygon[:-1]  # Remove duplicate end point

        print(f"  Generated profile polygon with {len(profile_polygon)} vertices.")

        # --- Revolve the profile ---
        print("  Revolving profile polygon...")
        # Revolve around Z-axis (index 2) which corresponds to Y-axis in Path2D frame
        # section_angle determines how far to revolve (2*pi for full circle)
        final_shell = trimesh.creation.revolve(
            profile_polygon,
            angle=2 * math.pi,
            section_segments=256,  # Use rim_segments for revolution smoothness
            # axis=[0,1,0] # Default revolve axis for Path2D is Y (which is Z in our XZ profile)
            # transform= # Optional transform if profile wasn't in XZ
        )

        if not final_shell or len(final_shell.faces) == 0:
            print("ERROR: Revolution resulted in empty mesh.")
            return None

        print(
            f"  Revolution complete: {len(final_shell.vertices)}V, {len(final_shell.faces)}F"
        )

        # --- Final Checks and Processing ---
        print("  Merging vertices...")
        final_shell.merge_vertices()
        print("  Fixing normals...")
        final_shell.fix_normals()  # Crucial after revolution

        # Check watertightness
        if not final_shell.is_watertight:
            print(
                f"WARN: Final shell mesh is not watertight after revolution. Attempting repair."
            )
            final_shell.fill_holes()
            final_shell.fix_normals()  # Fix again
            if not final_shell.is_watertight:
                print(f"WARN: Repair failed. Shell remains non-watertight.")
        else:
            print("  Final shell appears watertight.")

        # Optional final processing
        try:
            print("    Processing final base mesh...")
            final_shell.process()
            print(
                f"    Processing complete. Final mesh: {len(final_shell.vertices)}V, {len(final_shell.faces)}F"
            )
        except Exception as process_err:
            print(f"WARN: Final base mesh processing failed: {process_err}")

        print(
            f"  SUCCESS: Thick base SHELL generated for {hemisphere_type} using revolve."
        )
        return final_shell

    except Exception as e:
        print(f"ERROR creating thick base shell: {e}")
        import traceback

        traceback.print_exc()
        return None


# --- Wall Function (Re-enable Degeneracy Check, Remove Debug Prints) ---
def create_hemisphere_wall_mesh(
    wall_base_vertices_2d: List[Tuple[Tuple[float, float], ...]],
    max_maze_radius: float,  # Mapping domain radius
    sphere_radius: float,  # Sphere surface radius (base for wall patch)
    wall_height: float,
    num_subdivisions: int = 3,
    wall_thickness: float = const.DEFAULT_WALL_THICKNESS_3D,
    hemisphere_type: str = "upper",
    generate_equator_wall: bool = False,  # Controls if opening check is done
    opening_params: Optional[Tuple[float, float]] = None,
    outer_wall_center_r: Optional[float] = None,  # For identifying outer segments
) -> trimesh.Trimesh:
    """
    Creates 3D wall meshes using curved spherical patches for ALL segments.
    Re-enables degeneracy check and removes opening debug prints.
    Relies on the base shell being extended appropriately (for Hemi A).
    Uses outer_wall_center_r ONLY to identify segments for opening removal.
    """
    print(f"\n--- Creating 3D Hemi Walls (Standard Patch Method) ---")
    print(
        f"  Hemi: {hemisphere_type}, CheckOpening: {generate_equator_wall}, Opening: {opening_params is not None}"
    )
    print(
        f"  Mapping Domain/Wall Base Radius: {max_maze_radius:.4f}, Wall Height: {wall_height:.3f}"
    )
    # print(f"  Outer Wall Center Radius (for opening check): {outer_wall_center_r}") # Optional print

    map_func = (
        map_point_to_hemisphere
        if hemisphere_type == "upper"
        else map_point_to_lower_hemisphere
    )
    all_wall_meshes: List[trimesh.Trimesh] = []
    base_count = len(wall_base_vertices_2d)
    gen_count, skip_opening_count, skip_other_count, err_count = 0, 0, 0, 0
    processed_count = 0

    # --- Calculate Opening Range with Tolerance ---
    opening_start_angle_norm, opening_end_angle_norm = None, None
    if opening_params:
        connection_angle, connection_width_angle = opening_params
        angle_tolerance = 1e-5
        half_width = connection_width_angle / 2.0 + angle_tolerance
        opening_start_angle_norm = (connection_angle - half_width + 4 * math.pi) % (
            2 * math.pi
        )
        opening_end_angle_norm = (connection_angle + half_width + 4 * math.pi) % (
            2 * math.pi
        )
        # print(f"  Opening Check Range: [{opening_start_angle_norm:.4f}, {opening_end_angle_norm:.4f}]") # Optional

    print(f"  Processing {base_count} 2D base polygons...")
    for i, base_verts_2d_quad in enumerate(wall_base_vertices_2d):
        processed_count += 1
        # --- Input validation ---
        if not (
            isinstance(base_verts_2d_quad, (list, tuple))
            and len(base_verts_2d_quad) == 4
            and all(
                isinstance(p, (list, tuple)) and len(p) == 2 for p in base_verts_2d_quad
            )
        ):
            skip_other_count += 1
            continue

        mesh = None
        should_skip_for_opening = False
        exception_occurred = False

        # --- Check if this segment should be skipped ---
        if generate_equator_wall and opening_params:
            is_outer_wall_segment = False
            if (
                outer_wall_center_r is not None
                and outer_wall_center_r > const.GEOMETRY_TOLERANCE
            ):
                half_thick = wall_thickness / 2.0
                try:
                    avg_r_2d = np.mean([np.linalg.norm(p) for p in base_verts_2d_quad])
                    if abs(avg_r_2d - outer_wall_center_r) < half_thick * 1.1:
                        is_outer_wall_segment = True
                except Exception as e_dist:
                    pass

            if is_outer_wall_segment:
                try:
                    # Calculate segment mid-angle
                    v0_2d, v1_2d, v2_2d, v3_2d = base_verts_2d_quad
                    angle1 = math.atan2(v1_2d[1], v1_2d[0])
                    angle2 = math.atan2(v2_2d[1], v2_2d[0])
                    vec1 = np.array([math.cos(angle1), math.sin(angle1)])
                    vec2 = np.array([math.cos(angle2), math.sin(angle2)])
                    avg_vec = normalize(vec1 + vec2)
                    segment_mid_angle = math.atan2(avg_vec[1], avg_vec[0])
                    segment_mid_angle_norm = (segment_mid_angle + 2 * math.pi) % (
                        2 * math.pi
                    )

                    # Check if angle is within range
                    if is_angle_within(
                        segment_mid_angle_norm,
                        opening_start_angle_norm,
                        opening_end_angle_norm,
                    ):
                        should_skip_for_opening = True
                except Exception as e_angle:
                    pass

        # --- Generate Wall Mesh using Spherical Patch Method (if not skipped) ---
        if not should_skip_for_opening:
            try:
                # Define radii, subdivisions, etc.
                sphere_radius_base = sphere_radius
                sphere_radius_top = sphere_radius_base + wall_height
                num_subdiv_u = num_subdivisions
                num_subdiv_v = num_subdivisions
                cols = num_subdiv_u + 1
                num_grid_points_per_patch = (num_subdiv_u + 1) * (num_subdiv_v + 1)

                # Generate grid points
                base_grid_points_3d = _generate_spherical_patch_grid(
                    base_verts_2d_quad,
                    max_maze_radius,
                    sphere_radius_base,
                    num_subdiv_u,
                    num_subdiv_v,
                    map_func=map_func,
                )
                top_grid_points_3d = _generate_spherical_patch_grid(
                    base_verts_2d_quad,
                    max_maze_radius,
                    sphere_radius_top,
                    num_subdiv_u,
                    num_subdiv_v,
                    map_func=map_func,
                )
                verts = np.vstack((base_grid_points_3d, top_grid_points_3d))

                # --- RE-ENABLE Degeneracy Check ---
                if _is_mesh_degenerate(verts):
                    # print(f"DEBUG [{i}]: Skipping (Degenerate Vertices)") # Optional
                    skip_other_count += 1
                    continue  # Skip this segment if degenerate

                # Generate faces (top, bottom, sides)
                faces = []
                faces.extend(
                    _triangulate_grid(
                        num_subdiv_u,
                        num_subdiv_v,
                        start_index=num_grid_points_per_patch,
                        reverse_winding=(hemisphere_type == "upper"),
                    )
                )
                faces.extend(
                    _triangulate_grid(
                        num_subdiv_u,
                        num_subdiv_v,
                        start_index=0,
                        reverse_winding=(hemisphere_type == "lower"),
                    )
                )
                # Corrected Side faces
                for k in range(num_subdiv_v):
                    b0 = k * cols + 0
                    b1 = (k + 1) * cols + 0
                    t0 = b0 + num_grid_points_per_patch
                    t1 = b1 + num_grid_points_per_patch
                    faces.extend([[b0, b1, t1], [b0, t1, t0]])
                for k in range(num_subdiv_u):
                    b0 = num_subdiv_v * cols + k
                    b1 = num_subdiv_v * cols + (k + 1)
                    t0 = b0 + num_grid_points_per_patch
                    t1 = b1 + num_grid_points_per_patch
                    faces.extend([[b0, b1, t1], [b0, t1, t0]])
                for k in range(num_subdiv_v):
                    b0 = (num_subdiv_v - k) * cols + num_subdiv_u
                    b1 = (num_subdiv_v - k - 1) * cols + num_subdiv_u
                    t0 = b0 + num_grid_points_per_patch
                    t1 = b1 + num_grid_points_per_patch
                    faces.extend([[b0, b1, t1], [b0, t1, t0]])
                for k in range(num_subdiv_u):
                    b0 = 0 * cols + (num_subdiv_u - k)
                    b1 = 0 * cols + (num_subdiv_u - k - 1)
                    t0 = b0 + num_grid_points_per_patch
                    t1 = b1 + num_grid_points_per_patch
                    faces.extend([[b0, b1, t1], [b0, t1, t0]])

                # Create Trimesh object
                if faces:
                    faces_array = np.array(faces, dtype=np.int32)
                    if faces_array.ndim == 2 and faces_array.shape[1] == 3:
                        if np.all(np.isfinite(verts)):
                            mesh = trimesh.Trimesh(
                                vertices=verts, faces=faces_array, process=False
                            )

            except Exception as e:
                # print(f"DEBUG [{i}]: EXCEPTION during generation: {e}")
                exception_occurred = True
                err_count += 1
        # else: pass # Skipped for opening

        # --- Validation and Collection ---
        if should_skip_for_opening:
            skip_opening_count += 1
        elif mesh and len(mesh.faces) > 0 and np.all(np.isfinite(mesh.vertices)):
            if np.max(mesh.faces) < len(mesh.vertices):
                all_wall_meshes.append(mesh)
                gen_count += 1
            else:
                skip_other_count += 1  # Failed index check
        elif exception_occurred:
            pass  # Error already counted
        else:
            skip_other_count += 1  # Failed validation or no mesh

    # --- Summary and Combination ---
    print(f"  Wall Mesh Generation Summary:")
    print(
        f"    Processed={processed_count}, Generated={gen_count}, Skipped(Opening)={skip_opening_count}, Skipped(Other)={skip_other_count}, Errors={err_count}"
    )

    if not all_wall_meshes:
        print("ERROR: No valid wall meshes generated.")
        return trimesh.Trimesh()

    print(f"  Combining {len(all_wall_meshes)} wall meshes...")
    try:
        # Combine, merge, fix normals
        combined = trimesh.util.concatenate(all_wall_meshes)
        print(
            f"    Initial Combined: {len(combined.vertices)}V, {len(combined.faces)}F"
        )
        combined.merge_vertices()
        print(f"    After Merge: {len(combined.vertices)}V, {len(combined.faces)}F")
        combined.fix_normals()
        print("    Normals fixed.")

        # Optional post-processing
        try:
            processed_mesh = combined.copy()
            processed_mesh.process()
            print("    Processing complete.")
            combined = processed_mesh
        except Exception as process_err:
            print(
                f"WARN: mesh.process() failed: {process_err}. Using unprocessed mesh."
            )

        if not (combined and len(combined.faces) > 0):
            print("ERROR: Wall mesh invalid after combine/process.")
            return trimesh.Trimesh()

        print("  Wall mesh combination successful.")
        return combined

    except Exception as e:
        print(f"ERROR during wall mesh combine/process: {e}")
        import traceback

        traceback.print_exc()
        return trimesh.Trimesh()


# --- 2D Maze STL Function (Should be unaffected, but check imports) ---
# Make sure it still uses the correct extract_wall_bases_2d if called internally
# Or ensure it receives the bases correctly if called from main.py
# The provided main.py calls extract_wall_bases_2d separately for 2D, which is fine.
# No changes needed here unless extract_wall_bases_2d signature changed.

# Assuming extract_wall_bases_2d is imported from geometry
from geometry import extract_wall_bases_2d


def create_2d_maze_stl(
    grid: CircularGrid,
    wall_thickness: float,
    wall_height: float,
    base_height: float,
    output_filename: str,
    wall_bases: Optional[List[Tuple[Tuple[float, float], ...]]] = None,
    opening_params: Optional[
        Tuple[float, float]
    ] = None,  # For outer (equatorial) opening check
    internal_opening_pair: Optional[
        Tuple[str, str]
    ] = None,  # For internal wall removal
):
    """
    Creates an STL file for the 2D maze walls with a solid cylindrical base.
    Uses wall base polygons (optionally pre-extracted) for extrusion.
    Handles openings at the outer boundary (equator) via opening_params.
    Relies on extract_wall_bases_2d to handle pole openings.
    Includes focused debugging for equatorial opening.
    """
    print(f"\n--- Generating 2D Maze STL with Base: {output_filename} ---")
    print(
        f"    Wall H={wall_height:.2f}, Base H={base_height:.2f}, Total H={wall_height + base_height:.2f}"
    )
    print(f"    Using Wall Thickness for Offset: {wall_thickness:.3f}")
    print(
        f"    Equatorial Opening: {opening_params is not None}, Internal Opening: {internal_opening_pair is not None}"
    )

    # --- 1. Get Wall Base Polygons ---
    if wall_bases is None:
        if extract_wall_bases_2d is None:
            print("ERROR: extract_wall_bases_2d function not available.")
            return
        print("  Extracting wall bases internally for 2D STL...")
        # extract_wall_bases_2d handles internal AND pole openings.
        wall_bases = extract_wall_bases_2d(
            grid, wall_thickness, internal_opening_pair=internal_opening_pair
        )
        if not wall_bases:
            print("ERROR: No wall bases found. Cannot create 2D STL.")
            return
        print(f"  Extracted {len(wall_bases)} wall base polygons.")

    # --- 2. Prepare for Outer (Equatorial) Opening Check ---
    opening_start_angle_norm, opening_end_angle_norm = None, None
    if opening_params:
        connection_angle, connection_width_angle = opening_params
        angle_tolerance = 1e-5  # Use same tolerance as 3D check
        half_width = connection_width_angle / 2.0 + angle_tolerance
        opening_start_angle_norm = (connection_angle - half_width + 4 * math.pi) % (
            2 * math.pi
        )
        opening_end_angle_norm = (connection_angle + half_width + 4 * math.pi) % (
            2 * math.pi
        )
        print(
            f"  2D Equatorial Opening Range: [{opening_start_angle_norm:.4f}, {opening_end_angle_norm:.4f}]"
        )

    # --- 3. Generate Wall Meshes from Bases ---
    all_wall_meshes: List[trimesh.Trimesh] = []
    gen_count, skip_equator_opening_count, skip_other_count, err_count = 0, 0, 0, 0
    print(f"  Processing {len(wall_bases)} wall base polygons for extrusion...")

    # Helper for simple Z-extrusion
    def _create_extruded_prism_simple(
        base_verts_2d: Tuple[Tuple[float, float], ...],
        height: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(base_verts_2d) != 4:
            return None  # Expect quads
        try:
            v0, v1, v2, v3 = base_verts_2d
            base_verts = np.array(
                [
                    [v0[0], v0[1], 0.0],
                    [v1[0], v1[1], 0.0],
                    [v2[0], v2[1], 0.0],
                    [v3[0], v3[1], 0.0],
                ]
            )
            top_verts = base_verts + np.array([0.0, 0.0, height])
            verts = np.vstack((base_verts, top_verts))  # 0-3 base, 4-7 top
            faces = np.array(
                [
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],  # Sides
                    [4, 5, 6],
                    [4, 6, 7],  # Top cap
                    [3, 2, 1],
                    [3, 1, 0],  # Bottom cap (reversed)
                ],
                dtype=np.int32,
            )
            return verts, faces
        except Exception as e:
            print(f"WARN: Failed simple extrusion: {e}")
            return None

    # Pole opening check is removed - relying on extract_wall_bases_2d

    for i, base_verts_2d in enumerate(wall_bases):
        should_skip_for_equator_opening = False
        debug_str = f"DEBUG 2D [{i}]:"  # Start debug string

        # --- Check if this segment should be skipped for EQUATORIAL opening ---
        if opening_params:
            is_outer_wall_segment = False
            maze_outer_radius = grid.total_radius
            half_thick = wall_thickness / 2.0
            if maze_outer_radius > const.GEOMETRY_TOLERANCE:
                try:
                    avg_r_2d = np.mean([np.linalg.norm(p) for p in base_verts_2d])
                    if abs(avg_r_2d - maze_outer_radius) < half_thick * 1.1:
                        is_outer_wall_segment = True
                        debug_str += " Outer=T,"
                    # else: debug_str += " Outer=F," # Optional
                except Exception:
                    debug_str += " OuterCheckErr,"
                    pass  # Ignore error, assume not outer

            if is_outer_wall_segment:
                try:
                    v0_2d, v1_2d, v2_2d, v3_2d = base_verts_2d
                    angle1 = math.atan2(v1_2d[1], v1_2d[0])
                    angle2 = math.atan2(v2_2d[1], v2_2d[0])
                    vec1 = np.array([math.cos(angle1), math.sin(angle1)])
                    vec2 = np.array([math.cos(angle2), math.sin(angle2)])
                    avg_vec = normalize(vec1 + vec2)
                    segment_mid_angle = math.atan2(avg_vec[1], avg_vec[0])
                    segment_mid_angle_norm = (segment_mid_angle + 2 * math.pi) % (
                        2 * math.pi
                    )
                    debug_str += f" Angle={segment_mid_angle_norm:.4f},"

                    # Perform the check
                    within_range = is_angle_within(
                        segment_mid_angle_norm,
                        opening_start_angle_norm,
                        opening_end_angle_norm,
                    )
                    debug_str += f" InRange={within_range},"

                    if within_range:
                        should_skip_for_equator_opening = True

                except Exception as e_angle2d:
                    debug_str += f" AngleCheckErr({type(e_angle2d).__name__}),"
                    print(
                        f"WARN: Could not check 2D angle for equator base {i}: {e_angle2d}"
                    )
        # else: debug_str += " EqCheckSkip," # Optional

        # --- Extrude if not skipped for EQUATOR opening ---
        final_result = "Unknown"
        if not should_skip_for_equator_opening:
            try:
                extrusion_result = _create_extruded_prism_simple(
                    base_verts_2d, wall_height
                )
                if extrusion_result is None:
                    final_result = "Failed(ExtrusionHelper)"
                    skip_other_count += 1
                    continue  # Continue loop after setting result
                verts, faces = extrusion_result
                if _is_mesh_degenerate(verts):
                    final_result = "Failed(Degenerate)"
                    skip_other_count += 1
                    continue  # Continue loop after setting result

                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                if mesh and len(mesh.faces) > 0 and np.all(np.isfinite(mesh.vertices)):
                    if np.max(mesh.faces) < len(mesh.vertices):
                        all_wall_meshes.append(mesh)
                        gen_count += 1
                        final_result = "Generated"
                    else:
                        final_result = "Failed(Indices)"
                        skip_other_count += 1  # Index error
                else:
                    final_result = "Failed(BaseValid)"
                    skip_other_count += 1  # Basic check fail

            except Exception as e:
                final_result = f"Failed(Exception: {type(e).__name__})"
                print(f"ERROR: Creating 2D mesh prism for base {i}: {e}")
                err_count += 1
        else:
            final_result = "Skipped(Eq.Opening)"
            skip_equator_opening_count += 1  # Count skipped equator opening segments

        # Print debug info for segments checked for opening or failing
        if (
            opening_params
            and is_outer_wall_segment
            or final_result.startswith("Failed")
        ):
            print(f"{debug_str} Result={final_result}")

    print(
        f"  2D Wall Mesh Summary: Gen={gen_count}, Skip(Eq.Open)={skip_equator_opening_count}, Skip(Other)={skip_other_count}, Err={err_count}"
    )
    if not all_wall_meshes:
        print("ERROR: No valid 2D wall meshes generated. Aborting STL creation.")
        return

    # --- 4. Combine Wall Meshes ---
    print(f"  Combining {len(all_wall_meshes)} 2D wall meshes...")
    try:
        combined_walls_mesh = trimesh.util.concatenate(all_wall_meshes)
        combined_walls_mesh.merge_vertices()
        combined_walls_mesh.fix_normals()
    except Exception as e:
        print(f"ERROR combining/processing 2D wall meshes: {e}")
        return

    # --- 5. Create Base Cylinder Mesh ---
    base_mesh = None
    maze_outer_radius = grid.total_radius
    required_base_radius = (
        maze_outer_radius + wall_thickness / 2.0
    )  # Extend base under outer wall

    if (
        required_base_radius > const.GEOMETRY_TOLERANCE
        and base_height > const.GEOMETRY_TOLERANCE
    ):
        print(f"  Creating cylindrical base with radius: {required_base_radius:.3f}...")
        try:
            base_mesh = trimesh.creation.cylinder(
                radius=required_base_radius,
                height=base_height,
                sections=const.MAZE_2D_CYLINDER_SECTIONS,
            )
            base_mesh.apply_translation([0, 0, -base_height / 2.0])  # Top at z=0
            if not (base_mesh and base_mesh.is_volume):
                base_mesh = None
        except Exception as e:
            print(f"ERROR creating base cylinder: {e}")
            base_mesh = None
    else:
        print(f"  Skipping base cylinder creation.")

    # --- 6. Combine Walls and Base ---
    if base_mesh:
        print("  Combining final 2D walls and base...")
        try:
            if not combined_walls_mesh.is_watertight:
                combined_walls_mesh.fill_holes()
                combined_walls_mesh.fix_normals()
            if not base_mesh.is_watertight:
                base_mesh.fill_holes()
                base_mesh.fix_normals()
            print(f"    Attempting boolean union with engine: {BOOLEAN_ENGINE}...")
            final_mesh = trimesh.boolean.union(
                [combined_walls_mesh, base_mesh], engine=BOOLEAN_ENGINE
            )
        except Exception as bool_e:
            print(
                f"WARN: Boolean union failed ({bool_e}). Falling back to concatenation."
            )
            final_mesh = trimesh.util.concatenate([combined_walls_mesh, base_mesh])
            final_mesh.merge_vertices()
        print(
            f"    Combined Walls & Base: {len(final_mesh.vertices)}V, {len(final_mesh.faces)}F"
        )
    else:
        print("  Proceeding with processed 2D walls mesh only (no valid base).")
        final_mesh = combined_walls_mesh

    # --- 7. Export Final Mesh ---
    if not (final_mesh and len(final_mesh.faces) > 0):
        print("ERROR: Final 2D mesh is invalid or empty. Cannot export.")
        return
    print(f"  Exporting final 2D maze to {output_filename}...")
    try:
        final_mesh.fix_normals()
        export_status = final_mesh.export(output_filename)
        print("  Export complete.")
    except Exception as e:
        print(f"ERROR during final 2D mesh export: {e}")
        import traceback

        traceback.print_exc()
