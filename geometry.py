# geometry.py
import numpy as np
import math
from typing import List, Tuple, Set, Optional

# Import from other project modules
# (Ensure these are accessible)
from grid_core import CircularGrid, Cell
import constants as const
from utils import (
    angle_dist,
    is_angle_within,
    angles_overlap,
    normalize,
)  # Ensure math is imported if used directly


def extract_wall_centerlines(
    grid: CircularGrid,
    num_arc_segments_per_cell: int = const.NUM_ARC_SUBDIVISIONS_PER_CELL,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Extracts 2D wall CENTERLINE segments based on the grid structure and links.
    Used primarily for 2D plotting visualizations. Returns line segments as ((x1,y1), (x2,y2)).
    (Implementation from previous responses - assumed unchanged)
    """
    print("--- Extracting Wall Centerlines (for Visualization) ---")
    wall_segments = []
    processed_boundaries: Set[frozenset] = set()  # Keep track of boundaries checked

    # --- Internal Walls ---
    for cell in grid.get_all_cells():
        ring, sector = cell.coords
        r_i, r_o = cell.r_inner, cell.r_outer
        t_s, t_e = cell.theta_start, cell.theta_end  # Note: t_e might be > 2pi

        # Check Radial Walls (boundary with CW neighbour)
        cw_neighbour: Optional[Cell] = cell.neighbours.get(const.DIR_CW)  # Type hint
        if cw_neighbour:
            pair = frozenset([cell.id, cw_neighbour.id])
            if pair not in processed_boundaries:
                processed_boundaries.add(pair)
                if not cell.is_linked(cw_neighbour):
                    spoke_angle = t_e % (2 * math.pi)
                    p1 = (r_i * math.cos(spoke_angle), r_i * math.sin(spoke_angle))
                    p2 = (r_o * math.cos(spoke_angle), r_o * math.sin(spoke_angle))
                    if (
                        np.linalg.norm(np.array(p1) - np.array(p2))
                        > const.GEOMETRY_TOLERANCE
                    ):
                        wall_segments.append((p1, p2))

        # Check Circumferential Walls (boundary with IN neighbours)
        inner_neighbours: List[Cell] = cell.neighbours.get(
            const.DIR_IN, []
        )  # Type hint
        if isinstance(inner_neighbours, list) and inner_neighbours:
            needs_wall_segment = False
            for in_cell in inner_neighbours:
                pair = frozenset([cell.id, in_cell.id])
                if pair not in processed_boundaries:
                    processed_boundaries.add(pair)
                    if not cell.is_linked(in_cell):
                        needs_wall_segment = True
                        break
            if needs_wall_segment:
                r_wall = cell.r_inner
                actual_cell_width = cell.width
                if actual_cell_width < const.GEOMETRY_TOLERANCE:
                    continue
                # Use constant for subdivisions
                num_segments = const.NUM_ARC_SUBDIVISIONS_PER_CELL
                theta_step = actual_cell_width / num_segments
                current_theta_start = cell.theta_start
                for i in range(num_segments):
                    t1 = current_theta_start + i * theta_step
                    t2 = current_theta_start + (i + 1) * theta_step
                    p1 = (r_wall * math.cos(t1), r_wall * math.sin(t1))
                    p2 = (r_wall * math.cos(t2), r_wall * math.sin(t2))
                    if (
                        np.linalg.norm(np.array(p1) - np.array(p2))
                        > const.GEOMETRY_TOLERANCE
                    ):
                        wall_segments.append((p1, p2))

    # print(f"  Found {len(wall_segments)} internal centerline segments.") # Less verbose

    # --- Boundary Walls (Innermost and Outermost Rings) ---
    num_subdivisions_boundary = const.MAZE_2D_CYLINDER_SECTIONS

    # Innermost Boundary
    if grid.center_radius > const.GEOMETRY_TOLERANCE:
        entry_cell = grid.entry_cell
        r_wall_inner = grid.center_radius
        for i in range(num_subdivisions_boundary):
            theta1 = (i / num_subdivisions_boundary) * 2 * math.pi
            theta2 = ((i + 1) / num_subdivisions_boundary) * 2 * math.pi
            segment_mid_angle = ((theta1 + theta2) / 2.0 + 2 * math.pi) % (2 * math.pi)
            is_entry_segment = False
            if (
                entry_cell
                and entry_cell.ring == 0
                and is_angle_within(
                    segment_mid_angle, entry_cell.theta_start, entry_cell.theta_end
                )
            ):
                is_entry_segment = True
            if not is_entry_segment:
                p1 = (r_wall_inner * math.cos(theta1), r_wall_inner * math.sin(theta1))
                p2 = (r_wall_inner * math.cos(theta2), r_wall_inner * math.sin(theta2))
                if (
                    np.linalg.norm(np.array(p1) - np.array(p2))
                    > const.GEOMETRY_TOLERANCE
                ):
                    wall_segments.append((p1, p2))

    # Outermost Boundary - Visualize based on exit cell (might be pole or equator)
    exit_cell = grid.exit_cell
    outer_ring_idx = grid.num_rings - 1
    if outer_ring_idx >= 0:
        r_outer_boundary = grid.total_radius
        for i in range(num_subdivisions_boundary):
            theta1 = (i / num_subdivisions_boundary) * 2 * math.pi
            theta2 = ((i + 1) / num_subdivisions_boundary) * 2 * math.pi
            segment_mid_angle = ((theta1 + theta2) / 2.0 + 2 * math.pi) % (2 * math.pi)
            is_exit_segment = False
            # If exit is on the outer ring, check if this segment is part of it
            if (
                exit_cell
                and exit_cell.ring == outer_ring_idx
                and is_angle_within(
                    segment_mid_angle, exit_cell.theta_start, exit_cell.theta_end
                )
            ):
                is_exit_segment = True
            if not is_exit_segment:  # Draw wall if not part of the designated exit cell
                p1 = (
                    r_outer_boundary * math.cos(theta1),
                    r_outer_boundary * math.sin(theta1),
                )
                p2 = (
                    r_outer_boundary * math.cos(theta2),
                    r_outer_boundary * math.sin(theta2),
                )
                if (
                    np.linalg.norm(np.array(p1) - np.array(p2))
                    > const.GEOMETRY_TOLERANCE
                ):
                    wall_segments.append((p1, p2))

    print(
        f"--- Centerline Extraction Complete: Found {len(wall_segments)} total segments. ---"
    )
    return wall_segments


# --- Wall Base Extraction (Handles ONLY internal opening) ---
def extract_wall_bases_2d(
    grid: CircularGrid,
    wall_thickness: float,
    internal_opening_pair: Optional[Tuple[str, str]] = None,
) -> List[
    Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]
]:
    """
    Extracts 2D wall base polygons offset by thickness.
    Handles an optional *internal* opening specification to remove a single wall segment.
    Generates ALL boundary wall segments (inner and outer). The outer opening
    is handled during 3D mesh generation for Hemi A.
    """
    print(
        f"--- Extracting Wall Base Vertices (Internal Opening: {internal_opening_pair is not None}) ---"
    )
    wall_bases = []
    processed_boundaries: Set[frozenset] = set()
    half_thick = wall_thickness / 2.0
    EXTENSION_AMOUNT = half_thick
    opening_pair_set = (
        frozenset(internal_opening_pair) if internal_opening_pair else None
    )

    # --- 1. Internal Walls ---
    for cell in grid.get_all_cells():
        # (Generate internal radial walls with EXTENSION_AMOUNT, checking internal_opening_pair)
        # (Generate internal circumferential walls WITHOUT tangential extension, checking internal_opening_pair)
        # --- Radial ---
        cw_neighbour = cell.neighbours.get(const.DIR_CW)
        if cw_neighbour:
            pair = frozenset([cell.id, cw_neighbour.id])
            skip_wall = opening_pair_set and pair == opening_pair_set
            if pair not in processed_boundaries:
                processed_boundaries.add(pair)
                if not cell.is_linked(cw_neighbour) and not skip_wall:
                    ring, sector = cell.coords
                    r_i, r_o = cell.r_inner, cell.r_outer
                    t_e = cell.theta_end
                    spoke_angle = t_e
                    p1_center = np.array(
                        [r_i * math.cos(spoke_angle), r_i * math.sin(spoke_angle)]
                    )
                    p2_center = np.array(
                        [r_o * math.cos(spoke_angle), r_o * math.sin(spoke_angle)]
                    )
                    direction = p2_center - p1_center
                    dist = np.linalg.norm(direction)
                    if dist > const.GEOMETRY_TOLERANCE:
                        dir_norm = direction / dist
                        perp_dir = np.array([-dir_norm[1], dir_norm[0]])
                        v0_init = p1_center - perp_dir * half_thick
                        v1_init = p2_center - perp_dir * half_thick
                        v2_init = p2_center + perp_dir * half_thick
                        v3_init = p1_center + perp_dir * half_thick
                        v0 = v0_init - dir_norm * EXTENSION_AMOUNT
                        v1 = v1_init + dir_norm * EXTENSION_AMOUNT
                        v2 = v2_init + dir_norm * EXTENSION_AMOUNT
                        v3 = v3_init - dir_norm * EXTENSION_AMOUNT
                        if (
                            np.linalg.norm(v1 - v0) > const.GEOMETRY_TOLERANCE
                            and np.linalg.norm(v2 - v1) > const.GEOMETRY_TOLERANCE
                        ):
                            wall_bases.append(
                                (
                                    (v0[0], v0[1]),
                                    (v1[0], v1[1]),
                                    (v2[0], v2[1]),
                                    (v3[0], v3[1]),
                                )
                            )
                elif skip_wall:
                    print(
                        f"DEBUG: Skipping internal radial wall between {cell.id} and {cw_neighbour.id}"
                    )
        # --- Circumferential ---
        inner_neighbours = cell.neighbours.get(const.DIR_IN, [])
        unlinked_inner_neighbours_to_draw = []
        if isinstance(inner_neighbours, list):
            for in_cell in inner_neighbours:
                pair = frozenset([cell.id, in_cell.id])
                skip_wall = opening_pair_set and pair == opening_pair_set
                if pair not in processed_boundaries:
                    processed_boundaries.add(pair)
                    if not cell.is_linked(in_cell) and not skip_wall:
                        unlinked_inner_neighbours_to_draw.append(in_cell)
                elif skip_wall:
                    print(
                        f"DEBUG: Skipping internal circum. wall between {cell.id} and {in_cell.id}"
                    )
        if unlinked_inner_neighbours_to_draw:
            r_wall_center = cell.r_inner
            cell_width_angle = cell.width
            if cell_width_angle < const.GEOMETRY_TOLERANCE:
                continue
            num_subdiv = const.NUM_ARC_SUBDIVISIONS_PER_CELL
            theta_step = cell_width_angle / num_subdiv
            for i in range(num_subdiv):
                t1 = cell.theta_start + i * theta_step
                t2 = cell.theta_start + (i + 1) * theta_step
                mid_angle_seg = ((t1 + t2) / 2.0 + 2 * math.pi) % (2 * math.pi)
                needs_wall_here = any(
                    is_angle_within(
                        mid_angle_seg, unlinked_n.theta_start, unlinked_n.theta_end
                    )
                    for unlinked_n in unlinked_inner_neighbours_to_draw
                )
                if needs_wall_here and abs(t2 - t1) > const.GEOMETRY_TOLERANCE:
                    R_inner = max(0, r_wall_center - half_thick)
                    R_outer = r_wall_center + half_thick
                    v0 = np.array([R_inner * math.cos(t1), R_inner * math.sin(t1)])
                    v1 = np.array([R_outer * math.cos(t1), R_outer * math.sin(t1)])
                    v2 = np.array([R_outer * math.cos(t2), R_outer * math.sin(t2)])
                    v3 = np.array([R_inner * math.cos(t2), R_inner * math.sin(t2)])
                    if (
                        np.linalg.norm(v1 - v0) > const.GEOMETRY_TOLERANCE
                        and np.linalg.norm(v2 - v1) > const.GEOMETRY_TOLERANCE
                        and np.linalg.norm(v3 - v2) > const.GEOMETRY_TOLERANCE
                        and np.linalg.norm(v0 - v3) > const.GEOMETRY_TOLERANCE
                    ):
                        wall_bases.append(
                            (
                                (v0[0], v0[1]),
                                (v1[0], v1[1]),
                                (v2[0], v2[1]),
                                (v3[0], v3[1]),
                            )
                        )

    # --- 2. Boundary Walls ---
    num_subdivisions_boundary = const.MAZE_2D_CYLINDER_SECTIONS

    # --- Innermost boundary wall (Skip entry segment) ---
    if grid.center_radius > const.GEOMETRY_TOLERANCE:
        entry_cell = grid.entry_cell
        r_wall_center = grid.center_radius
        # ... (Generate inner boundary v0,v1,v2,v3 and append if not entry_segment) ...
        for i in range(num_subdivisions_boundary):
            theta1 = (i / num_subdivisions_boundary) * 2 * math.pi
            theta2 = ((i + 1) / num_subdivisions_boundary) * 2 * math.pi
            if abs(theta2 - theta1) < const.GEOMETRY_TOLERANCE:
                continue
            segment_mid_angle = ((theta1 + theta2) / 2.0 + 2 * math.pi) % (2 * math.pi)
            is_entry_segment = (
                entry_cell
                and entry_cell.ring == 0
                and is_angle_within(
                    segment_mid_angle, entry_cell.theta_start, entry_cell.theta_end
                )
            )
            if not is_entry_segment:
                R_inner = max(0, r_wall_center - half_thick)
                R_outer = r_wall_center + half_thick
                v0 = np.array([R_inner * math.cos(theta1), R_inner * math.sin(theta1)])
                v1 = np.array([R_outer * math.cos(theta1), R_outer * math.sin(theta1)])
                v2 = np.array([R_outer * math.cos(theta2), R_outer * math.sin(theta2)])
                v3 = np.array([R_inner * math.cos(theta2), R_inner * math.sin(theta2)])
                if (
                    np.linalg.norm(v1 - v0) > const.GEOMETRY_TOLERANCE
                    and np.linalg.norm(v2 - v1) > const.GEOMETRY_TOLERANCE
                    and np.linalg.norm(v3 - v2) > const.GEOMETRY_TOLERANCE
                    and np.linalg.norm(v0 - v3) > const.GEOMETRY_TOLERANCE
                ):
                    wall_bases.append(
                        ((v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1]), (v3[0], v3[1]))
                    )

    # --- Outermost boundary wall (GENERATE FULL WALL - Opening handled in 3D step) ---
    outer_ring_idx = grid.num_rings - 1
    if outer_ring_idx >= 0:
        r_wall_center = grid.total_radius
        print(
            f"DEBUG: Generating FULL outermost boundary wall bases (Ring {outer_ring_idx})"
        )
        outer_wall_segments_generated = 0
        for i in range(num_subdivisions_boundary):
            theta1 = (i / num_subdivisions_boundary) * 2 * math.pi
            theta2 = ((i + 1) / num_subdivisions_boundary) * 2 * math.pi
            if abs(theta2 - theta1) < const.GEOMETRY_TOLERANCE:
                continue
            # Generate ALL segments
            R_inner = max(0, r_wall_center - half_thick)
            R_outer = r_wall_center + half_thick
            v0 = np.array([R_inner * math.cos(theta1), R_inner * math.sin(theta1)])
            v1 = np.array([R_outer * math.cos(theta1), R_outer * math.sin(theta1)])
            v2 = np.array([R_outer * math.cos(theta2), R_outer * math.sin(theta2)])
            v3 = np.array([R_inner * math.cos(theta2), R_inner * math.sin(theta2)])
            if (
                np.linalg.norm(v1 - v0) > const.GEOMETRY_TOLERANCE
                and np.linalg.norm(v2 - v1) > const.GEOMETRY_TOLERANCE
                and np.linalg.norm(v3 - v2) > const.GEOMETRY_TOLERANCE
                and np.linalg.norm(v0 - v3) > const.GEOMETRY_TOLERANCE
            ):
                wall_bases.append(
                    ((v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1]), (v3[0], v3[1]))
                )
                outer_wall_segments_generated += 1
        print(
            f"DEBUG: Generated {outer_wall_segments_generated} base polygons for the full outermost boundary."
        )

    print(
        f"--- Wall Base Extraction Complete: Found {len(wall_bases)} base polygons total ---"
    )
    return wall_bases
