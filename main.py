# main.py
import numpy as np
import trimesh
import math
import random
import time
import traceback
import os

# Import project modules
import constants as const
from utils import (
    SCIPY_AVAILABLE,
    map_point_to_hemisphere,
    map_point_to_lower_hemisphere,
    is_angle_within,
)
from grid_core import CircularGrid, Cell
from maze_gen import generate_maze
from geometry import extract_wall_bases_2d
from mesh_builder import (
    create_2d_maze_stl,
    create_hemisphere_wall_mesh,  # Will use reverted version
    create_hemisphere_base_shell,  # Will be modified
)
from visualization import (
    find_solution_path,
    visualize_maze_connectivity,
    visualize_maze_links,
    visualize_maze_solution,
    visualize_maze_walls,
)


def run_two_hemisphere_generation():
    start_time = time.time()
    print(f"SciPy Available: {SCIPY_AVAILABLE}")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Configuration ---")
    base_shell_outer_radius_offset = 0.1
    grid_radius = const.DEFAULT_SPHERE_RADIUS  # Radius of maze cell area
    wall_thickness = const.DEFAULT_WALL_THICKNESS_3D
    wall_height = const.DEFAULT_WALL_HEIGHT_3D
    base_thickness = const.HEMI_BASE_THICKNESS
    num_rings = const.DEFAULT_NUM_RINGS
    base_spokes = const.DEFAULT_STARTING_SPOKE_COUNT
    doubling_rings = const.DEFAULT_DOUBLING_RINGS
    center_radius_ratio = 1.0 / (num_rings + 1.0) if num_rings > 0 else 0.0
    center_open_radius = (
        max(0.0, min(grid_radius * center_radius_ratio, grid_radius * 0.9))
        if num_rings > 0
        else 0.0
    )
    wall_subdivisions = 3  # Default subdivisions for walls

    # --- NEW: Calculate Radii based on extending the base ---
    outer_wall_center_r = grid_radius  # Centerline radius of the outermost wall
    # Base shell needs to extend to the outer edge of the outermost wall
    base_shell_outer_radius = outer_wall_center_r + wall_thickness / 2.0
    # Mapping domain should match the extent of the base shell
    mapping_radius = base_shell_outer_radius

    print(f"  Grid Radius (Outer Wall Center): {outer_wall_center_r:.3f}")
    print(f"  Extended Base Shell Outer Radius: {base_shell_outer_radius:.3f}")
    print(f"  Mapping Radius (for map_func): {mapping_radius:.3f}")
    print(
        f"  Wall T/H: {wall_thickness:.3f}/{wall_height:.3f}, Base T: {base_thickness:.3f}"
    )
    print(
        f"  Rings/Hemi: {num_rings}, Spokes: {base_spokes}, Center R: {center_open_radius:.3f}"
    )

    opening_params = None
    internal_opening_A, internal_opening_B = None, None
    base_A, base_B, walls_A, walls_B = None, None, None, None
    grid_A = None

    # === HEMISPHERE A (Upper) - Prep ===
    print("\n--- Preparing Hemisphere A (Upper) ---")
    try:
        # 1. Create Grid A & Generate Maze A (Using original grid_radius)
        grid_A = CircularGrid(
            num_rings, grid_radius, center_open_radius, base_spokes, doubling_rings
        )
        # ... (determine entry/exit, generate maze - unchanged) ...
        grid_A._determine_entry_exit_cells(entry_location="pole", exit_location="pole")
        if not grid_A.entry_cell or not grid_A.exit_cell:
            raise RuntimeError("Failed initial entry/exit A.")
        generate_maze(grid_A)

        # 2. Find Path A, define opening params - (unchanged)
        # ... (find solution path, get opening_params) ...
        solution_path_A = find_solution_path(
            grid_A, grid_A.entry_cell, grid_A.exit_cell
        )
        if not solution_path_A or len(solution_path_A) < 2:
            raise RuntimeError("Failed pathfinding A.")
        pre_connect_cell_A = solution_path_A[-2]
        temp_exit_cell_A = solution_path_A[-1]
        internal_opening_A = (pre_connect_cell_A.id, temp_exit_cell_A.id)
        connection_angle = (
            pre_connect_cell_A.theta_start + pre_connect_cell_A.theta_end
        ) / 2.0
        connection_width_angle = pre_connect_cell_A.width
        if connection_width_angle <= 0:
            raise ValueError("Invalid opening width A.")
        opening_params = (connection_angle, connection_width_angle)
        print(
            f"  Hemi A Internal Opening: {internal_opening_A}, Outer Opening: {opening_params}"
        )

        # 3. Extract Wall Bases A (Unchanged)
        wall_bases_A = extract_wall_bases_2d(
            grid_A, wall_thickness, internal_opening_pair=internal_opening_A
        )
        if not wall_bases_A:
            raise RuntimeError("Failed wall bases A.")

        # 4. Create Extended Base SHELL A (Pass new outer radius, Annulus removed internally)
        print("  Creating Hemi A base shell (Extended Radius)...")
        base_A = create_hemisphere_base_shell(
            outer_shell_radius=base_shell_outer_radius
            + base_shell_outer_radius_offset,  # <-- Pass extended radius
            base_thickness=base_thickness,
            subdivisions=const.HEMI_SPHERE_SUBDIVISIONS,
            hemisphere_type="upper",
        )
        if not base_A:
            raise RuntimeError("Failed base shell A.")

        # 5. Create Walls Mesh A (Treat outer wall like inner, use correct radii)
        print("  Creating Hemi A walls (Standard method for all)...")
        walls_A = create_hemisphere_wall_mesh(
            wall_bases_A,
            mapping_radius,  # <-- Mapping domain up to extended base radius
            base_shell_outer_radius,  # <-- Sphere surface for wall base is extended radius
            wall_height,
            wall_subdivisions,  # <-- Ensure using non-swapped version
            wall_thickness,
            hemisphere_type="upper",
            generate_equator_wall=True,  # <-- Still needed to trigger opening check
            opening_params=opening_params,
            outer_wall_center_r=outer_wall_center_r,  # <-- Still needed for opening check ID
        )
        
        if not walls_A or len(walls_A.faces) == 0:
            raise RuntimeError("Failed wall mesh A.")

    except Exception as e:
        print(f"ERROR Hemi A Prep: {e}")
        traceback.print_exc()
        opening_params = None

    # === HEMISPHERE B (Lower) - Prep ===
    print("\n--- Preparing Hemisphere B (Lower) ---")
    grid_B = None
    try:
        if opening_params is None:
            raise RuntimeError("Cannot proceed without opening params.")
        connection_angle, _ = opening_params

        # 1. Create Grid B & Set Entry/Exit (Using original grid_radius)
        grid_B = CircularGrid(
            num_rings, grid_radius, center_open_radius, base_spokes, doubling_rings
        )
        # ... (determine entry/exit, generate maze - unchanged) ...
        grid_B._determine_entry_exit_cells(entry_location="pole", exit_location="pole")
        if not grid_B.entry_cell:
            raise RuntimeError("Cannot set equator entry B.")
        generate_maze(grid_B)

        # 2. Find Internal Opening B - (unchanged)
        # ... (find internal_opening_B) ...
        first_link_B = next(
            (
                link
                for link in grid_B.entry_cell.links
                if link.ring < grid_B.entry_cell.ring
            ),
            None,
        )
        if not first_link_B:
            first_link_B = next(iter(grid_B.entry_cell.links), None)
        if not first_link_B:
            raise RuntimeError(f"No link for Hemi B entry {grid_B.entry_cell.id}")
        internal_opening_B = (grid_B.entry_cell.id, first_link_B.id)
        print(f"  Hemi B Internal Opening: {internal_opening_B}")

        # 3. Extract Wall Bases B (Unchanged)
        wall_bases_B = extract_wall_bases_2d(
            grid_B, wall_thickness, internal_opening_pair=internal_opening_B
        )
        if not wall_bases_B:
            raise RuntimeError("Failed wall bases B.")

        # 4. Create Extended Base SHELL B (Pass new outer radius)
        print("  Creating Hemi B base shell (Extended Radius)...")
        print("  Creating Hemi B base shell (Original Radius)...")
        base_B = create_hemisphere_base_shell(
            outer_shell_radius=grid_radius
            + base_shell_outer_radius_offset,  # <-- Use original grid_radius
            base_thickness=base_thickness,
            subdivisions=const.HEMI_SPHERE_SUBDIVISIONS,
            hemisphere_type="upper",
        )
        if not base_B:
            raise RuntimeError("Failed base shell B.")

        # 5. Create Walls Mesh B (Internal walls only, standard method)
        # Inside Hemi B prep
        print("  Creating Hemi B walls (Standard method, Original Radius)...")
        walls_B = create_hemisphere_wall_mesh(
            wall_bases_B,
            grid_radius,  # <-- Use original grid_radius for mapping domain
            grid_radius,  # <-- Use original grid_radius for sphere surface
            wall_height,
            wall_subdivisions,
            wall_thickness,
            hemisphere_type="upper",
            generate_equator_wall=True,  # Already False
            # opening_params=None,       # Not needed
            outer_wall_center_r=outer_wall_center_r,  # Still needed? Maybe not, but harmless.
            # Let's keep it for now.
        )
        if not walls_B or len(walls_B.faces) == 0:
            raise RuntimeError("Failed wall mesh B.")

    except Exception as e:
        print(f"ERROR Hemi B Prep: {e}")
        traceback.print_exc()

    # === Combine and Export ===
    # (Combine/Export logic remains the same, maybe update filenames)
    if base_A and walls_A and base_B and walls_B:
        print("\n--- Combining Final Meshes ---")
        try:
            file_A = os.path.join(output_dir, "hemisphere_A.stl")
            file_B = os.path.join(output_dir, "hemisphere_B.stl")
            # Hemi A
            print("  Combining Hemi A Base + Walls A...")
            final_A = trimesh.util.concatenate([base_A, walls_A])
            final_A.merge_vertices()
            final_A.fix_normals()
            if not (final_A and len(final_A.faces) > 0):
                raise RuntimeError("Final Hemi A invalid.")
            print(
                f"  Exporting Hemisphere A ({len(final_A.vertices)}V, {len(final_A.faces)}F) to {file_A}..."
            )
            final_A.export(file_A)
            # Hemi B
            print("  Combining Hemi B Base + Walls B...")
            final_B = trimesh.util.concatenate([base_B, walls_B])
            final_B.merge_vertices()
            final_B.fix_normals()
            if not (final_B and len(final_B.faces) > 0):
                raise RuntimeError("Final Hemi B invalid.")
            print(
                f"  Exporting Hemisphere B ({len(final_B.vertices)}V, {len(final_B.faces)}F) to {file_B}..."
            )
            final_B.export(file_B)
            print("\n--- Two Hemisphere Maze Generation Complete ---")
        except Exception as e_combine:
            print(f"ERROR during final combining/export: {e_combine}")
            traceback.print_exc()
    else:
        print("\n--- Skipping Final Combination (Component generation failed) ---")

    # --- Visualizations ---
    print("\n--- Generating Visualizations ---")
    try:
        # visualize_grid_layout(grid, filename="output/grid_layout.png")
        visualize_maze_links(grid_A, filename="output/maze_links.png")
        visualize_maze_walls(grid_A, filename="output/maze_walls_centerlines.png")
        visualize_maze_connectivity(grid_A, filename="output/maze_connectivity.png")
        visualize_maze_solution(grid_A, filename="output/maze_solution.png")
    except Exception as e:
        print(f"An error occurred during visualization generation: {e}")
        # traceback.print_exc() # Uncomment for detailed viz errors

    # --- Create 2D Flat Visualization STL ---
    print("\n--- Generating 2D Flat STL ---")
    try:
        create_2d_maze_stl(
            grid=grid_A,
            opening_params=opening_params,
            wall_thickness=const.MAZE_2D_WALL_THICKNESS,
            wall_height=const.MAZE_2D_WALL_HEIGHT,
            base_height=const.MAZE_2D_BASE_HEIGHT,
            output_filename="output/maze_2d_flat.stl",
        )
    except Exception as e:
        print(f"An error occurred during 2D STL generation: {e}")
        traceback.print_exc()

    end_time = time.time()
    print(f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    run_two_hemisphere_generation()
