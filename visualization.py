# visualization.py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import math
from collections import deque
from typing import List, Tuple, Optional, Dict

# Import from other project modules
from grid_core import CircularGrid, Cell
from geometry import extract_wall_centerlines # Uses centerlines for viz
import constants as const

# --- Pathfinding (Often used with visualization) ---
def find_solution_path(
    grid: CircularGrid, start_cell: Optional[Cell], end_cell: Optional[Cell]
) -> Optional[List[Cell]]:
    """Finds the shortest path between two cells using Breadth-First Search on links."""
    print(f"--- Finding path from {start_cell.id if start_cell else 'None'} to {end_cell.id if end_cell else 'None'} ---")
    if not (start_cell and end_cell and start_cell.id in grid.cells and end_cell.id in grid.cells):
        print("ERROR: Invalid start or end cell provided.")
        return None

    # BFS initialization
    queue = deque([start_cell])
    # Keep track of predecessors to reconstruct the path
    predecessor: Dict[str, Optional[str]] = {start_cell.id: None}
    # visited set equivalent to keys in predecessor
    path_found = False

    while queue:
        current_cell = queue.popleft()

        if current_cell.id == end_cell.id:
            path_found = True
            print("  Path found!")
            break

        # Explore linked neighbours
        for neighbour_cell in current_cell.links:
            if neighbour_cell and neighbour_cell.id not in predecessor:
                predecessor[neighbour_cell.id] = current_cell.id
                queue.append(neighbour_cell)

    if not path_found:
        print("  Path not found!")
        return None

    # Reconstruct path
    path_cells: List[Cell] = []
    curr_id: Optional[str] = end_cell.id
    while curr_id is not None:
        cell = grid.cells.get(curr_id)
        if cell:
             path_cells.append(cell)
        else:
             print(f"ERROR: Path reconstruction failed, cell ID {curr_id} not found.")
             return None # Error in path reconstruction
        curr_id = predecessor.get(curr_id)

    path_cells.reverse() # Reverse to get path from start to end

    # Final validation
    if not path_cells or path_cells[0].id != start_cell.id:
        print("ERROR: Path reconstruction failed (start cell mismatch).")
        return None

    print(f"  Path length: {len(path_cells)} cells.")
    return path_cells


# --- Visualization Helpers ---
def _setup_polar_plot(grid: CircularGrid) -> Tuple[plt.Figure, plt.Axes]:
    """Creates and configures a polar plot axis."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.set_rorigin(-grid.max_radius * 0.1) # Offset origin slightly for visibility
    ax.set_rmax(grid.max_radius * 1.05)
    ax.set_rticks(grid.ring_radii) # Show ring boundaries
    # ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False)) # Example spokes
    ax.set_xticklabels([]) # Hide angular labels
    ax.set_yticklabels([]) # Hide radial labels
    ax.grid(True, linestyle='--', alpha=0.5) # Add faint grid
    return fig, ax

def _draw_cell_outlines(ax: plt.Axes, grid: CircularGrid):
    """Draws the basic grid structure."""
    for cell in grid.get_all_cells():
        r_i, r_o = cell.r_inner, cell.r_outer
        t_s, t_e = cell.theta_start, cell.theta_end
        # Draw arcs using multiple points for smoothness
        num_arc_points = 10 # Adjust for smoother arcs if needed
        arc_thetas = np.linspace(t_s, t_e, num_arc_points)
        ax.plot(arc_thetas, np.full_like(arc_thetas, r_i), color=const.VIS_CELL_OUTLINE_COLOR, lw=const.VIS_CELL_OUTLINE_LW)
        ax.plot(arc_thetas, np.full_like(arc_thetas, r_o), color=const.VIS_CELL_OUTLINE_COLOR, lw=const.VIS_CELL_OUTLINE_LW)
        # Draw radial spokes
        ax.plot([t_s, t_s], [r_i, r_o], color=const.VIS_CELL_OUTLINE_COLOR, lw=const.VIS_CELL_OUTLINE_LW)
        # The end spoke t_e is drawn by the next cell's t_s, except for the last sector which wraps.
        # We could explicitly draw the last spoke if needed, but outline usually sufficient.


def _draw_links(ax: plt.Axes, grid: CircularGrid):
    """Draws lines connecting linked cell centers."""
    drawn_pairs = set()
    link_count = 0
    for cell in grid.get_all_cells():
        for linked_cell in cell.links:
            pair = frozenset([cell.id, linked_cell.id])
            if pair not in drawn_pairs:
                ax.plot([cell.center_theta, linked_cell.center_theta],
                        [cell.center_r, linked_cell.center_r],
                        const.VIS_LINK_LINE_STYLE,
                        lw=const.VIS_LINK_LINE_LW,
                        alpha=const.VIS_LINK_LINE_ALPHA,
                        )
                drawn_pairs.add(pair)
                link_count += 1
    return link_count

def _draw_wall_centerlines(ax: plt.Axes, grid: CircularGrid):
    """Draws wall centerlines based on unlinked neighbours."""
    # Use the dedicated function for consistency
    wall_segments = extract_wall_centerlines(grid)
    print(f"  Visualizing {len(wall_segments)} wall centerlines...")
    for segment in wall_segments:
        p1_cart, p2_cart = segment
        x1, y1 = p1_cart
        x2, y2 = p2_cart
        # Convert Cartesian segment points back to polar for plotting
        theta1, r1 = math.atan2(y1, x1), math.sqrt(x1**2 + y1**2)
        theta2, r2 = math.atan2(y2, x2), math.sqrt(x2**2 + y2**2)
        # Plotting requires handling angle wrapping if the line crosses the 0/2pi boundary
        # A simple way is to plot segment by segment, matplotlib handles short lines okay
        ax.plot([theta1, theta2], [r1, r2],
                const.VIS_WALL_LINE_STYLE,
                lw=const.VIS_WALL_LINE_LW,
                alpha=const.VIS_WALL_LINE_ALPHA) # Use general wall alpha


def _draw_entry_exit(ax: plt.Axes, grid: CircularGrid, **kwargs):
    """Marks entry and exit cells."""
    if grid.entry_cell:
        ax.plot(grid.entry_cell.center_theta, grid.entry_cell.center_r,
                const.VIS_ENTRY_MARKER,
                markersize=kwargs.get('entry_msize', const.VIS_ENTRY_MARKER_SIZE),
                alpha=kwargs.get('alpha', const.VIS_ENTRY_MARKER_ALPHA),
                mfc=kwargs.get('entry_mfc', 'lime'), # Default colors
                mec=kwargs.get('entry_mec', 'black'),
                label="Entry")
    if grid.exit_cell:
         ax.plot(grid.exit_cell.center_theta, grid.exit_cell.center_r,
                 const.VIS_EXIT_MARKER,
                 markersize=kwargs.get('exit_msize', const.VIS_EXIT_MARKER_SIZE),
                 alpha=kwargs.get('alpha', const.VIS_EXIT_MARKER_ALPHA),
                 mfc=kwargs.get('exit_mfc', 'red'), # Default colors
                 mec=kwargs.get('exit_mec', 'black'),
                 label="Exit")


# --- Main Visualization Functions ---

def visualize_grid_layout(grid: CircularGrid, filename="grid_layout.png"):
    """Visualizes the basic grid cell structure."""
    print(f"--- Generating Grid Layout Visualization: {filename} ---")
    try:
        fig, ax = _setup_polar_plot(grid)
        _draw_cell_outlines(ax, grid)
        _draw_entry_exit(ax, grid)
        ax.set_title("Grid Cell Layout")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Layout visualization saved to {filename}")
    except ImportError:
        print("ERROR: Matplotlib not found, cannot generate visualization.")
    except Exception as e:
        print(f"ERROR during visualization: {e}")


def visualize_maze_links(grid: CircularGrid, filename="maze_links.png"):
    """Visualizes the generated maze links (passages) between cells."""
    print(f"--- Generating Maze Links Visualization: {filename} ---")
    try:
        fig, ax = _setup_polar_plot(grid)
        # Optionally draw faint outlines
        # _draw_cell_outlines(ax, grid)
        link_count = _draw_links(ax, grid)
        _draw_entry_exit(ax, grid, entry_msize=const.VIS_SOLUTION_ENTRY_MARKER_SIZE,
                         exit_msize=const.VIS_SOLUTION_EXIT_MARKER_SIZE) # Make markers larger
        ax.set_title(f"Maze Links ({link_count} Passages)")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Links visualization saved to {filename}")
    except ImportError:
        print("ERROR: Matplotlib not found, cannot generate visualization.")
    except Exception as e:
        print(f"ERROR during visualization: {e}")


def visualize_maze_walls(grid: CircularGrid, filename="maze_walls.png"):
    """Visualizes the maze walls (using centerlines)."""
    print(f"--- Generating Maze Walls Visualization: {filename} ---")
    try:
        fig, ax = _setup_polar_plot(grid)
        print("  Extracting wall centerlines for walls plot...")
        _draw_wall_centerlines(ax, grid) # Calls extract_wall_centerlines internally
        _draw_entry_exit(ax, grid)
        ax.set_title("Maze Walls (Centerlines)")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Walls visualization saved to {filename}")
    except ImportError:
        print("ERROR: Matplotlib not found, cannot generate visualization.")
    except Exception as e:
        print(f"ERROR during visualization: {e}")

def visualize_maze_solution(grid: CircularGrid, filename="maze_solution.png"):
    """Finds and visualizes the solution path from entry to exit."""
    print(f"--- Generating Maze Solution Visualization: {filename} ---")
    solution_path = find_solution_path(grid, grid.entry_cell, grid.exit_cell)
    if not solution_path:
        print("  Could not find solution path, cannot visualize.")
        return

    try:
        fig, ax = _setup_polar_plot(grid)
        # Draw walls faintly in the background
        print("  Extracting wall centerlines for solution plot...")
        _draw_wall_centerlines(ax, grid)

        # Draw Solution Path
        print(f"  Visualizing solution path ({len(solution_path)} cells)...")
        path_thetas = [cell.center_theta for cell in solution_path]
        path_rs = [cell.center_r for cell in solution_path]
        ax.plot(path_thetas, path_rs,
                const.VIS_SOLUTION_LINE_STYLE,
                lw=const.VIS_SOLUTION_LINE_LW,
                alpha=const.VIS_SOLUTION_LINE_ALPHA)

        # Mark Start/End prominently
        _draw_entry_exit(ax, grid,
                         entry_msize=const.VIS_SOLUTION_ENTRY_MARKER_SIZE,
                         exit_msize=const.VIS_SOLUTION_EXIT_MARKER_SIZE,
                         entry_mfc=const.VIS_SOLUTION_ENTRY_MFC,
                         entry_mec=const.VIS_SOLUTION_ENTRY_MEC,
                         exit_mfc=const.VIS_SOLUTION_EXIT_MFC,
                         exit_mec=const.VIS_SOLUTION_EXIT_MEC)

        ax.set_title("Maze Solution Path")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Solution visualization saved to {filename}")
    except ImportError:
        print("ERROR: Matplotlib not found, cannot generate visualization.")
    except Exception as e:
        print(f"ERROR during visualization: {e}")


def visualize_maze_connectivity(grid: CircularGrid, filename="maze_connectivity.png"):
    """Visualizes cell connectivity and distance from the start cell."""
    print(f"--- Generating Connectivity Visualization: {filename} ---")
    if not grid.cells:
        print("Grid empty, cannot visualize connectivity.")
        return

    start_node = grid.entry_cell or grid.random_cell()
    if not start_node:
        print("Cannot find start node for connectivity check.")
        return
    if not grid.entry_cell:
        print(f"Warning: No entry cell defined. Starting connectivity check from random cell {start_node.id}")

    # Perform BFS to find distances from start node via links
    distances: Dict[str, int] = {cid: -1 for cid in grid.cells} # -1 means not visited
    distances[start_node.id] = 0
    queue = deque([start_node])
    max_distance = 0
    visited_count = 1

    while queue:
        current_cell = queue.popleft()
        current_dist = distances.get(current_cell.id, -2) # Should always be >= 0 here
        if current_dist < 0: continue # Should not happen

        max_distance = max(max_distance, current_dist)

        for neighbour in current_cell.links:
            if neighbour and distances.get(neighbour.id, -2) == -1: # If neighbour not visited
                distances[neighbour.id] = current_dist + 1
                visited_count += 1
                queue.append(neighbour)

    print(f"  Connectivity check visited {visited_count}/{grid.size()} cells.")
    if visited_count < grid.size():
        print("  WARNING: Not all cells are reachable from the start node!")

    # Plotting
    try:
        fig, ax = _setup_polar_plot(grid)
        ax.set_axis_off() # Turn off grid/labels for fill plot
        cmap = cm.viridis # Colormap for distance
        norm = mcolors.Normalize(vmin=0, vmax=max(1, max_distance)) # Normalize distances for color mapping
        unreachable_color = const.VIS_CONN_UNREACHABLE_COLOR

        print("  Coloring cells based on distance...")
        for cell_id, cell in grid.cells.items():
            distance = distances.get(cell_id, -1)
            color = unreachable_color if distance == -1 else cmap(norm(distance))

            # Draw filled cell sector
            r_i, r_o = cell.r_inner, cell.r_outer
            t_s, t_e = cell.theta_start, cell.theta_end
            # Create vertices for the polygon patch
            num_arc_points = 10 # More points for smoother fill
            thetas_outer = np.linspace(t_s, t_e, num_arc_points)
            thetas_inner = np.linspace(t_e, t_s, num_arc_points) # Reverse for correct polygon order
            verts_r = np.concatenate([r_o * np.ones_like(thetas_outer), r_i * np.ones_like(thetas_inner)])
            verts_theta = np.concatenate([thetas_outer, thetas_inner])
            ax.fill(verts_theta, verts_r, color=color, edgecolor="dimgrey", linewidth=0.1, alpha=0.95)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # Needed for colorbar
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.08)
        cbar.set_label(f"Distance from Start Cell ({start_node.id})")
        if visited_count < grid.size():
            cbar.ax.set_title("Grey = Unreachable Cells", fontsize=8, color="red")

        ax.set_title(f"Maze Connectivity ({visited_count}/{grid.size()} Reachable)")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Connectivity visualization saved to {filename}")

    except ImportError:
        print("ERROR: Matplotlib not found, cannot generate visualization.")
    except Exception as e:
        print(f"ERROR during visualization: {e}")