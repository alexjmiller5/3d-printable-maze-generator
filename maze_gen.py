# maze_gen.py
import random
from typing import List

# Import from other project modules
from grid_core import Cell, CircularGrid

def generate_maze(grid: CircularGrid):
    """
    Generates maze passages within the grid using the Recursive Backtracking algorithm.
    Modifies the 'links' attribute of the cells in the grid.
    """
    print("--- Starting Maze Generation (Recursive Backtracking) ---")
    if grid.size() == 0:
        print("ERROR: Grid has no cells, cannot generate maze.")
        return

    # Reset previous maze state (if any)
    for cell in grid.get_all_cells():
        cell.unmark_visited()
        cell.links = set()

    # Initialize stack and starting cell
    stack: List[Cell] = []
    start_cell = grid.entry_cell if grid.entry_cell else grid.random_cell()
    if not start_cell:
        print("ERROR: Cannot determine a starting cell for maze generation.")
        return

    print(f"  Starting maze generation at cell: {start_cell.id}")
    start_cell.mark_visited()
    stack.append(start_cell)
    visited_count = 1

    # Main loop
    while stack:
        current_cell = stack[-1]
        unvisited_neighbours = current_cell.get_unvisited_neighbours()

        if unvisited_neighbours:
            # Choose a random unvisited neighbour
            next_cell = random.choice(unvisited_neighbours)
            # Link the current cell to the chosen neighbour
            current_cell.link(next_cell)
            # Mark the neighbour as visited and push it onto the stack
            next_cell.mark_visited()
            stack.append(next_cell)
            visited_count += 1
        else:
            # No unvisited neighbours, backtrack
            stack.pop()

    print(f"--- Maze Generation Complete: Linked {visited_count}/{grid.size()} cells. ---")

    # Sanity check: Ensure all cells were visited
    if visited_count < grid.size():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: MAZE GENERATION FAILED TO VISIT ALL CELLS! Visited {visited_count}/{grid.size()}.")
        print("This indicates an issue with neighbour linking or the grid structure.")
        unvisited_example = next((c for c in grid.get_all_cells() if not c.is_visited()), None)
        if unvisited_example:
            print(f"Example unvisited cell: {unvisited_example.id}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")