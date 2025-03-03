import numpy as np
import matplotlib.pyplot as plt
import trimesh
import random
from queue import PriorityQueue

def generate_maze(size):
    """Generates a more complex random maze using Prim's Algorithm with outer walls and marked start/end points."""
    maze = np.ones((size, size))  # 1 = Wall, 0 = Path
    
    # Initialize the maze grid
    start_x, start_y = 1, 1
    maze[start_x, start_y] = 0  # Open the start position
    walls = []
    
    # Add walls of the start cell to the list
    for dx, dy in [(0,2), (2,0), (-2,0), (0,-2)]:
        nx, ny = start_x + dx, start_y + dy
        if 1 <= nx < size-1 and 1 <= ny < size-1:
            walls.append((nx, ny, start_x, start_y))
    
    while walls:
        # Select a random wall
        rand_wall = random.choice(walls)
        x, y, px, py = rand_wall
        
        # Check if the new cell is still a wall
        if maze[x, y] == 1:
            maze[x, y] = 0
            maze[(x + px) // 2, (y + py) // 2] = 0
            
            # Add the new cell's walls to the list
            for dx, dy in [(0,2), (2,0), (-2,0), (0,-2)]:
                nx, ny = x + dx, y + dy
                if 1 <= nx < size-1 and 1 <= ny < size-1 and maze[nx, ny] == 1:
                    walls.append((nx, ny, x, y))
        
        walls.remove(rand_wall)
    
    # Mark start and end points
    maze[1, 1] = 2  # Start point
    maze[size-2, size-2] = 3  # End point
    
    return maze

def visualize_maze(maze):
    """Displays the 2D maze using matplotlib with start/end points."""
    plt.imshow(maze, cmap='gray')
    plt.axis('off')
    plt.scatter(1, 1, c='green', marker='o', label='Start')
    plt.scatter(len(maze)-2, len(maze)-2, c='red', marker='o', label='End')
    plt.legend()
    plt.show()

def maze_to_3d(maze, wall_height=10, wall_thickness=0.1, base_thickness=2):
    """Converts a 2D maze into a 3D model using trimesh with an outer wall and a solid base."""
    size = maze.shape[0]
    vertices = []
    faces = []
    
    # Create the solid base
    base_vertices = []
    for x in range(size + 1):
        for y in range(size + 1):
            base_vertices.append((x * wall_thickness, y * wall_thickness, -base_thickness))
    
    base_faces = []
    for x in range(size):
        for y in range(size):
            base_faces.append(((x * (size + 1) + y),
                               ((x + 1) * (size + 1) + y),
                               ((x + 1) * (size + 1) + (y + 1))))
            base_faces.append(((x * (size + 1) + y),
                               ((x + 1) * (size + 1) + (y + 1)),
                               (x * (size + 1) + (y + 1))))
    
    vertices.extend(base_vertices)
    faces.extend(base_faces)
    
    # Create the maze walls
    for x in range(size):
        for y in range(size):
            if maze[x, y] == 1:  # Add walls only
                base_x, base_y = x * wall_thickness, y * wall_thickness
                vertices.extend([
                    (base_x, base_y, 0),
                    (base_x + wall_thickness, base_y, 0),
                    (base_x, base_y + wall_thickness, 0),
                    (base_x + wall_thickness, base_y + wall_thickness, 0),
                    (base_x, base_y, wall_height),
                    (base_x + wall_thickness, base_y, wall_height),
                    (base_x, base_y + wall_thickness, wall_height),
                    (base_x + wall_thickness, base_y + wall_thickness, wall_height)
                ])
                base = len(vertices) - 8
                faces.extend([
                    (base, base+1, base+5), (base, base+5, base+4),
                    (base+1, base+3, base+7), (base+1, base+7, base+5),
                    (base+3, base+2, base+6), (base+3, base+6, base+7),
                    (base+2, base, base+4), (base+2, base+4, base+6),
                    (base+4, base+5, base+7), (base+4, base+7, base+6)
                ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def export_stl(mesh, filename="maze.stl"):
    """Exports the 3D maze model as an STL file."""
    mesh.export(filename)

def main():
    """Main function to generate, visualize, and export a 3D maze with an outer wall and base."""
    size = 21  # Increased size for more complexity
    maze = generate_maze(size)
    
    print("Maze generated successfully with outer walls, marked start/end points, and a base!")
    visualize_maze(maze)
    
    maze_mesh = maze_to_3d(maze)
    export_stl(maze_mesh)
    print(f"3D Maze exported as 'maze.stl'. Ready for 3D printing!")

if __name__ == "__main__":
    main()
