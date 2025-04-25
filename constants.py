# --- General Geometry ---
DEFAULT_SPHERE_RADIUS = 10.0

# --- Hemisphere Generation ---
DEFAULT_WALL_THICKNESS_3D = 0.25  # Thickness for hemisphere walls
DEFAULT_WALL_HEIGHT_3D = 2.0  # Height for hemisphere walls
HEMI_SPHERE_SUBDIVISIONS = 300  # Controls smoothness of the base hemisphere surface.
HEMI_BASE_THICKNESS = 1.0  # Or your desired thickness, e.g., 0.5, 1.0, etc.

# --- Grid Structure ---
DEFAULT_NUM_RINGS = 15  # Number of concentric rings (e.g., 9 rings: 0-8)
DEFAULT_STARTING_SPOKE_COUNT = 6
# Example: Rings where the number of spokes doubles relative to the inner ring
DEFAULT_DOUBLING_RINGS = None  # Let CircularGrid calculate defaults if None
# Example: specific_doubling_rings = [1, 3, 7]

# --- 2D STL Visualization ---
MAZE_2D_WALL_THICKNESS = 0.35
MAZE_2D_WALL_HEIGHT = 1.5
MAZE_2D_BASE_HEIGHT = MAZE_2D_WALL_HEIGHT / 3.0  # Configurable base height
MAZE_2D_CYLINDER_SECTIONS = 128  # Smoothness for the cylindrical base edge
NUM_ARC_SUBDIVISIONS_PER_CELL = 5  # How many trapezoids approximate each arc wall

# --- Cell Directions ---
DIR_CW = "CW"  # Clockwise
DIR_CCW = "CCW"  # Counter-Clockwise
DIR_IN = "IN"  # Inward (towards center)
DIR_OUT = "OUT"  # Outward (towards edge)

# --- Tolerances ---
GEOMETRY_TOLERANCE = 1e-9  # For floating point comparisons
ANGLE_OVERLAP_TOLERANCE = 1e-9
MESH_VERTEX_DISTANCE_TOLERANCE_SQ = (
    1e-12  # Squared tolerance for merging/checking degenerate
)

# --- Visualization ---
VIS_CONN_UNREACHABLE_COLOR = "lightgrey"
VIS_CELL_OUTLINE_COLOR = "lightgrey"
VIS_CELL_OUTLINE_LW = 0.5
VIS_NEIGHBOR_LINE_STYLE = "b--"
VIS_NEIGHBOR_LINE_LW = 0.4
VIS_NEIGHBOR_LINE_ALPHA = 0.6
VIS_LINK_LINE_STYLE = "g-"
VIS_LINK_LINE_LW = 1.0
VIS_LINK_LINE_ALPHA = 0.7
VIS_WALL_LINE_STYLE = "k-"
VIS_WALL_LINE_LW = 1.5
VIS_WALL_LINE_ALPHA = 0.7  # Used in solution plot walls
VIS_SOLUTION_LINE_STYLE = "r-"
VIS_SOLUTION_LINE_LW = 2.0
VIS_SOLUTION_LINE_ALPHA = 0.9
VIS_ENTRY_MARKER = "go"
VIS_ENTRY_MARKER_SIZE = 6
VIS_ENTRY_MARKER_ALPHA = 0.8
VIS_EXIT_MARKER = "ro"
VIS_EXIT_MARKER_SIZE = 6
VIS_EXIT_MARKER_ALPHA = 0.8
VIS_SOLUTION_ENTRY_MARKER_SIZE = 8
VIS_SOLUTION_ENTRY_MFC = "lime"
VIS_SOLUTION_ENTRY_MEC = "black"
VIS_SOLUTION_EXIT_MARKER_SIZE = 8
VIS_SOLUTION_EXIT_MFC = "red"
VIS_SOLUTION_EXIT_MEC = "black"
