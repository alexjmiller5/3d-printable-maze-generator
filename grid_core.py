# grid_core.py
import numpy as np
import math
import random
from typing import List, Tuple, Optional, Set, Dict, Iterator, Union

# Import from other project modules
import constants as const
from utils import angles_overlap, is_angle_within


class Cell:
    """Represents a single cell in the fixed-spoke circular grid."""

    def __init__(self, ring: int, sector: int):
        self.ring = ring
        self.sector = sector
        self.id = f"{ring},{sector}"
        self.coords = (ring, sector)  # Store as tuple for convenience
        self.r_inner: float = 0.0
        self.r_outer: float = 0.0
        self.theta_start: float = 0.0
        self.theta_end: float = 0.0  # NOTE: Can be > 2pi if it wraps past 0
        self.width: float = 0.0  # Angular width (theta_end - theta_start, handles wrap)
        self.height: float = 0.0  # Radial height (r_outer - r_inner)
        self.center_r: float = 0.0
        self.center_theta: float = 0.0
        self.center_x: float = 0.0
        self.center_y: float = 0.0

        # Neighbours: CW/CCW store single Cell, IN/OUT store List[Cell]
        self.neighbours: Dict[str, Union[Optional["Cell"], List["Cell"]]] = {
            const.DIR_CW: None,
            const.DIR_CCW: None,
            const.DIR_IN: [],
            const.DIR_OUT: [],
        }
        self.links: Set["Cell"] = set()  # Cells connected by passages
        self._visited: bool = False  # Used by maze generation algorithms

    def link(self, other_cell: Optional["Cell"]):
        """Creates a bidirectional link between this cell and another."""
        if other_cell and other_cell not in self.links:
            self.links.add(other_cell)
            other_cell.links.add(self)

    def is_linked(self, other_cell: Optional["Cell"]) -> bool:
        """Checks if this cell is linked to another cell."""
        return other_cell in self.links

    def mark_visited(self):
        """Marks the cell as visited (for algorithms)."""
        self._visited = True

    def unmark_visited(self):
        """Marks the cell as not visited."""
        self._visited = False

    def is_visited(self) -> bool:
        """Checks if the cell has been marked as visited."""
        return self._visited

    def get_unvisited_neighbours(self) -> List["Cell"]:
        """Gets list of valid, unvisited neighbours reachable from this cell."""
        potential_neighbours = []
        # Add neighbours from all directions, handling None or lists
        cw = self.neighbours[const.DIR_CW]
        ccw = self.neighbours[const.DIR_CCW]
        _in = self.neighbours[const.DIR_IN]
        out = self.neighbours[const.DIR_OUT]

        if cw:
            potential_neighbours.append(cw)
        if ccw:
            potential_neighbours.append(ccw)
        if isinstance(_in, list):
            potential_neighbours.extend(_in)
        if isinstance(out, list):
            potential_neighbours.extend(out)

        # Filter for uniqueness and visited status, ensure reciprocal neighbourhood
        unique_unvisited: Dict[str, Cell] = {}
        for neighbour in potential_neighbours:
            if neighbour and not neighbour.is_visited():
                # Check if the neighbour also considers this cell a neighbour
                is_reachable = False
                n_neigh = neighbour.neighbours
                if (
                    n_neigh.get(const.DIR_CW) == self
                    or n_neigh.get(const.DIR_CCW) == self
                ):
                    is_reachable = True
                elif isinstance(
                    n_neigh.get(const.DIR_IN), list
                ) and self in n_neigh.get(const.DIR_IN, []):
                    is_reachable = True
                elif isinstance(
                    n_neigh.get(const.DIR_OUT), list
                ) and self in n_neigh.get(const.DIR_OUT, []):
                    is_reachable = True

                if is_reachable:
                    unique_unvisited[neighbour.id] = (
                        neighbour  # Use dict to ensure uniqueness
                    )

        return list(unique_unvisited.values())

    def __repr__(self) -> str:
        return f"Cell({self.id})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Cell) and self.id == other.id


class CircularGrid:
    """
    Represents the circular grid structure with fixed radial spokes,
    potentially varying sector counts per ring, and an open center.
    """

    def __init__(
        self,
        num_rings: int,
        total_radius: float,
        center_radius: float,
        base_spokes: int = const.DEFAULT_STARTING_SPOKE_COUNT,
        doubling_rings: Optional[List[int]] = const.DEFAULT_DOUBLING_RINGS,
    ):
        if num_rings <= 0:
            raise ValueError("Number of rings must be positive.")
        if not (0 <= center_radius < total_radius - const.GEOMETRY_TOLERANCE):
            raise ValueError(
                f"Center radius ({center_radius}) must be non-negative and strictly less than total radius ({total_radius})."
            )
        if base_spokes <= 0:
            raise ValueError("Base spokes count must be positive.")

        self.num_rings = num_rings
        self.center_radius = max(0.0, center_radius)  # Ensure non-negative
        self.total_radius = total_radius
        self.max_radius = total_radius  # Alias for clarity in some contexts

        self.cells: Dict[str, Cell] = {}
        self.ring_radii: List[float] = (
            []
        )  # Radii defining ring boundaries (len = num_rings + 1)
        self.num_spokes_per_ring: List[int] = (
            []
        )  # Number of sectors in each ring (len = num_rings)
        self.spoke_angles_per_ring: List[np.ndarray] = (
            []
        )  # Start angles for each sector (len = num_rings)
        self.entry_cell: Optional[Cell] = None
        self.exit_cell: Optional[Cell] = None

        print(
            f"--- Initializing Grid (Center r={self.center_radius:.2f}, Outer r={self.total_radius:.2f}, Rings={self.num_rings}) ---"
        )
        self._calculate_grid_geometry(base_spokes, doubling_rings)
        self._create_cells()
        self._link_neighbours()
        self._determine_entry_exit_cells()
        print(f"--- Grid Initialized: {self.size()} cells ---")

    def _calculate_grid_geometry(
        self, base_spokes: int, doubling_rings: Optional[List[int]]
    ):
        """Calculates radii, spoke counts, and angles for each ring."""
        print("  Calculating grid geometry...")
        self.ring_radii = np.linspace(
            self.center_radius, self.total_radius, self.num_rings + 1
        ).tolist()  # Use list for easier type consistency

        # Determine rings where spoke count doubles
        if doubling_rings is None:
            # Default: Double at roughly 1/3 and 2/3 of the way out
            r1 = self.num_rings // 3
            r2 = self.num_rings * 2 // 3
            effective_doubling_rings = sorted(
                list(set([r for r in [r1, r2] if 0 < r < self.num_rings]))
            )
            print(f"    Using default spoke doubling rings: {effective_doubling_rings}")
        else:
            effective_doubling_rings = sorted(
                list(set(r for r in doubling_rings if 0 < r < self.num_rings))
            )
            print(
                f"    Using provided spoke doubling rings: {effective_doubling_rings}"
            )

        # Calculate spoke counts per ring
        self.num_spokes_per_ring = []
        current_spokes = base_spokes
        doubling_schedule = {ring_idx: True for ring_idx in effective_doubling_rings}

        for r in range(self.num_rings):
            if r in doubling_schedule:
                print(
                    f"    Doubling spokes from {current_spokes} to {current_spokes*2} entering ring {r}"
                )
                current_spokes *= 2
            self.num_spokes_per_ring.append(current_spokes)

        # Calculate spoke start angles per ring
        self.spoke_angles_per_ring = []
        for r in range(self.num_rings):
            num_s = self.num_spokes_per_ring[r]
            angles = np.linspace(0, 2 * math.pi, num_s, endpoint=False)
            self.spoke_angles_per_ring.append(angles)

    def _create_cells(self):
        """Instantiates all Cell objects based on calculated geometry."""
        print("  Creating cells...")
        self.cells = {}
        for ring in range(self.num_rings):
            num_spokes = self.num_spokes_per_ring[ring]
            if num_spokes == 0:
                continue  # Should not happen with validation, but safe check
            angles = self.spoke_angles_per_ring[ring]
            r_inner = self.ring_radii[ring]
            r_outer = self.ring_radii[ring + 1]
            delta_theta = (2 * math.pi) / num_spokes

            for sector in range(num_spokes):
                cell = Cell(ring, sector)
                cell.r_inner = r_inner
                cell.r_outer = r_outer
                cell.height = r_outer - r_inner
                cell.theta_start = angles[sector]
                # Ensure theta_end correctly represents the angle, potentially > 2pi
                cell.theta_end = angles[sector] + delta_theta
                cell.width = delta_theta  # Actual angular width
                # Calculate center properties
                cell.center_r = (r_inner + r_outer) / 2.0
                # Center angle should wrap correctly
                cell.center_theta = (cell.theta_start + cell.width / 2.0) % (
                    2 * math.pi
                )
                cell.center_x = cell.center_r * math.cos(cell.center_theta)
                cell.center_y = cell.center_r * math.sin(cell.center_theta)

                self.cells[cell.id] = cell

    def _link_neighbours(self):
        """Determines and sets the neighbours for each cell."""
        print("  Linking neighbours...")
        link_count = 0
        processed_pairs = set()  # Avoid double processing (e.g., A->B and B->A)

        for cell in self.cells.values():
            ring, sector = cell.coords
            if ring >= len(self.num_spokes_per_ring):
                continue  # Should not happen

            num_spokes_curr = self.num_spokes_per_ring[ring]

            # --- Link CW / CCW Neighbours ---
            if num_spokes_curr > 1:
                cw_sector = (sector + 1) % num_spokes_curr
                ccw_sector = (sector - 1 + num_spokes_curr) % num_spokes_curr
                cw_cell = self.get_cell(ring, cw_sector)
                ccw_cell = self.get_cell(ring, ccw_sector)

                # Link CW (CCW link is handled by the neighbour's CW link)
                if cw_cell:
                    pair = frozenset([cell.id, cw_cell.id])
                    if pair not in processed_pairs:
                        cell.neighbours[const.DIR_CW] = cw_cell
                        cw_cell.neighbours[const.DIR_CCW] = cell
                        processed_pairs.add(pair)
                        link_count += 1

            # --- Link IN / OUT Neighbours ---
            # Check IN neighbours (ring-1)
            if ring > 0:
                num_spokes_in = self.num_spokes_per_ring[ring - 1]
                for s_in in range(num_spokes_in):
                    in_cell = self.get_cell(ring - 1, s_in)
                    if in_cell and angles_overlap(
                        cell.theta_start,
                        cell.theta_end,
                        in_cell.theta_start,
                        in_cell.theta_end,
                        tolerance=const.ANGLE_OVERLAP_TOLERANCE,
                    ):
                        pair = frozenset([cell.id, in_cell.id])
                        if pair not in processed_pairs:
                            # Ensure IN/OUT lists exist before appending
                            cell.neighbours.setdefault(const.DIR_IN, []).append(in_cell)
                            in_cell.neighbours.setdefault(const.DIR_OUT, []).append(
                                cell
                            )
                            processed_pairs.add(pair)
                            link_count += 1
            # OUT neighbours (ring+1) are handled by the outer cell's IN check

        print(
            f"  Neighbour linking complete. Established {link_count} neighbour relationships."
        )

    def _determine_entry_exit_cells(
        self,
        entry_location: Union[str, Tuple[str, float]] = "pole",
        exit_location: Union[str, Tuple[str, float]] = "pole",
    ):
        """
        Selects entry and exit cells based on location specification.
        'pole': Innermost/Outermost ring, random sector.
        ('equator', angle): Outermost ring, sector containing the angle.
        """
        print(
            f"  Determining entry ({entry_location}) and exit ({exit_location}) cells..."
        )
        self.entry_cell = None
        self.exit_cell = None
        outer_ring_idx = self.num_rings - 1

        # Helper to find cell at equator angle
        def find_equator_cell(target_angle: float) -> Optional[Cell]:
            if outer_ring_idx < 0 or outer_ring_idx >= len(self.num_spokes_per_ring):
                return None
            ring_idx = outer_ring_idx
            num_spokes = self.num_spokes_per_ring[ring_idx]
            if num_spokes <= 0:
                return None

            for sector in range(num_spokes):
                cell = self.get_cell(ring_idx, sector)
                # Use is_angle_within utility for robustness
                if cell and is_angle_within(
                    target_angle, cell.theta_start, cell.theta_end
                ):
                    return cell
            print(
                f"    Warning: No cell found containing equator angle {target_angle:.3f} in ring {ring_idx}."
            )
            return None  # Should ideally find one

        # --- Determine Entry Cell ---
        if entry_location == "pole":
            if (
                self.num_rings > 0
                and 0 < len(self.num_spokes_per_ring)
                and self.num_spokes_per_ring[0] > 0
            ):
                entry_ring = 0
                num_entry_spokes = self.num_spokes_per_ring[entry_ring]
                random_entry_sector = random.randint(0, num_entry_spokes - 1)
                self.entry_cell = self.get_cell(entry_ring, random_entry_sector)
            else:
                print("    Warning: Cannot set pole entry (invalid ring 0).")
        elif isinstance(entry_location, tuple) and entry_location[0] == "equator":
            self.entry_cell = find_equator_cell(entry_location[1])

        if self.entry_cell:
            print(f"    Entry cell set to: {self.entry_cell.id}")
        else:
            print(
                f"    ERROR: Could not set entry cell for location '{entry_location}'. Maze generation might fail."
            )

        # --- Determine Exit Cell ---
        if exit_location == "pole":
            if (
                outer_ring_idx >= 0
                and outer_ring_idx < len(self.num_spokes_per_ring)
                and self.num_spokes_per_ring[outer_ring_idx] > 0
            ):
                exit_ring = outer_ring_idx
                num_exit_spokes = self.num_spokes_per_ring[exit_ring]
                # Get list of valid cells in the outer ring
                potential_exit_cells = [
                    c
                    for c in (
                        self.get_cell(exit_ring, s) for s in range(num_exit_spokes)
                    )
                    if c is not None
                ]

                # Try not to pick the same as entry if entry is also on outer ring
                if (
                    self.entry_cell
                    and self.entry_cell.ring == exit_ring
                    and len(potential_exit_cells) > 1
                ):
                    possible_exits = [
                        c for c in potential_exit_cells if c.id != self.entry_cell.id
                    ]
                    if possible_exits:
                        self.exit_cell = random.choice(possible_exits)
                    else:  # Only one cell available, must be the same
                        self.exit_cell = (
                            self.entry_cell if potential_exit_cells else None
                        )
                elif potential_exit_cells:
                    self.exit_cell = random.choice(potential_exit_cells)
                else:
                    print(
                        "    Warning: No valid cells found in outer ring for pole exit."
                    )

        elif isinstance(exit_location, tuple) and exit_location[0] == "equator":
            self.exit_cell = find_equator_cell(exit_location[1])

        if self.exit_cell:
            print(f"    Exit cell set to: {self.exit_cell.id}")
        else:
            print(
                f"    Warning: Could not set exit cell for location '{exit_location}'."
            )

    def get_cell(self, ring: int, sector: int) -> Optional[Cell]:
        """Safely retrieves a cell by ring and sector index, handling wrapping."""
        if not (0 <= ring < self.num_rings):
            return None
        if ring >= len(self.num_spokes_per_ring):  # Index check
            return None
        num_spokes = self.num_spokes_per_ring[ring]
        if num_spokes <= 0:
            return None  # No spokes in this ring
        # Wrap sector index correctly
        actual_sector = sector % num_spokes
        cell_id = f"{ring},{actual_sector}"
        return self.cells.get(cell_id)

    def random_cell(self) -> Cell:
        """Returns a random cell from the grid."""
        if not self.cells:
            raise IndexError("Cannot select random cell from empty grid.")
        return random.choice(list(self.cells.values()))

    def size(self) -> int:
        """Returns the total number of cells in the grid."""
        return len(self.cells)

    def get_all_cells(self) -> Iterator[Cell]:
        """Returns an iterator over all cells in the grid."""
        yield from self.cells.values()

    @property
    def inner_radius(self) -> float:
        """Returns the radius of the central open space."""
        return self.center_radius
