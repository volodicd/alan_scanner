# wavefront_detector.py

import math
from collections import deque

class WavefrontFrontierDetector:
    """Implementation of the Wavefront Frontier Detection (WFD) algorithm"""

    def __init__(self, grid_resolution=50):
        self.grid_resolution = grid_resolution
        self.grid = {}  # Sparse representation: {(x, y): cell_type}

        # Cell types
        self.UNKNOWN = 0
        self.OPEN_SPACE = 1
        self.OCCUPIED = 2

        # Temporary state for WFD
        self.map_open_list = set()
        self.map_close_list = set()
        self.frontier_open_list = set()
        self.frontier_close_list = set()

        self.frontiers = []  # List of lists of frontier points

    def update_grid(self, x, y, cell_type):
        gx, gy = self.to_grid(x, y)
        self.grid[(gx, gy)] = cell_type

    def get_cell_type(self, x, y):
        gx, gy = self.to_grid(x, y)
        return self.grid.get((gx, gy), self.UNKNOWN)

    def to_grid(self, x, y):
        return (
            round(x / self.grid_resolution) * self.grid_resolution,
            round(y / self.grid_resolution) * self.grid_resolution,
        )

    def is_frontier_point(self, x, y):
        if self.get_cell_type(x, y) != self.UNKNOWN:
            return False

        # Check for at least one open neighbor
        for dx, dy in self._directions():
            if self.get_cell_type(x + dx, y + dy) == self.OPEN_SPACE:
                return True
        return False

    def detect_frontiers(self, robot_x, robot_y):
        # Reset state
        self.map_open_list.clear()
        self.map_close_list.clear()
        self.frontier_open_list.clear()
        self.frontier_close_list.clear()
        self.frontiers.clear()

        start = self.to_grid(robot_x, robot_y)
        queue_m = deque([start])
        self.map_open_list.add(start)

        for point in self._bfs(queue_m):
            if self.is_frontier_point(*point):
                frontier = self._extract_frontier(point)
                if len(frontier) >= 3:
                    self.frontiers.append(frontier)

        return self._calculate_frontier_medians(robot_x, robot_y)

    def _bfs(self, queue):
        while queue:
            x, y = queue.popleft()
            if (x, y) in self.map_close_list:
                continue
            self.map_close_list.add((x, y))

            yield (x, y)

            for dx, dy in self._directions():
                nx, ny = x + dx, y + dy
                if (nx, ny) not in self.map_open_list and (nx, ny) not in self.map_close_list:
                    if self.get_cell_type(nx, ny) == self.OPEN_SPACE:
                        queue.append((nx, ny))
                        self.map_open_list.add((nx, ny))

    def _extract_frontier(self, start):
        queue_f = deque([start])
        self.frontier_open_list.add(start)
        frontier = []

        while queue_f:
            x, y = queue_f.popleft()
            if (x, y) in self.frontier_close_list:
                continue
            self.frontier_close_list.add((x, y))

            if self.is_frontier_point(x, y):
                frontier.append((x, y))
                for dx, dy in self._directions():
                    nx, ny = x + dx, y + dy
                    neighbor = (nx, ny)
                    if neighbor not in self.frontier_open_list and neighbor not in self.frontier_close_list:
                        queue_f.append(neighbor)
                        self.frontier_open_list.add(neighbor)

        for fx, fy in frontier:
            self.map_close_list.add((fx, fy))

        return frontier

    def _calculate_frontier_medians(self, robot_x, robot_y):
        medians = []
        for frontier in self.frontiers:
            if not frontier:
                continue
            xs = sorted(x for x, _ in frontier)
            ys = sorted(y for _, y in frontier)
            median_x = xs[len(xs) // 2]
            median_y = ys[len(ys) // 2]
            dist = math.hypot(median_x - robot_x, median_y - robot_y)
            medians.append((median_x, median_y, dist))

        medians.sort(key=lambda t: t[2])
        return medians

    def _directions(self):
        return [
            (0, self.grid_resolution),
            (self.grid_resolution, 0),
            (0, -self.grid_resolution),
            (-self.grid_resolution, 0),
        ]
