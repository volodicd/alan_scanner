#!/usr/bin/env python3

import math
import heapq

class AStarPathfinder:
    """A* pathfinding algorithm implementation for TurtleBot navigation"""
    
    def __init__(self, grid_size=50):
        """Initialize the A* pathfinder
        
        Args:
            grid_size (int): Size of grid cells in cm
        """
        self.grid_size = grid_size
        self.obstacles = set()  # Set of (grid_x, grid_y) positions
        
    def add_obstacle(self, x, y):
        """Add an obstacle to the pathfinder
        
        Args:
            x (float): X coordinate in cm
            y (float): Y coordinate in cm
        """
        grid_x = round(x / self.grid_size) * self.grid_size
        grid_y = round(y / self.grid_size) * self.grid_size
        self.obstacles.add((grid_x, grid_y))
    
    def remove_obstacle(self, x, y):
        """Remove an obstacle from the pathfinder
        
        Args:
            x (float): X coordinate in cm
            y (float): Y coordinate in cm
        """
        grid_x = round(x / self.grid_size) * self.grid_size
        grid_y = round(y / self.grid_size) * self.grid_size
        self.obstacles.discard((grid_x, grid_y))
    
    def clear_obstacles(self):
        """Clear all obstacles"""
        self.obstacles.clear()
    
    def sync_obstacles_from_grid(self, grid, occupied_value):
        """Sync obstacles from a grid representation
        
        Args:
            grid (dict): Dictionary mapping (x, y) to cell values
            occupied_value: Value that represents an occupied cell
        """
        self.clear_obstacles()
        for (x, y), value in grid.items():
            if value == occupied_value:
                self.obstacles.add((x, y))
    
    def heuristic(self, x1, y1, x2, y2):
        """Calculate Manhattan distance heuristic
        
        Args:
            x1, y1: Start coordinates
            x2, y2: Goal coordinates
            
        Returns:
            float: Manhattan distance
        """
        return abs(x1 - x2) + abs(y1 - y2)
    
    def find_path(self, start_x, start_y, goal_x, goal_y, bounds=(-500, 500, -500, 500)):
        """Find path using A* algorithm
        
        Args:
            start_x, start_y: Starting position in cm
            goal_x, goal_y: Goal position in cm
            bounds: (min_x, max_x, min_y, max_y) bounds for valid positions
            
        Returns:
            list: List of (x, y) waypoints, or empty list if no path found
        """
        # Round to grid
        start_x = round(start_x / self.grid_size) * self.grid_size
        start_y = round(start_y / self.grid_size) * self.grid_size
        goal_x = round(goal_x / self.grid_size) * self.grid_size
        goal_y = round(goal_y / self.grid_size) * self.grid_size
        
        # Check if start or goal is an obstacle
        if (start_x, start_y) in self.obstacles or (goal_x, goal_y) in self.obstacles:
            return []
        
        # A* algorithm data structures
        open_list = []  # Priority queue (f_score, (x, y))
        closed_set = set()
        g_score = {(start_x, start_y): 0}
        came_from = {}
        
        # Add start to open list
        f = self.heuristic(start_x, start_y, goal_x, goal_y)
        heapq.heappush(open_list, (f, (start_x, start_y)))
        
        # Directions: right, up, left, down
        directions = [(self.grid_size, 0), (0, self.grid_size), 
                     (-self.grid_size, 0), (0, -self.grid_size)]
        
        # Optional: Add diagonal movements for smoother paths
        # directions.extend([(self.grid_size, self.grid_size), (self.grid_size, -self.grid_size),
        #                    (-self.grid_size, self.grid_size), (-self.grid_size, -self.grid_size)])
        
        min_x, max_x, min_y, max_y = bounds
        
        while open_list:
            # Get node with lowest f-score
            _, current = heapq.heappop(open_list)
            
            # Check if goal reached
            if current == (goal_x, goal_y):
                # Reconstruct path
                path = [(goal_x, goal_y)]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
                
            # Add to closed set
            closed_set.add(current)
            
            # Check neighbors
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                # Skip if out of bounds
                if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                    continue
                
                # Skip if obstacle
                if neighbor in self.obstacles:
                    continue
                    
                # Skip if in closed set
                if neighbor in closed_set:
                    continue
                    
                # Calculate tentative g score
                # Use Euclidean distance for diagonal movements if enabled
                if dx != 0 and dy != 0:
                    move_cost = math.sqrt(2) * self.grid_size
                else:
                    move_cost = self.grid_size
                
                g = g_score[current] + move_cost
                
                # Update if better path found
                if neighbor not in g_score or g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = g
                    f = g + self.heuristic(nx, ny, goal_x, goal_y)
                    heapq.heappush(open_list, (f, neighbor))
        
        # No path found
        return []
    
    def smooth_path(self, path):
        """Smooth the path by removing unnecessary waypoints
        
        Args:
            path: List of (x, y) waypoints
            
        Returns:
            list: Smoothed path
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to connect current point to furthest visible point
            for j in range(len(path) - 1, i, -1):
                if self._is_path_clear(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                # If no direct path found, add next point
                i += 1
                if i < len(path):
                    smoothed.append(path[i])
        
        return smoothed
    
    def _is_path_clear(self, start, end):
        """Check if path between two points is clear of obstacles
        
        Args:
            start: (x, y) tuple
            end: (x, y) tuple
            
        Returns:
            bool: True if path is clear
        """
        x1, y1 = start
        x2, y2 = end
        
        # Use Bresenham's line algorithm to check all cells along the path
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            # Check if current cell is an obstacle
            grid_x = round(x / self.grid_size) * self.grid_size
            grid_y = round(y / self.grid_size) * self.grid_size
            
            if (grid_x, grid_y) in self.obstacles:
                return False
            
            # Check if reached end
            if abs(x - x2) < self.grid_size/2 and abs(y - y2) < self.grid_size/2:
                break
            
            # Move to next cell
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx * self.grid_size
            if e2 < dx:
                err += dx
                y += sy * self.grid_size
        
        return True
