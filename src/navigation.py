"""
A* path planning implementation for robot navigation.
Generates obstacle-avoiding paths on a grid-based map.
"""

import heapq
import math
import numpy as np
import copy

from dataclasses import dataclass
from itertools import count
from src.consts import *


@dataclass
class Node:
    """
    Represents a grid cell for path planning.
    """
    position: tuple
    nature: int

    parent = None
    f = None
    g = None
    h = None
    opened = None
    closed = None


class Navigation:
    """
    Handles path planning using A* algorithm with grid-based navigation.
    """

    def __init__(self):
        """
        Initialize navigation system with empty grid and node structures.
        """
        self.rows, self.cols = GRID_SHAPE
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.plan = []
        self.node_grid = [[Node(position=(r, c), nature=CELL_VOID)
                           for c in range(self.cols)] for r in range(self.rows)]
        self.counter = count()

    def update_plan(self, grid, robot_position, goal_position):
        """
        Update navigation plan based on current grid state and robot position.
        
        Args:
            grid: Occupancy grid with obstacles
            robot_position: Current robot position
            goal_position: Target destination
            
        Returns:
            Tuple of (waypoints, full_path)
        """
        self.grid = copy.deepcopy(grid)

        # Clear previous robot and target markers
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == CELL_ROBOT or self.grid[r, c] == CELL_TARGET:
                    self.grid[r, c] = CELL_VOID

        self.node_grid = [[Node(position=(r, c), nature=self.grid[r, c]) for c in range(
            self.grid.shape[1])] for r in range(self.grid.shape[0])]
        self.plan_and_path = self.generate_plan_and_path(
            robot_position, goal_position)

        return self.plan_and_path

    def a_star(self, start, end):
        """
        A* pathfinding algorithm to find optimal path.
        
        Args:
            start: Starting position tuple
            end: Goal position tuple
            
        Returns:
            List of positions forming the path
        """
        self.node_grid = [[Node(position=(r, c), nature=self.grid[r, c])
                           for c in range(self.cols)] for r in range(self.rows)]
        open_heap = []

        # Initialize start node
        start_node = self.node_grid[start[Y]][start[X]]
        end_node = self.node_grid[end[Y]][end[X]]
        start_node.f = 0
        start_node.g = 0

        heapq.heappush(
            open_heap, (start_node.f, next(self.counter), start_node))
        start_node.opened = True

        while open_heap:
            node = heapq.heappop(open_heap)[2]
            node.closed = True

            # Check if goal reached
            if (node.position == end):
                return self.construct_path(end_node)

            # Explore neighbors
            neighbors = self.get_neighbors(node)

            for neighbor in neighbors:
                if neighbor.closed:
                    continue

                # Calculate movement cost
                base_cost = 1 if ((neighbor.position[X] - node.position[X]) == 0 or (
                    neighbor.position[Y] - node.position[Y]) == 0) else math.sqrt(2)
                cost_multiplier = 1000 if neighbor.nature == CELL_MARGIN else 1
                g_score = node.g + (base_cost * cost_multiplier)

                # Update neighbor if better path found
                if (not neighbor.opened) or (g_score < neighbor.g):
                    neighbor.g = g_score
                    neighbor.h = (WEIGHT * self.chebyshev_distance(abs(neighbor.position[X] - end[X]), abs(
                        neighbor.position[Y] - end[Y]))) if ((neighbor.h is None) or neighbor.h == 0) else neighbor.h
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = node

                    if not neighbor.opened:
                        heapq.heappush(
                            open_heap, (neighbor.f, next(self.counter), neighbor))
                        neighbor.opened = True
                    else:
                        for index, (f, _, n) in enumerate(open_heap):
                            if n == neighbor:
                                open_heap[index] = (
                                    neighbor.f, next(self.counter), neighbor)
                                heapq.heapify(open_heap)
                                break

        return []

    def get_neighbors(self, node):
        """
        Get valid neighboring cells for pathfinding.
        Includes diagonal movement with corner-cutting prevention.

           offsets     diagonalOffsets
        ┌───┬───┬───┐   ┌───┬───┬───┐
        │   │ 0 │   │   │ 0 │   │ 1 │
        ├───┼───┼───┤   ├───┼───┼───┤
        │ 3 │   │ 1 │   │   │   │   │
        ├───┼───┼───┤   ├───┼───┼───┤
        │   │ 2 │   │   │ 3 │   │ 2 │
        └───┴───┴───┘   └───┴───┴───┘
        
        Args:
            node: Current node
            
        Returns:
            List of valid neighbor nodes
        """
        neighbors = []

        movements = [(1, 0), (0, 1), (-1, 0), (0, -1),
                     (1, -1), (1, 1), (-1, 1), (-1, -1)]

        s0 = s1 = s2 = s3 = False

        # Check cardinal directions first
        i = 0
        for move in movements[:4]:
            new_pos = (node.position[Y] + move[Y], node.position[X] + move[X])

            if (0 <= new_pos[0] < GRID_SHAPE[0] and 0 <= new_pos[1] < GRID_SHAPE[1] and
                    (self.grid[new_pos[0], new_pos[1]] == CELL_VOID or self.grid[new_pos[0], new_pos[1]] == CELL_MARGIN)):

                if i == 0:
                    s0 = True
                elif i == 1:
                    s1 = True
                elif i == 2:
                    s2 = True
                elif i == 3:
                    s3 = True

                neighbors.append(self.node_grid[new_pos[0]][new_pos[1]])

            i += 1

        # Check diagonals only if both adjacent cardinals are free
        for move in movements[4:]:
            new_pos = (node.position[Y] + move[Y], node.position[X] + move[X])

            if (0 <= new_pos[0] < GRID_SHAPE[0] and 0 <= new_pos[1] < GRID_SHAPE[1] and
                    (self.grid[new_pos[0], new_pos[1]] == CELL_VOID or self.grid[new_pos[0], new_pos[1]] == CELL_MARGIN)):

                if move == (-1, 1) and s0 and s3:
                    neighbors.append(self.node_grid[new_pos[0]][new_pos[1]])
                elif move == (1, 1) and s0 and s1:
                    neighbors.append(self.node_grid[new_pos[0]][new_pos[1]])
                elif move == (1, -1) and s1 and s2:
                    neighbors.append(self.node_grid[new_pos[0]][new_pos[1]])
                elif move == (-1, -1) and s2 and s3:
                    neighbors.append(self.node_grid[new_pos[0]][new_pos[1]])

        return neighbors

    def construct_path(self, node):
        """
        Reconstruct path from goal to start using parent links.
        
        Args:
            node: Goal node
            
        Returns:
            List of positions from start to goal
        """
        path = [(node.position)]
        while node.parent is not None:
            node = node.parent
            path.append(node.position)
        return path[::-1]

    def chebyshev_distance(self, dx, dy):
        """
        Calculate Chebyshev distance heuristic.
        
        Args:
            dx: X distance
            dy: Y distance
            
        Returns:
            Chebyshev distance
        """
        return max(dx, dy)

    def generate_plan_and_path(self, start, end):
        """
        Generate waypoints and full path from start to end.
        Waypoints are generated at direction changes.
        
        Args:
            start: Starting position
            end: Goal position
            
        Returns:
            Tuple of (waypoints, full_path)
        """
        path = []

        if end is not None:
            path = self.a_star(start, end)

        if not (len(path) > 0):
            return ([], [])

        path = path[1:]

        # Extract waypoints at direction changes
        plan = []
        previous_direction = None
        for i in range(1, len(path)):
            current_direction = (path[i][0] - path[i - 1][0],
                                 path[i][1] - path[i - 1][1])
            if current_direction != previous_direction:
                plan.append(path[i - 1])
                previous_direction = current_direction
        plan = plan[1:]
        plan.append(end)
        return (plan, path)


if __name__ == "__main__":
    import cv2
    import numpy as np

    nav = Navigation()

    start = [0, 0]
    end = [0, 1]

    dragging_start = False
    dragging_end = False
    path = []
    plan = []
    drawing_obstacle = False
    erasing_obstacle = False

    CELL_SIZE = 8
    WINDOW_WIDTH = GRID_SHAPE[1] * CELL_SIZE
    WINDOW_HEIGHT = GRID_SHAPE[0] * CELL_SIZE

    def mouse_callback(event, x, y, flags, param):
        global dragging_start, dragging_end, drawing_obstacle, erasing_obstacle, path, plan

        grid_x = min(x // CELL_SIZE, GRID_SHAPE[1] - 1)
        grid_y = min(y // CELL_SIZE, GRID_SHAPE[0] - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            if (grid_y, grid_x) == tuple(start):
                dragging_start = True
            elif (grid_y, grid_x) == tuple(end):
                dragging_end = True
            elif nav.grid[grid_y, grid_x] == CELL_OBSTACLE:
                erasing_obstacle = True
                nav.grid[grid_y, grid_x] = CELL_VOID
                path = []
                plan = []
            else:
                drawing_obstacle = True
                nav.grid[grid_y, grid_x] = CELL_OBSTACLE
                path = []
                plan = []

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_start:
                start[0], start[1] = grid_y, grid_x
                path = []
                plan = []
            elif dragging_end:
                end[0], end[1] = grid_y, grid_x
                path = []
                plan = []
            elif drawing_obstacle:
                nav.grid[grid_y, grid_x] = CELL_OBSTACLE
                path = []
                plan = []
            elif erasing_obstacle:
                nav.grid[grid_y, grid_x] = CELL_VOID
                path = []
                plan = []

        elif event == cv2.EVENT_LBUTTONUP:
            dragging_start = False
            dragging_end = False
            drawing_obstacle = False
            erasing_obstacle = False

    cv2.namedWindow("Pathfinding")
    cv2.setMouseCallback("Pathfinding", mouse_callback)

    while True:
        # Create visualization grid
        grid_visual = np.ones(
            (GRID_SHAPE[0], GRID_SHAPE[1], 3), dtype=np.uint8) * 255

        # Draw obstacles
        for r in range(GRID_SHAPE[0]):
            for c in range(GRID_SHAPE[1]):
                if nav.grid[r, c] == CELL_OBSTACLE:
                    grid_visual[r, c] = (0, 0, 0)

        # Draw path and waypoints
        for pos in path:
            grid_visual[pos[0], pos[1]] = (0, 255, 255)

        grid_visual[start[0], start[1]] = (0, 255, 0)
        grid_visual[end[0], end[1]] = (0, 0, 255)

        for pos in plan:
            grid_visual[pos[0], pos[1]] = (255, 0, 0)

        # Resize and draw grid lines
        display = cv2.resize(grid_visual, (WINDOW_WIDTH, WINDOW_HEIGHT),
                             interpolation=cv2.INTER_NEAREST)

        for i in range(0, WINDOW_HEIGHT, CELL_SIZE):
            cv2.line(display, (0, i), (WINDOW_WIDTH, i), (200, 200, 200), 1)
        for i in range(0, WINDOW_WIDTH, CELL_SIZE):
            cv2.line(display, (i, 0), (i, WINDOW_HEIGHT), (200, 200, 200), 1)

        cv2.imshow("Pathfinding", display)

        key = cv2.waitKey(1) & 0xFF

        # Handle keyboard commands
        if key == ord('s'):
            nav.node_grid = [[Node(position=(r, c), nature=nav.grid[r, c])
                              for c in range(GRID_SHAPE[1])] for r in range(GRID_SHAPE[0])]
            nav.counter = count()

            path = nav.a_star(tuple(start), tuple(end))
            if path:
                plan_and_path = nav.generate_plan_and_path(
                    tuple(start), tuple(end))
                plan = plan_and_path[0]
                path = plan_and_path[1]
            else:
                plan = []

        elif key == ord('c'):
            nav.grid = np.zeros((GRID_SHAPE[0], GRID_SHAPE[1]), dtype=int)
            path = []
            plan = []

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
