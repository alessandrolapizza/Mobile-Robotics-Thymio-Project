"""
Computer vision system for robot and target detection using ArUco markers.
Handles camera capture, marker detection, perspective correction, and grid mapping.
"""

import copy
import math
import cv2
import numpy as np

from src.utils import *


class Vision:
    """
    Vision processing system for detecting robot, target, and obstacles using camera and ArUco markers.
    """

    def __init__(self, camera=0):
        """
        Initialize vision system with camera and display window.
        
        Args:
            camera: Camera device index
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(WINDOW_NAME, "Control Center")
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        self.grid = None
        self.trust_target = False
        self.trust_robot = False
        self.lock = False
        self.lock_counter = 0
        self.raw_frame = None
        self.per_frame = None
        self.lock_markers = None
        self.markers = None
        self.matrix = None
        self.robot = None
        self.target = None
        self.counter = 0

    def step(self):
        """
        Execute one complete vision processing cycle.
        Captures frame, detects markers, builds grid, and tracks objects.
        """
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Capture and validate frame
        self.capture()
        if self.raw_frame is None:
            self.trust_robot = False
            self.trust_target = False
            return

        # Detect ArUco markers and establish coordinate system
        self.detect_markers()
        if not self.lock and (self.markers is None or self.markers[1] is None or not all(
                mid in self.markers[1].flatten() for mid in VISION_MARKERS)):
            self.trust_robot = False
            self.trust_target = False
            return
        elif not self.lock and (self.markers is not None and self.markers[1] is not None and all(
                mid in self.markers[1].flatten() for mid in VISION_MARKERS)):
            self.lock_markers = filter_markers(self.markers, VISION_MARKERS)
            self.lock = True

        # Locate robot and target
        if self.lock and self.matrix is not None:
            self.find_targets()

        # Build obstacle grid
        self.aruco_projection()
        if self.lock and self.grid is None and self.counter > 1:
            self.build_grid()
            return

        if self.lock and self.counter != 2:
            self.counter += 1

        self.trust_robot = False if self.robot is None else True
        self.trust_target = False if self.target is None else True

    def capture(self):
        """
        Capture frame from camera.
        """
        ret, frame = self.cap.read()
        self.raw_frame = frame if ret else None

    def detect_markers(self):
        """
        Detect ArUco markers in current frame.
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.markers = detector.detectMarkers(self.raw_frame)

    def aruco_projection(self):
        """
        Apply perspective transformation using corner markers to get top-down view.

        ┌───┬───┐
        │ 1 │ 2 │
        ├───┼───┤ The frame visible on the screen.
        │ 4 │ 3 │
        └───┴───┘
        """
        height, width = self.raw_frame.shape[:2]

        # Determine which markers to hide
        markers = VISION_MARKERS
        if self.trust_robot:
            markers.append(VISION_ROBOT_MARKER)
        if self.trust_target:
            markers.append(VISION_TARGET_MARKER)
        frame = self._hide_markers(markers)

        # Compute transformation matrix on first lock
        if self.matrix is None:
            corners, ids, rejected = self.markers
            indices = {marker_id[0]: i for i, marker_id in enumerate(ids)}

            projection_points = np.float32([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ])

            markers_centers = []
            for marker_id in VISION_MARKERS:
                index = indices[marker_id]
                c = corners[index][0]
                center_x = np.mean(c[:, 0])
                center_y = np.mean(c[:, 1])
                markers_centers.append([center_x, center_y])

            markers_centers = np.float32(markers_centers)

            self.matrix = cv2.getPerspectiveTransform(markers_centers, projection_points)

        self.per_frame = cv2.warpPerspective(frame, self.matrix, (width, height))

    def build_grid(self):
        """
        Build occupancy grid from perspective-corrected frame.
        Identifies obstacles and applies safety margins.
        """
        frame_gray = cv2.cvtColor(self.per_frame, cv2.COLOR_BGR2GRAY)
        frame_filtered = cv2.bilateralFilter(frame_gray, VISION_FILTER_DIAM, VISION_FILTER_SIGMA_COLOR,
                                             VISION_FILTER_SIGMA_SPACE)
        _, frame_tresh = cv2.threshold(
            frame_filtered, VISION_TRESH, VISION_TRESH_MAX, cv2.THRESH_BINARY_INV)
        rows, cols = GRID_SHAPE
        h, w = frame_tresh.shape
        cell_height = h // rows
        cell_width = w // cols

        # Classify each grid cell
        grid = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                y0 = r * cell_height
                x0 = c * cell_width
                y1 = h if r == rows - 1 else (y0 + cell_height)
                x1 = w if c == cols - 1 else (x0 + cell_width)
                cell = frame_tresh[y0:y1, x0:x1]
                grid[r, c] = CELL_OBSTACLE if np.mean(
                    cell) > 255 / 2 else CELL_VOID

        # Add safety margin around obstacles
        obstacle_mask = (grid == CELL_OBSTACLE).astype(np.uint8) * 255
        ksize = max(1, 2 * VISION_OBSTACLE_MARGIN + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(obstacle_mask, kernel)
        margin_positions = (dilated > 0) & (grid == CELL_VOID)
        grid[margin_positions] = CELL_MARGIN

        self.grid = grid


    def _hide_markers(self, markers):
        """
        Mask ArUco markers with white rectangles to prevent false obstacle detection.
        
        Args:
            markers: List of marker IDs to hide
            
        Returns:
            Frame with markers masked
        """
        corners, ids, rejected = self.markers
        if ids is None:
            return self.raw_frame
        indices = {marker_id[0]: i for i, marker_id in enumerate(ids)}
        frame = self.raw_frame
        for marker_id in markers:
            if marker_id not in indices:
                continue
            index = indices[marker_id]
            pts = self.markers[0][index][0].reshape(-1, 1, 2)
            rect = cv2.minAreaRect(pts)
            (cx, cy), (w_box, h_box), angle = rect
            w_box = max(1.0, w_box + 2 * VISION_MARKERS_PADDING)
            h_box = max(1.0, h_box + 2 * VISION_MARKERS_PADDING)
            box = cv2.boxPoints(((cx, cy), (w_box, h_box), angle))
            box = np.int32(np.round(box))
            cv2.fillPoly(frame, [box], COLOR_WHITE)
        return frame

    def render_grid(self, path, plan, kal_pos, est_pos, kalman_variance=None):
        """
        Render visualization of grid with path, obstacles, and robot states.
        
        Args:
            path: Full planned path
            plan: Waypoints list
            kal_pos: Kalman-filtered position
            est_pos: Estimated position from wheels
            kalman_variance: Optional variance for uncertainty visualization
            
        Returns:
            RGB visualization image
        """
        h, w = self.raw_frame.shape[:2]
        vis = np.ones_like(self.raw_frame, dtype=np.uint8) * 255
        if self.grid is None:
            return vis

        rows, cols = GRID_SHAPE

        cell_height = h // rows
        cell_width = w // cols

        path_set = set(path) if path is not None else set()
        plan_set = set(plan) if plan is not None else set()

        grid = copy.deepcopy(self.grid)

        # Validate positions are within grid bounds
        if self.robot is not None and (abs(self.robot[1][0]) > GRID_SHAPE[0] or abs(self.robot[1][1]) > GRID_SHAPE[1]):
            self.robot = None
        if self.target is not None and (abs(self.target[0]) > GRID_SHAPE[0] or abs(self.target[1]) > GRID_SHAPE[1]):
            self.target = None

        # Mark positions on grid
        if est_pos is not None:
            grid[est_pos[0], est_pos[1]] = CELL_EST_POS
        if kal_pos is not None:
            grid[kal_pos[0], kal_pos[1]] = CELL_KAL_POS
        if self.robot is not None:
            grid[self.robot[1][0], self.robot[1][1]] = CELL_ROBOT
        if self.target is not None:
            grid[self.target[0], self.target[1]] = CELL_TARGET

        # Draw grid cells with appropriate colors
        for r in range(rows):
            for c in range(cols):
                y0 = r * cell_height
                x0 = c * cell_width
                y1 = h if r == rows - 1 else (y0 + cell_height)
                x1 = w if c == cols - 1 else (x0 + cell_width)
                color = COLOR_BLACK
                if grid[r, c] == CELL_MARGIN:
                    color = COLOR_GRAY
                if (r, c) in plan_set:
                    color = COLOR_ORANGE
                if grid[r, c] == CELL_EST_POS:
                    color = COLOR_BLUE
                if grid[r, c] == CELL_KAL_POS:
                    color = COLOR_PURPLE
                if grid[r, c] == CELL_ROBOT:
                    color = COLOR_RED
                elif grid[r, c] == CELL_TARGET:
                    color = COLOR_GREEN
                if grid[r, c] != CELL_VOID or (r, c) in path_set or (r, c) in plan_set:
                    cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1),
                                  color, thickness=-1)

        # Draw grid lines
        for r in range(1, rows):
            y = r * cell_height
            cv2.line(vis, (0, y), (w, y), GRID_COLOR, GRID_THICKNESS)
        for c in range(1, cols):
            x = c * cell_width
            cv2.line(vis, (x, 0), (x, h), GRID_COLOR, GRID_THICKNESS)

        # Draw orientation arrows
        if self.robot is not None:
            vis = draw_arrow(vis, self.robot[1], self.robot[0], 100, color=COLOR_RED)

        if est_pos is not None:
            vis = draw_arrow(vis, (est_pos[0], est_pos[1]), est_pos[2], 100, color=COLOR_BLUE)

        # Draw uncertainty ellipse
        if kalman_variance is not None and kal_pos is not None:
            var_x_mm, var_y_mm, kalman_state = kalman_variance
            unit = list(get_unit())
            std_x, std_y = np.sqrt(var_x_mm), np.sqrt(var_y_mm)
            axis_x = max(int(std_x / unit[1] * cell_width), 5)
            axis_y = max(int(std_y / unit[0] * cell_height), 5)
            cx = int(kal_pos[1] * cell_width + cell_width / 2)
            cy = int(kal_pos[0] * cell_height + cell_height / 2)
            cv2.ellipse(vis, (cx, cy), (axis_x, axis_y), 0, 0, 360, COLOR_BLUE, 2)

        return vis

    def find_targets(self):
        """
        Detect and localize robot and target markers.
        """
        corners, ids, rejected = self.markers

        robot = None
        target = None

        if ids is None or corners is None:
            self.robot = robot
            self.target = target
            return

        flat_ids = ids.flatten()

        # Locate robot marker
        if VISION_ROBOT_MARKER in flat_ids:
            index = np.where(flat_ids == VISION_ROBOT_MARKER)[0][0]
            center = np.mean(corners[index][0], axis=0)
            orientation = self._angle_projection(VISION_ROBOT_MARKER)
            robot = orientation, to_grid_units(
                self.raw_frame, self._point_projection(center))

        # Locate target marker
        if VISION_TARGET_MARKER in flat_ids:
            index = np.where(flat_ids == VISION_TARGET_MARKER)[0][0]
            center = np.mean(corners[index][0], axis=0)
            target = to_grid_units(
                self.raw_frame, self._point_projection(center))

        self.robot = robot
        self.target = target

    def release(self):
        """
        Clean up resources and close windows.
        """
        cv2.destroyAllWindows()
        self.cap.release()

    def _point_projection(self, point):
        """
        Transform point using perspective matrix.
        
        Args:
            point: Point coordinates to transform
            
        Returns:
            Transformed point coordinates
        """
        re_point = point.astype(np.float32).reshape(1, 1, 2)
        return cv2.perspectiveTransform(re_point, self.matrix)[0][0]

    def _angle_projection(self, id):
        """
        Calculate projected orientation angle of marker.
        
        Args:
            id: Marker ID
            
        Returns:
            Angle in radians or None
        """
        corners, ids, _ = self.markers
        if ids is None:
            return None
        flat_ids = ids.flatten()
        matches = np.where(flat_ids == id)[0]
        if matches.size == 0:
            return None
        idx = matches[0]
        pts = corners[idx][0].astype(np.float32)
        pts = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2), self.matrix).reshape(-1, 2)

        p0 = pts[0]
        p1 = pts[1]
        angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return angle
