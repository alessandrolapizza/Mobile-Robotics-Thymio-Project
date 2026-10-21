"""
Utility functions for visualization and coordinate transformations.
Provides drawing helpers and conversion between coordinate systems.
"""

import cv2
import numpy as np
import math

from src.consts import *


def draw_control_room(vis, frame, plan):
    """
    Display status bar with system information overlaid on camera frame.
    
    Args:
        vis: Vision system instance
        frame: Camera frame to augment
        plan: Current navigation plan
    """
    if frame.ndim == 2:
        print("Input has wrong dimension")
        exit(1)

    # Create status bar
    height, width, = frame.shape[:2]
    bar = np.zeros((STATUS_BAR_HEIGHT, width, 3), dtype=np.uint8)
    bar.fill(0)

    text_x = TEXT_PADDING + STATUS_INDICATOR_SIZE + STATUS_BAR_LABEL_GAP

    def show_status(name, position, value, gray_text=None):
        """
        Draw status indicator with label and optional value text.
        """
        marker_color = COLOR_GREEN if value else COLOR_RED
        padding = (position[0] * STATUS_BAR_SPACING[0], position[1] * STATUS_BAR_SPACING[1])
        cv2.rectangle(bar,
                      (TEXT_PADDING + padding[0], TEXT_PADDING + padding[1]),
                      (STATUS_INDICATOR_SIZE + TEXT_PADDING + padding[0],
                       STATUS_INDICATOR_SIZE + TEXT_PADDING + padding[1]),
                      marker_color, -1)
        cv2.putText(
            bar,
            name,
            (text_x + padding[0], TEXT_PADDING + STATUS_INDICATOR_SIZE - TEXT_DELTA + padding[1]),
            cv2.FONT_HERSHEY_DUPLEX,
            STATUS_BAR_FONTSCALE,
            COLOR_WHITE,
            STATUS_BAR_FONTTHICKNESS,
            cv2.LINE_AA
        )
        if gray_text is not None:
            cv2.putText(
                bar,
                gray_text,
                (text_x + padding[0] + TEXT_GRAY_POSITION,
                 TEXT_PADDING + STATUS_INDICATOR_SIZE - TEXT_DELTA + padding[1]),
                cv2.FONT_HERSHEY_DUPLEX,
                STATUS_BAR_FONTSCALE,
                COLOR_GRAY,
                STATUS_BAR_FONTTHICKNESS,
            )

    # Display system status indicators
    show_status("MARKERS", (0, 0), vis.trust_target and vis.trust_robot)
    show_status("ROBOT", (0, 1), vis.robot is not None,
                gray_text='(' + str(round(vis.robot[0], 3)) + ', ' + str(vis.robot[1][0]) + ', ' + str(
                    vis.robot[1][1]) + ')' if vis.robot else None)
    show_status("TARGET", (1, 0), vis.target is not None,
                gray_text='(' + str(vis.target[0]) + ', ' + str(vis.target[1]) + ')' if vis.target else None)
    show_status("PLAN", (1, 1), plan is None or len(plan) > 0)
    show_status("LOCK", (2, 0), vis.lock)

    combined_frame = np.vstack((frame, bar))

    cv2.imshow(WINDOW_NAME, combined_frame)
    cv2.waitKey(2)


def draw_arrow(img, pnt, theta, length, thickness=2, color=COLOR_RED, tip_length=0.5, degrees=False):
    """
    Draw directional arrow on image using origin point and angle.
    
    Args:
        img: Image to draw on
        pnt: Origin point in grid coordinates
        theta: Arrow angle
        length: Arrow length in pixels
        thickness: Line thickness
        color: Arrow color
        tip_length: Arrowhead size ratio
        degrees: Whether theta is in degrees
        
    Returns:
        Modified image
    """
    h, w = img.shape[:2]
    cell_height = h // GRID_SHAPE[0]
    cell_width = w // GRID_SHAPE[1]
    if degrees:
        theta = math.radians(theta)
    dx = -length * math.sin(-theta)
    dy = length * math.cos(-theta)
    x0, y0 = int(round(pnt[0] * cell_height + cell_height / 2)), int(round(pnt[1] * cell_width + cell_width / 2))
    x1, y1 = int(round(pnt[0] * cell_height + cell_height / 2 + dx)), int(
        round(pnt[1] * cell_width + cell_width / 2 + dy))
    cv2.arrowedLine(img, (y0, x0), (y1, x1), color, thickness=thickness, tipLength=tip_length)
    return img


def to_grid_units(frame, coordinate):
    """
    Convert pixel coordinates to grid cell coordinates.
    
    Args:
        frame: Reference frame for dimensions
        coordinate: Pixel coordinate tuple
        
    Returns:
        Grid coordinate tuple
    """
    h, w = frame.shape[:2]
    x, y = coordinate
    cell_height = h // GRID_SHAPE[0]
    cell_width = w // GRID_SHAPE[1]
    return tuple((int(y // cell_height), int(x // cell_width)))


def get_unit():
    """
    Calculate real-world dimensions of grid cells.
    
    Returns:
        Generator of cell dimensions in millimeters
    """
    return (x / y for x, y in zip(PLATEAU_DIM, GRID_SHAPE))


def filter_markers(markers, marker_ids_to_keep):
    """
    Filter detected ArUco markers to keep only specified IDs.
    
    Args:
        markers: Detected markers tuple (corners, ids, rejected)
        marker_ids_to_keep: List of marker IDs to retain
        
    Returns:
        Filtered markers tuple or None
    """
    if markers is None or markers[1] is None:
        return None

    corners, ids, rejected = markers

    mask = np.isin(ids.flatten(), marker_ids_to_keep)

    if not np.any(mask):
        return None

    filtered_corners = [corners[i] for i in range(len(corners)) if mask[i]]
    filtered_ids = ids[mask].reshape(-1, 1)

    return filtered_corners, filtered_ids, rejected
