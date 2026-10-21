"""
Extended Kalman Filter implementation for Thymio robot state estimation.
Fuses wheel PWM commands and vision measurements to estimate robot position and orientation.
"""

import numpy as np

from src.consts import *


class ThymioEKF:
    """
    Extended Kalman Filter for tracking Thymio robot pose and velocity.
    """

    def __init__(self, init_pose, conversion_square_to_mm, L=85, dt=0.1):
        """
        Initialize the EKF with initial pose and conversion parameters.
        
        Args:
            init_pose: Initial [x, y, theta] in grid coordinates
            conversion_square_to_mm: Conversion from grid units to millimeters
            L: Wheel base width in millimeters
            dt: Time step in seconds
        """
        self.conversion_wheel_speed_to_mm_s = 0.4109589041095 * (1-0.75)
        self.conversion_square_to_mm = conversion_square_to_mm
        self.dt = dt
        self.L = L

        self.wheels_estimate = (0, 0, 0)

        # Measurement noise parameters from experiments
        self.speed_variance = 122.86838250672817
        self.speed_variance_2 = self.speed_variance
        self.angle_variance = (self.speed_variance + self.speed_variance_2) * (dt**2) / (self.L**2)
        self.position_variance = self.speed_variance * dt**2

        # Initialize state vector [x, y, theta, vL, vR]
        self.x = np.array([init_pose[0]*self.conversion_square_to_mm[1],
                          init_pose[1]*self.conversion_square_to_mm[0], init_pose[2], 0.0, 0.0])

        self.P = np.eye(5) * 1.0

        # Process and measurement noise matrices
        self.Q = np.diag([self.position_variance, self.position_variance,
                         self.angle_variance, self.speed_variance, self.speed_variance_2])
        self.R_vision = np.diag([2**2, 2**2, (np.deg2rad(3))**2])

    def predict(self, vL_raw, vR_raw, rotation):
        """
        Prediction step using wheel PWM commands.
        
        Args:
            vL_raw: Left wheel speed (raw sensor value)
            vR_raw: Right wheel speed (raw sensor value)
            rotation: Whether robot is rotating in place
        """
        theta = self.x[2]

        # Convert wheel speeds to mm/s
        vL = vL_raw*self.conversion_wheel_speed_to_mm_s
        vR = vR_raw*self.conversion_wheel_speed_to_mm_s

        self.x[3] = vL
        self.x[4] = vR

        v = (vL + vR) / 2
        w = ((vL - vR) / self.L) * (12 if rotation else 1.0)

        # Update pose with nonlinear motion model
        self.x[0] += (v * np.cos(theta) * self.dt)
        self.x[1] += (v * np.sin(theta) * self.dt)
        self.x[2] += w * self.dt
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

        self.wheels_estimate = (round(self.x[1]/self.conversion_square_to_mm[0]), round(
            self.x[0]/self.conversion_square_to_mm[1]), self.x[2])

        # Compute Jacobian for linearization
        F = np.eye(5)
        F[0, 2] = -v * np.sin(theta) * self.dt
        F[1, 2] = v * np.cos(theta) * self.dt
        F[0, 3] = 0.5 * np.cos(theta) * self.dt
        F[0, 4] = 0.5 * np.cos(theta) * self.dt
        F[1, 3] = 0.5 * np.sin(theta) * self.dt
        F[1, 4] = 0.5 * np.sin(theta) * self.dt
        F[2, 3] = self.dt / self.L
        F[2, 4] = -self.dt / self.L

        self.P = F @ self.P @ F.T + self.Q

    def update_vision(self, cam_pose):
        """
        Correction step using vision measurements.
        
        Args:
            cam_pose: [x, y, theta] from camera in grid coordinates
        """
        cell_x, cell_y, theta_rad = cam_pose

        # Convert to millimeters
        x_mm = cell_x * self.conversion_square_to_mm[1]
        y_mm = cell_y * self.conversion_square_to_mm[0]

        z = np.array([x_mm, y_mm, theta_rad])
        h = self.x[:3]

        # Measurement Jacobian
        H = np.zeros((3, 5))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        # Innovation and Kalman gain
        y = z - h
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R_vision
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_vision @ K.T

    def step(self, vL, vR, rotation, cam_pose=None):
        """
        Complete filter step with prediction and optional correction.
        
        Args:
            vL: Left wheel speed
            vR: Right wheel speed
            rotation: Whether robot is rotating
            cam_pose: Optional vision measurement
            
        Returns:
            Tuple of (y, x, theta) in grid coordinates
        """
        self.predict(vL, vR, rotation)

        if cam_pose is not None:
            self.update_vision(cam_pose)

        x_mm = self.x[0]
        y_mm = self.x[1]
        theta = self.x[2]

        x_square = int(round(x_mm / self.conversion_square_to_mm[1]))
        y_square = int(round(y_mm / self.conversion_square_to_mm[0]))

        return (y_square, x_square, theta)
