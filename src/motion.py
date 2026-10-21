"""
Motion control system for the Thymio robot.
Implements path following with global and local controllers for obstacle avoidance.
"""

import math

from src.consts import *


class Motion:
    """
    Controls Thymio robot motion including path following and obstacle avoidance.
    """

    def __init__(self, node):
        """
        Initialize motion controller with Thymio node connection.
        
        Args:
            node: Thymio robot node instance
        """
        self.node = node
        self.local = False
        self.counter = 0

    async def follow_path(self, plan, pose, grid):
        """
        Follow planned path to reach waypoints.
        
        Args:
            plan: List of waypoints to follow
            pose: Current robot pose (position, orientation)
            grid: Occupancy grid for navigation
            
        Returns:
            Boolean indicating if robot is rotating
        """
        if self.check_reached(plan, pose):
            if len(plan) == 0:
                await self.reset()
                return False

            return await self.follow_path(plan, pose, grid)

        if self.aligned(pose, plan) or self.local:
            await self.goto_waypoint(plan, pose)
            return False
        else:
            await self.align(pose, plan)
            return True

    async def goto_waypoint(self, plan, pose):
        """
        Move towards current waypoint using appropriate controller.
        
        Args:
            plan: List of waypoints
            pose: Current robot pose
        """
        prox_values = self.node.v.prox.horizontal[:5]

        # Switch to local controller if obstacle detected
        if any(value > PROX_TRESHOLD for value in prox_values) and not self.local:
            self.local = True

        if self.local:
            await self.local_controller(plan, pose)
        else:
            await self.global_controller()

    async def local_controller(self, plan, pose):
        """
        Obstacle avoidance controller using proximity sensors.
        Combines attractive force to goal with repulsive forces from obstacles.
        
        Args:
            plan: List of waypoints
            pose: Current robot pose
        """
        prox_values = self.node.v.prox.horizontal[:5]

        # Sensor angles relative to robot heading
        sensor_angles = [
            math.pi / 6,
            math.pi / 12,
            0,
            -math.pi / 12,
            -math.pi / 6
        ]

        max_prox = 4500.0
        k_repulsive = 1.5

        angle_to_goal = self.angle_difference(plan, pose)

        # Compute repulsive forces from obstacles
        repulsive_turn = 0.0
        total_obstacle_intensity = 0.0

        for i, prox_value in enumerate(prox_values):
            if prox_value > 0:
                normalized_prox = min(prox_value / max_prox, 1.0)
                total_obstacle_intensity += normalized_prox
                force_magnitude = k_repulsive * normalized_prox
                repulsive_turn -= force_magnitude * sensor_angles[i]

        # Balance goal attraction and obstacle repulsion
        obstacle_weight = min(1.0, total_obstacle_intensity)
        goal_weight = max(0.3, 1.0 - obstacle_weight)*0.45

        combined_turn = goal_weight * (angle_to_goal) + obstacle_weight * repulsive_turn

        # Adjust speed based on front obstacle proximity
        center_obstacle_factor = max(0.3, 1.0 - prox_values[2] / max_prox)
        base_speed = int(FORWARD_SPEED * center_obstacle_factor)

        # Compute motor speeds with turn adjustment
        k_turn = 80
        turn_speed = int(k_turn * combined_turn)
        max_turn = 150
        turn_speed = max(-max_turn, min(max_turn, turn_speed))

        left_speed = base_speed - turn_speed
        right_speed = base_speed + turn_speed

        left_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, left_speed))
        right_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, right_speed))

        await self.node.set_variables({
            "motor.left.target": [left_speed],
            "motor.right.target": [right_speed],
        })

        # Switch back to global controller when obstacles clear
        if all(value <= PROX_TRESHOLD for value in prox_values):
            self.counter += 1
            if self.counter >= 30:
                self.local = False
                self.counter = 0
        else:
            self.counter = 0

    def angle_difference(self, plan, pose):
        """
        Compute angle difference between robot heading and waypoint direction.
        
        Args:
            plan: List of waypoints
            pose: Current robot pose
            
        Returns:
            Angle difference in radians
        """
        x_wp = plan[0][X]
        y_wp = plan[0][Y]

        dx = x_wp - pose[0][X]
        dy = y_wp - pose[0][Y]
        target_angle = math.atan2(dy, dx)
        angle_diff = -(target_angle - pose[1])

        angle_diff = angle_diff - 2 * math.pi * math.floor((angle_diff + math.pi) / (2 * math.pi))

        return angle_diff

    async def global_controller(self):
        """
        Simple forward motion controller for obstacle-free paths.
        """
        await self.node.set_variables({
            "motor.left.target": [FORWARD_SPEED],
            "motor.right.target": [FORWARD_SPEED],
        })

    async def align(self, pose, plan):
        """
        Rotate robot to face target waypoint.
        
        Args:
            pose: Current robot pose
            plan: List of waypoints
        """
        angle_diff = self.angle_difference(plan, pose)

        if len(plan) == 1:
            target_speed = ROTATION_SPEED_FINAL
        else:
            target_speed = ROTATION_SPEED

        if angle_diff > 0:
            target_speed *= -1

        await self.node.set_variables({
            "motor.left.target": [target_speed],
            "motor.right.target": [-target_speed],
        })

    async def reset(self):
        """
        Stop all motor movement.
        """
        await self.node.set_variables({
            "motor.left.target": [0],
            "motor.right.target": [0],
        })

    def aligned(self, pose, plan):
        """
        Check if robot is aligned with target waypoint.
        
        Args:
            pose: Current robot pose
            plan: List of waypoints
            
        Returns:
            Boolean indicating alignment status
        """
        angle_diff = self.angle_difference(plan, pose)

        if len(plan) == 1:
            return abs(angle_diff) < ANGLE_TOLERANCE_FINAL
        else:
            return abs(angle_diff) < ANGLE_TOLERANCE

    def check_reached(self, plan, pose):
        """
        Check if current waypoint has been reached.
        Removes waypoint from plan if reached.
        
        Args:
            plan: List of waypoints
            pose: Current robot pose
            
        Returns:
            Boolean indicating if waypoint was reached
        """
        x_wp = plan[0][X]
        y_wp = plan[0][Y]
        dx = x_wp - pose[0][X]
        dy = y_wp - pose[0][Y]
        distance = (dx**2 + dy**2)**0.5

        if len(plan) == 1:
            if distance < DISTANCE_TOLERANCE_FINAL:
                plan.pop(0)
                return True
            return False
        else:
            if distance < DISTANCE_TOLERANCE:
                plan.pop(0)
                return True
            return False


async def test(path, pose):
    with ClientAsync() as client:
        with await client.lock() as node:
            await node.wait_for_variables({"motor.left.target", "motor.right.target", "prox.horizontal"})
            motion = Motion(node)

            for pose in poses:
                await motion.follow_path(path, pose)


if __name__ == "__main__":
    import asyncio
    import math
    from tdmclient import ClientAsync

    path = [(100, 100), (200, 200), (300, 100)]
    poses = [
        (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99,
                                                                                                                                                                                                                                                                        (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)), (99, 99, (3.14/4)),
        (50, 50, 0.1),
        (50, 50, 0.1),
        (50, 50, 0)
    ]

    asyncio.run(test(path, poses))
