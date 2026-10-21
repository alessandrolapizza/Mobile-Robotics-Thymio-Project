"""
Main entry point for the Thymio robot navigation system.
Coordinates vision, Kalman filtering, navigation planning, and motion control.
"""

from time import sleep

import asyncio
from tdmclient import ClientAsync

from src.navigation import Navigation
from src.motion import Motion
from src.kalman import ThymioEKF
from src.vision import *


async def main():
    """
    Main control loop for robot navigation and obstacle avoidance.
    Integrates vision tracking, Kalman filtering, path planning, and motion control.
    """
    with ClientAsync() as client:
        with await client.lock() as node:
            await node.wait_for_variables({"motor.left.target", "motor.right.target",
                                           "motor.left.speed", "motor.right.speed",
                                           "prox.horizontal"})
            
            # Initialize system components
            mot = Motion(node)
            vis = Vision()
            nav = Navigation()
            sleep(2)

            # Initialize state variables
            plan = []
            path = []
            kalman = None
            previous_vis_trust_robot = False
            compute_path = False
            rotation = False
            robot_position = None
            robot_position_wheels_kalman = None

            while True:
                # Process vision input
                vis.step()
                if vis.raw_frame is None:
                    continue

                robot_pose_vision = vis.robot

                # Initialize Kalman filter on first robot detection
                if vis.trust_robot and kalman is None:
                    kalman = ThymioEKF([robot_pose_vision[1][X], robot_pose_vision[1][Y],
                                        robot_pose_vision[0]],
                                       list(get_unit()), dt=DT)

                # Update robot position estimate with Kalman filter
                if kalman is not None:
                    vL = node.v.motor.left.speed
                    vR = node.v.motor.right.speed

                    robot_position = kalman.step(vL, vR, rotation,
                                                 [robot_pose_vision[1][1], robot_pose_vision[1][0],
                                                  robot_pose_vision[0]] if vis.trust_robot else None)

                    robot_position_wheels_kalman = kalman.wheels_estimate

                # Detect robot kidnapping (vision re-acquired after loss)
                if vis.trust_robot and not previous_vis_trust_robot:
                    compute_path = True
                previous_vis_trust_robot = vis.trust_robot

                # Compute path when needed
                if (compute_path or (not plan and vis.trust_robot and vis.trust_target)) and vis.grid is not None:
                    print("Computing new path...")
                    planning = nav.update_plan(vis.grid,
                                               (robot_position[0],
                                                robot_position[1]),
                                               vis.target)
                    plan = planning[0]
                    path = planning[1]
                    compute_path = False

                # Execute motion control
                if plan:
                    rotation = await mot.follow_path(plan, ((robot_position[0], robot_position[1]), robot_position[2]),
                                                     vis.grid)
                else:
                    await mot.reset()

                await client.sleep(DT)

                # Prepare Kalman variance for visualization
                kalman_variance = None
                if kalman is not None:
                    kalman_variance = (kalman.P[0, 0], kalman.P[1, 1], kalman.x)

                # Render and display visualization
                view = vis.render_grid(path, plan, robot_position, robot_position_wheels_kalman,
                                       kalman_variance=kalman_variance)

                frame = vis.per_frame if vis.per_frame is not None else vis.raw_frame
                if view is not None:
                    blended = cv2.addWeighted(
                        frame, FRAME_OPACITY, view.astype(np.uint8), GRID_OPACITY, 0)
                    draw_control_room(vis, blended, plan)
                else:
                    draw_control_room(vis, frame, plan)

    vis.release()


asyncio.run(main())
