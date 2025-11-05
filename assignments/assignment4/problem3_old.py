"""
Assignment 4 Problem 3: Antipodal Grasp

NOTE:
    First install open3d using: 'python3 -m pip install open3d'
    On Mac you also need 'brew install libomp'
"""
import os
from typing import Tuple, List
import numpy as np
import open3d as o3d
import mengine as m

def load_object(idx, obj_position):
    obj_names = ["bowl", "cheezit", "spam", "mustard", "tomato_soup_can", "mug"]

    obj_name = obj_names[idx]
    if obj_name == "bowl":
        object = m.URDF(filename='./bowl/object.urdf', static=False, position=obj_position, orientation=[0, 0, 0, 1])
    else:
        object = m.Shape(m.Mesh(filename=os.path.join(m.directory, 'ycb', f'{obj_name}.obj'), scale=[1, 1, 1]), static=False, mass=1.0, position=obj_position, orientation=[0, 0, 0, 1], rgba=None, visual=True, collision=True)
    return object


def sample_grasp_ee_poses(obj, num_samples=100) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample end effector poses around the object.

    Returns:
        ee_poses: A list of end effector poses (position, orientation)
    """
    # AABB -> center & extents
    obj_min, obj_max = obj.get_AABB()
    center = 0.5 * (np.array(obj_min) + np.array(obj_max))
    extents = np.array(obj_max) - np.array(obj_min)

    # rim height ~ upper section of the object; ring radius slightly outside the rim
    rim_z = center[2] + 0.34 * extents[2] 
    radius = 0.5 * max(extents[:2]) + 0.030
    inward_offset = 0.020  # push slightly toward center so fingers meet the rim

    thetas = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
    poses: List[Tuple[np.ndarray, np.ndarray]] = []

    for th in thetas:
        # position on ring
        pos = np.array([center[0] + radius * np.cos(th),
                        center[1] + radius * np.sin(th),
                        rim_z])

        # frame: y-axis -> to center; z-axis -> down; x completes RHS
        radial = center - pos
        y_axis = radial / (np.linalg.norm(radial) + 1e-9)
        z_axis = np.array([0.0, 0.0, -1.0])  # wrist down
        x_axis = np.cross(y_axis, z_axis)
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = np.array([1.0, 0.0, 0.0])
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= (np.linalg.norm(z_axis) + 1e-9)

        R = np.column_stack([x_axis, y_axis, z_axis])
        if np.linalg.det(R) < 0:
            x_axis = -x_axis
            R = np.column_stack([x_axis, y_axis, z_axis])
        quat = m.get_quaternion(R)

        # nudge inward so the gripper meets the rim
        pos = pos + inward_offset * y_axis

        poses.append((pos, quat))

    return poses



def get_antipodal_score(robot_joint_angles, pc, normals) -> float:
    """Compute antipodal score for a candidate robot configuration."""
    score = 0.0

    # No IK or in collision → 0
    if robot_joint_angles is None or robot_in_collision(robot_joint_angles):
        return 0.0

    # Set robot to evaluate
    prev_joint_angles = robot.get_joint_angles(robot.controllable_joints)
    robot.control(robot_joint_angles, set_instantly=True)

    # Put the evaluation box at current EE pose
    p, o = robot.get_link_pos_orient(robot.end_effector)
    antipodal_region.set_base_pos_orient(p, o)

    # Gripper finger axis in world: local +y
    tip_world = robot.local_to_global_coordinate_frame([0, 0.1, 0], link=robot.end_effector)[0]
    axis = tip_world - p
    nrm = np.linalg.norm(axis)
    if nrm < 1e-8:
        robot.control(prev_joint_angles, set_instantly=True)
        return 0.0
    axis /= nrm

    # Select object points inside the evaluation box
    local_pts = np.array([antipodal_region.global_to_local_coordinate_frame(pt)[0] for pt in pc])
    inside = np.logical_and(np.all(local_pts > -half_extents, axis=1),
                            np.all(local_pts <  half_extents, axis=1))
    if not np.any(inside):
        robot.control(prev_joint_angles, set_instantly=True)
        return 0.0

    n_in = normals[inside]
    proj = n_in @ axis  # alignment of normals with finger axis (± for opposite sides)

    # Alignment: want |proj| close to 1; Balance: want both signs (two opposing contacts)
    align = float(np.mean(np.abs(proj)))
    frac_pos = (proj > 0).sum() / len(proj)
    balance = 1.0 - abs(frac_pos - 0.5) * 2.0  # 1 when evenly split, 0 when one-sided

    score = 0.7 * align + 0.3 * balance

    # Restore
    robot.control(prev_joint_angles, set_instantly=True)
    return score


def find_best_grasp(obj, **kwargs) -> np.ndarray:
    """
    Find a robot configuration to grasp the given object.
    Returns best joint angles (or None if none feasible).
    """
    max_sample = int(kwargs.get('max_sample', 120))
    min_score  = float(kwargs.get('min_score', 0.55))

    # One-time point cloud
    pc, normals = get_point_cloud(obj)

    best_q, best_s = None, -1.0
    poses = sample_grasp_ee_poses(obj, num_samples=max_sample)

    # Bias IK to current pose to avoid flips
    seed = robot.get_joint_angles(robot.controllable_joints)

    for pos, quat in poses:
        q = robot.ik(robot.end_effector,
                     target_pos=pos, target_orient=quat,
                     use_current_joint_angles=True)
        if q is None:
            continue

        s = get_antipodal_score(q, pc, normals)
        if s > best_s:
            best_s, best_q = s, q

    if best_q is None:
        print("[warn] no feasible grasp found")
        return None
    if best_s < min_score:
        print(f"[warn] best antipodal score {best_s:.3f} < threshold {min_score:.2f}")
    else:
        print(f"[info] best antipodal score {best_s:.3f}")

    return best_q

def get_point_cloud(obj):
    """Returns object's point cloud and normals."""
    # Create two cameras
    camera1 = m.Camera(camera_pos=[0, -0.25, 1], look_at_pos=obj.get_base_pos_orient()[0], fov=60,
                       camera_width=1920 // 4, camera_height=1080 // 4)
    camera2 = m.Camera(camera_pos=[0, 0.25, 1], look_at_pos=obj.get_base_pos_orient()[0], fov=60,
                       camera_width=1920 // 4, camera_height=1080 // 4)
    # Show the object
    obj.change_visual(link=obj.base, rgba=[1, 1, 1, 1])
    # Capture a point cloud from the camera
    pc1, rgba1 = camera1.get_point_cloud(body=obj)
    pc2, rgba2 = camera2.get_point_cloud(body=obj)
    pc = np.concatenate([pc1, pc2], axis=0)
    # rgba = np.concatenate([rgba1, rgba2], axis=0)
    # Visualize the point cloud
    # m.DebugPoints(pc, points_rgb=rgba[:, :3], size=10)
    # Hide the object
    obj.change_visual(link=obj.base, rgba=[1, 1, 1, 0.75])

    # Create open3d point cloud from array of points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # Estimate normals for each point
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    return pc, normals


def robot_in_collision(q):
    """Returns True if the robot is in collision at the given joint angles (q).
    For simplicity, we only consider robot collision with table and objects.
    Robot self collision or collision with cubes is optional.
    """
    # set robot to joint angles
    prev_joint_angles = robot.get_joint_angles(robot.controllable_joints)
    robot.control(q, set_instantly=True)

    # robot-obstacle collision
    for obstacle in obstacles:
        if len(robot.get_closest_points(obstacle, distance=0)[-1]) != 0:
            robot.control(prev_joint_angles, set_instantly=True)
            return True

    robot.control(prev_joint_angles, set_instantly=True)
    return False


def moveto(ee_pose=None, joint_angles=None):
    """Move robot to a given ee_pose or joint angles. If both are given, ee_pose is used."""
    if ee_pose is not None:
        joint_angles = robot.ik(robot.end_effector, target_pos=ee_pose[0], target_orient=ee_pose[1],
                                use_current_joint_angles=True)
    if joint_angles is None:
        return

    robot.control(joint_angles)
    while np.linalg.norm(robot.get_joint_angles(robot.controllable_joints) - joint_angles) > 0.03:
        m.step_simulation(realtime=True)
    return


# Create environment and ground plane
env = m.Env()
ground = m.Ground()

# Create table
table = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'), static=True, position=[0, 0, 0],
               orientation=[0, 0, 0, 1])

# Create object
# ------ TODO: Experiment with different objects ------
# object options: 0 - bowl, 1 - cheezit, 2 - spam, 3 - mustard, 4 - tomato_soup_can, 5 - mug
# by default, start with idx = 0 for the bowl
obj_idx = 0
object = load_object(obj_idx, [0, 0, 0.8])
# --------------- End of experimenting ----------------

object.set_whole_body_frictions(lateral_friction=2000, spinning_friction=2000, rolling_friction=0)
m.step_simulation(50)

obstacles = [table, object]

# Create Panda manipulator
robot = m.Robot.Panda(position=[0.5, 0, 0.76])
robot.motor_gains = 0.01

# Move end effector to a starting position using IK
target_joint_angles = robot.ik(robot.end_effector,
                               target_pos=[0, 0, 1], target_orient=m.get_quaternion(np.array([np.pi, 0, 0])))
robot.control(target_joint_angles, set_instantly=True)
robot.set_gripper_position([1] * 2, set_instantly=True)  # Open gripper

# Create a region that will allow us to identify points within the gripper
position, orientation = robot.get_link_pos_orient(robot.end_effector)
half_extents = np.array([0.01, 0.04, 0.01])
antipodal_region = m.Shape(m.Box(half_extents), static=True, collision=False, position=position,
                           orientation=orientation, rgba=[0, 1, 0, 0])
gripper_line_vector = robot.local_to_global_coordinate_frame([0, 0.2, 0], link=robot.end_effector)[0]
# gripper_line = m.Line(position, gripper_line_vector, radius=0.005, rgba=[0, 0, 0, 1])

for _ in range(3):
    # Find best grasp
    robot_joint_angles = find_best_grasp(object)

    # MOVETO bowl
    moveto(joint_angles=robot_joint_angles)

    # CLOSE gripper
    robot.set_gripper_position([0] * 2, force=5000)
    m.step_simulation(steps=100, realtime=True)

    # MOVE upwards
    pos, ori = robot.get_link_pos_orient(robot.end_effector)
    moveto(ee_pose=(pos + [0, 0, 0.2], ori))
    input('Press enter to next grasping attempt...')

    # OPEN gripper
    robot.set_gripper_position([1] * 2)
    m.step_simulation(steps=50, realtime=True)
