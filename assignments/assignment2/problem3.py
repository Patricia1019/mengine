import time
import os
import mengine as m
import numpy as np


def invertQ(q):
    """
    Invert a quaternion, this function is optional and you could use it in line_intersection if you want
    """
    # ------ TODO Student answer below -------
    # NOTE: Optional, you do not need to use this function
    q = np.array(q, dtype=float)
    x, y, z, w = q
    
    norm_sq = x**2 + y**2 + z**2 + w**2
    
    # Handle zero or near-zero quaternion case
    if norm_sq < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    return np.array([-x / norm_sq, -y / norm_sq, -z / norm_sq, w / norm_sq])
    # ------ Student answer above -------


def line_intersection(p1, p2, q1, q2):
    """
    Find the intersection of two 3D line segments p1-p2 and q1-q2.
    If there is an intersection, returns the point. Otherwise, returns None.
    """
    # ------ TODO Student answer below -------
    d1 = p2 - p1
    d2 = q2 - q1 
    
    # Check for degenerate line segments
    d1_len = np.linalg.norm(d1)
    d2_len = np.linalg.norm(d2)
    
    if d1_len < 1e-12 or d2_len < 1e-12:
        return None
    
    # Vector between starting points
    dp = p1 - q1
    
    # Check if lines are parallel
    cross_d1_d2 = np.cross(d1, d2)
    cross_norm = np.linalg.norm(cross_d1_d2)
    
    # If lines are parallel (cross product is zero)
    if cross_norm < 1e-12:
        # Check if lines are coincident (same line)
        cross_dp_d1 = np.cross(dp, d1)
        if np.linalg.norm(cross_dp_d1) < 1e-12:
            # Lines are coincident, find overlap
            # Project q1 and q2 onto line 1
            dot_d1 = np.dot(d1, d1)
            if dot_d1 > 1e-12:
                t1 = np.dot(q1 - p1, d1) / dot_d1
                t2 = np.dot(q2 - p1, d1) / dot_d1
                
                # Check for overlap in [0,1] range
                t_min = max(0.0, min(t1, t2))
                t_max = min(1.0, max(t1, t2))
                
                if t_min <= t_max:
                    # Return midpoint of overlap
                    t_mid = (t_min + t_max) * 0.5
                    return p1 + t_mid * d1
        return None
    
    # If lines are not parallel
    cross_norm_sq = cross_norm * cross_norm
    
    cross_d2_dp = np.cross(d2, dp)
    cross_d1_dp = np.cross(d1, dp)
    
    t1 = np.dot(cross_d2_dp, cross_d1_d2) / cross_norm_sq
    t2 = np.dot(cross_d1_dp, cross_d1_d2) / cross_norm_sq
    
    # Check if intersection points are within the line segments [0,1]
    if 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:
        # Calculate the two closest points
        point1 = p1 + t1 * d1
        point2 = q1 + t2 * d2
        
        # Check if the lines actually intersect (distance between closest points)
        distance = np.linalg.norm(point1 - point2)
        
        if distance < 1e-6:  # Lines intersect within tolerance
            return (point1 + point2) * 0.5
        
        # For nearly intersecting lines, still return the midpoint
        # This is useful for the instantaneous center calculation
        if distance < 0.01:  # More generous tolerance for IC calculation
            return (point1 + point2) * 0.5
    
    return None
    # ------ Student answer above -------


# Create environment and ground plane
env = m.Env()
# ground = m.Ground()
env.set_gui_camera(look_at_pos=[0, 0.4, 0.25])

fbl = m.URDF(filename=os.path.join(m.directory, 'fourbarlinkage.urdf'),
             static=True, position=[0, 0, 0.3], orientation=[0, 0, 0, 1])
fbl.controllable_joints = [0, 1, 2]
# Create a constraint for the 4th joint to create a closed loop
fbl.create_constraint(parent_link=1, child=fbl, child_link=4, joint_type=m.p.JOINT_POINT2POINT, joint_axis=[
                      0, 0, 0], parent_pos=[0, 0, 0], child_pos=[0, 0, 0])
m.step_simulation(steps=20, realtime=False)

coupler_links = [1, 3, 5]

links = [1, 3]
global_points = []
previous_global_points = []
lines = [None, None]
lines_start_end = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]

for link in links:
    global_points.append(fbl.get_link_pos_orient(link)[0])
    previous_global_points.append(global_points[-1])
    point = m.Shape(m.Sphere(radius=0.02), static=True,
                    position=global_points[-1], rgba=[0, 0, 1, 1])

intersect_points_local = []
intersect_points_local_bodies = []

for i in range(10000):
    fbl.control([np.radians(i)]*3)

    if i > 3:
        for j, (link, global_position, previous_global_position) in enumerate(zip(links, global_points, previous_global_points)):
            p_new = fbl.get_link_pos_orient(link)[0]
            ic_vector_of_motion = p_new - previous_global_position
            ic_bisector = np.cross(ic_vector_of_motion, [0, 1, 0])
            ic_bisector = ic_bisector / np.linalg.norm(ic_bisector)
            previous_global_points[j] = p_new

            lines[j] = m.Line(p_new-ic_bisector, p_new+ic_bisector,
                              radius=0.005, rgba=[0, 0, 1, 0.5], replace_line=lines[j])
            lines_start_end[j] = (p_new-ic_bisector, p_new+ic_bisector)

        if len(intersect_points_local) < 400:
            # stop drawing if we have drawn 500 points
            intersect_point = line_intersection(
                lines_start_end[0][0], lines_start_end[0][1], lines_start_end[1][0], lines_start_end[1][1])

            if intersect_point is not None:
                m.Shape(m.Sphere(radius=0.005), static=True,
                        position=intersect_point, collision=False, rgba=[1, 0, 0, 1])
                # draw moving centrode
                # get intersection point in local frame w.r.t. link 4
                p, _ = fbl.global_to_local_coordinate_frame(intersect_point, link=3)
                local_intersect_point = np.array(p)

                intersect_points_local.append(local_intersect_point)
                # get global coordinates of intersection point
                intersect_point_local_body = m.Shape(m.Sphere(radius=0.005), static=True,
                                                     position=intersect_point, collision=False, rgba=[0, 1, 0, 1])
                intersect_points_local_bodies.append(
                    intersect_point_local_body)

        # redraw intersection points of moving centrode
        for body, point_local in zip(intersect_points_local_bodies, intersect_points_local):
            p, _ = fbl.local_to_global_coordinate_frame(point_local, link=3)
            body.set_base_pos_orient(p)

    m.step_simulation(realtime=True)

    if i == 500 or i == 600 or i == 700:
        print('--------------------------------------------------------------')
        print(f'Frame {i}: Please save screenshot and include in writeup')
        input("Press Enter to continue...")
