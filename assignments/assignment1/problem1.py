import os
import numpy as np
import mengine as m
np.set_printoptions(precision=3, suppress=True)

# NOTE: This problem asks you to convert between the different rotation representations.

# Create environment and ground plane
env = m.Env()
ground = m.Ground([0, 0, -0.5])
env.set_gui_camera(look_at_pos=[0, 0, 0])

# position definition
x = np.array([0.2, 0, 0])

# Create points to rotate
# point rotated using euler angles
point_e = m.Shape(m.Sphere(radius=0.03), static=True,
                  position=x, rgba=[0, 1, 0, 0.2])
# point rotated using axis-angle
point_aa = m.Shape(m.Sphere(radius=0.025), static=True,
                   position=x, rgba=[1, 0, 0, 0.2])
# point rotated using rotation matrix
point_r = m.Shape(m.Sphere(radius=0.02), static=True,
                  position=x, rgba=[0, 0, 1, 0.2])


def rodrigues_formula(n, x, theta):
    # Rodrigues' formula for axis-angle: rotate a point x around an axis n by angle theta
    # input: n, x, theta: axis, point, angle
    # output: x_new: new point after rotation
    # ------ TODO Student answer below -------
    n = n / np.linalg.norm(n)  # ensure n is a unit vector
    x_new = x * np.cos(theta) + np.cross(n, x) * np.sin(theta) + n * np.dot(n, x) * (1 - np.cos(theta))
    return x_new
    # ------ Student answer above -------


def rotate_euler(alpha, beta, gamma, x):
    # Rotate a point x using euler angles (alpha, beta, gamma)
    # input: alpha, beta, gamma: euler angles
    # output: x_new: new point after rotation

    # ------ TODO Student answer below -------
    R = euler_to_rotation_matrix(alpha, beta, gamma)
    x_new = R.dot(x)
    return x_new
    # ------ Student answer above -------


def euler_to_rotation_matrix(alpha, beta, gamma):
    # Convert euler angles (alpha, beta, gamma) to rotation matrix
    # input: alpha, beta, gamma: euler angles
    # output: R: rotation matrix

    # ------ TODO Student answer below -------
    def Rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0.0],
                         [np.sin(a), np.cos(a), 0.0],
                         [0.0, 0.0, 1.0]], dtype=float)
    def Ry(b):
        return np.array([[np.cos(b), 0.0, np.sin(b)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(b), 0.0, np.cos(b)]], dtype=float)
    R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    return R
    # ------ Student answer above -------


def euler_to_axis_angle(alpha, beta, gamma):
    # Convert euler angles (alpha, beta, gamma) to axis-angle representation (n, theta)
    # input: alpha, beta, gamma: euler angles
    # output: n, theta
    # ------ TODO Student answer below -------
    R = euler_to_rotation_matrix(alpha, beta, gamma)
    
    # Extract angle from rotation matrix trace
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta :
        # Extract rotation axis from rotation matrix
        n = np.array([R[2, 1] - R[1, 2],
                      R[0, 2] - R[2, 0],
                      R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
        return n, theta
    if np.abs(theta) < 1e-6: # No rotation
        n = np.array([1, 0, 0]) # arbitrary axis
        theta = 0
    elif np.abs(theta - np.pi) < 1e-6: # 180 degree rotation
        diag = np.diag(R)
        max_idx = np.argmax(diag)
        n = np.zeros(3)
        n[max_idx] = np.sqrt((R[max_idx, max_idx] + 1) / 2)
        
        for i in range(3):
            if i != max_idx:
                n[i] = R[max_idx, i] / (2 * n[max_idx])
    else:
        # Extract rotation axis from rotation matrix
        n = np.array([R[2, 1] - R[1, 2],
                      R[0, 2] - R[2, 0],
                      R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
        return n, theta
    return n, theta
    # ------ Student answer above -------


x_new_e = np.array([0.2, 0, 0])
x_new_r = np.array([0.2, 0, 0])
x_new_aa = np.array([0.2, 0, 0])

for alpha, beta, gamma in zip([20, -25, 0], [45, 5, 135], [10, 90, -72]):
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    (n, theta) = euler_to_axis_angle(alpha, beta, gamma)
    R = euler_to_rotation_matrix(alpha, beta, gamma)

    # positions of rotated points for each representation
    x_new_e = rotate_euler(alpha, beta, gamma, x)
    x_new_r = R.dot(x)
    x_new_aa = rodrigues_formula(n, x, theta)

    print('-'*20)
    print('Euler angles:', np.degrees(alpha), np.degrees(beta), np.degrees(gamma))
    print('Axis angle:', n, np.degrees(theta))
    print('Rotation matrix:', R)
    print('x_new_e:', x_new_e)
    print('x_new_r:', x_new_r)
    print('x_new_aa:', x_new_aa)
    print('-'*20)

    point_e.set_base_pos_orient(x_new_e)
    point_r.set_base_pos_orient(x_new_r)
    point_aa.set_base_pos_orient(x_new_aa)

    # NOTE: Press enter to continue to next angles
    print('Press enter in the simulator to continue to the next angle set')
    keys = m.get_keys()
    while True:
        keys = m.get_keys()
        if 'return' in keys:
            break
        m.step_simulation(realtime=True)
    m.step_simulation(steps=50, realtime=True)
