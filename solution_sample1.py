# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

from final_env import FinalEnv, SolutionBase
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2axangle, qmult, qinverse

global_time_left = 200000.

def skew_symmetric_matrix(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def G_inv(theta):
    """
   Direct implementation of book formula:
   G(θ) = Iθ + (1-cos θ)[ω] + (θ-sin θ)[ω]²
   """
    theta_norm = np.linalg.norm(theta)
    if theta_norm < 1e-6:
        return np.eye(3)
    
    # Get unit vector (ω)
    omega = theta / theta_norm
    omega_skew = skew_symmetric_matrix(omega)
    
    # Compute each term
    term1 = theta_norm * np.eye(3)                    # Iθ
    term2 = (1 - np.cos(theta_norm)) * omega_skew     # (1-cos θ)[ω]
    term3 = (theta_norm - np.sin(theta_norm)) * omega_skew @ omega_skew  # (θ-sin θ)[ω]²
    
    G = term1 + term2 + term3
    return np.linalg.inv(G) 

def pose2exp_coordinate(pose):
    # Different variable names and extraction method
    T = pose  # use full transform name
    R = T[:3, :3]
    p = T[:3, 3]
    
    # Different way to check identity case
    trace_R = np.trace(R)
    if abs(trace_R - 3) < 1e-2:  # slightly different threshold
        # Pure translation case
        w = np.zeros(3)
        v = p
        magnitude = np.sqrt(np.sum(v**2))  # different way to compute norm
        return np.concatenate([w, v/magnitude]), magnitude
    
    # Rotation case - different order and structure
    angle = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))  # added clip for stability
    
    # Different way to compute rotation axis
    skew = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))
    
    # Compute rotation vector
    w = angle * skew
    
    # Get translation component
    v = G_inv(w) @ p
    
    # Different way to compute magnitude
    magnitude = np.sqrt(w @ w)
    
    # Different order of operations in return
    result = np.zeros(6)
    result[:3] = w/magnitude
    result[3:] = v
    
    return result, magnitude

def pose2mat(pose):
    """You need to implement this function

    You will need to implement this function first before any other functions.
    In this function, you need to convert a (position: pose.p, quaternion: pose.q) into a SE(3) matrix

    You can not directly use external library to transform quaternion into rotation matrix.
    Only numpy can be used here.
    Args:
        pose: sapien Pose object, where Pose.p and Pose.q are position and quaternion respectively

    Hint: the convention of quaternion

    Returns:
        (4, 4) transformation matrix represent the same pose

    """
    pos = pose.p  # 3D position
    q = pose.q    # Quaternion [w,x,y,z]
    
    # Extract quaternion components
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    # Ensure orthogonality (clean up numerical errors)
    U, _, Vh = np.linalg.svd(R)
    R = U @ Vh
    
    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = pos
    
    return T


global_time_left = 200000.

class Solution(SolutionBase):
    """
    This is a very bad baseline solution
    It operates in the following ways:
    1. roughly align the 2 spades
    2. move the spades towards the center
    3. lift 1 spade and move the other away
    4. somehow transport the lifted spade to the bin
    5. pour into the bin
    6. go back to 1
    """
    def init(self, env: FinalEnv):
        self.phase = 0
        self.drive = 0
        meta = env.get_metadata()
        self.box_ids = meta['box_ids']
        r1, r2, c1, c2, c3, c4 = env.get_agents()

        self.ps = [1000, 800, 600, 600, 200, 200, 100]
        self.ds = [1000, 800, 600, 600, 200, 200, 100]
        r1.configure_controllers(self.ps, self.ds)
        r2.configure_controllers(self.ps, self.ds)
        
        #determine target bin coordinates
        self.bin_id = meta['bin_id']
        self.bin_position = self.locate_bin(c4)
        self.bin_position[0] += 0.1  # Offset X-axis towards center of bin
        self.bin_position[1] -= 0.1  # Offset Y-axis towards center of bin
        self.bin_position[2] += 0.4  # Offset in Z-axis for safety
    

    def act(self, env: FinalEnv, current_timestep: int):
        global global_time_left
        r1, r2, c1, c2, c3, c4 = env.get_agents()

        pf_left = f = r1.get_compute_functions()['passive_force'](True, True, False)
        pf_right = f = r2.get_compute_functions()['passive_force'](True, True, False)

        if self.phase == 0:
            if 20000 - (current_timestep // 5) < 4100:
                return False

            t1 = [2, 1, 0, -1.5, -1, 1, -2]
            t2 = [-2, 1, 0, -1.5, 1, 1, -2]

            r1.set_action(t1, [0] * 7, pf_left)
            r2.set_action(t2, [0] * 7, pf_right)

            if np.allclose(r1.get_observation()[0], t1, 0.05, 0.05) and np.allclose(
                    r2.get_observation()[0], t2, 0.05, 0.05):
                self.phase = 1
                self.counter = 0
                self.selected_x = None

        if self.phase == 1:
            self.counter += 1

            if (self.counter == 1):

                selected, flag = self.pick_box1(c4)

                if 20000 - current_timestep // 5 < global_time_left:
                    if flag == False:
                        return False

                self.selected_x = selected[0]
                self.selected_y = selected[1]

            target_pose_left = Pose([self.selected_x, 0.5, 0.67], euler2quat(np.pi, -np.pi / 6, -np.pi / 2))
            self.diff_drive(r1, 9, target_pose_left)
            
            # self.jacobian_drive(r1, 9, target_pose_left)

            target_pose_right = Pose([self.selected_x, -0.5, 0.6], euler2quat(np.pi, -np.pi / 4, np.pi / 2))
            self.diff_drive(r2, 9, target_pose_right)
            # self.jacobian_drive(r2, 9, target_pose_right)

            if self.counter == 2000 / 5:
                self.phase = 2
                self.counter = 0
                
                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = 0.07
                self.pose_left = Pose(p, q)

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = -0.07
                self.pose_right = Pose(p, q)

        if self.phase == 2:
            self.counter += 1
            self.diff_drive(r1, 9, self.pose_left)
            self.diff_drive(r2, 9, self.pose_right)

            if self.counter == 3000 / 5:
                self.phase = 3

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                # p[1] = 0.5
                p[2] += 0.0
                q = euler2quat(np.pi, -np.pi / 4, -np.pi / 2)
                self.pose_left = Pose(p, q)
                print("raised r1")

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.0
                q = euler2quat(np.pi, -np.pi / 1.8, np.pi / 2)
                self.pose_right = Pose(p, q)

                self.counter = 0

        if self.phase == 3:

            if self.counter < 500 / 5:
                self.counter += 1
                self.diff_drive(r1, 9, self.pose_left)
                self.diff_drive(r2, 9, self.pose_right)
                #self.jacobian_drive(r1, 9, self.pose_left)
                #self.jacobian_drive(r2, 9, self.pose_right)

            elif self.counter < 1500 / 5:
                self.counter += 1
                t1 = [2, 1, 0, -1.5, -1, 1, -2]
                r1.set_action(t1, [0] * 7, pf_left)
                self.diff_drive(r2, 9, self.pose_right)
                #self.jacobian_drive(r2, 9, self.pose_right)

            else:
                self.phase = 4
                self.counter = 0

        if self.phase == 4:
            self.counter += 1
            if (self.counter < 3000 / 5):
                print("Phase 4a") #debug
                #TODO: collision-free trajectory to get r2 off the ground
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                #self.diff_drive2(r2, 9, Pose(p, q), [4, 5, 6], [0, 0, 0, -1, 0], [0, 1, 2, 3, 4])
                self.jacobian_drive(r2, 9, Pose(p, q))
                
            elif (self.counter < 6000 / 5):
                print("Phase 4b") #debug
                #TODO: collision-free trajectory to get r2 away from r1
                p = [-1, -0., 1.2]  # a milestone to control the trajectory for avoiding collision
                q = euler2quat(0, -np.pi / 3, 0)
                self.jacobian_drive(r2, 9, Pose(p, q), speed=0.4)
                
            elif (self.counter < 9000 / 5):
                print("Phase 4c") #debug
                # move r2 to bin
                approach_orientation = euler2quat(0, -np.pi / 2, 0) 
                target_pose = Pose(self.bin_position, approach_orientation)
                self.diff_drive(r2, index=9, target_pose=target_pose)
                
            elif (self.counter < 10000 / 5):
                print("Phase 4d") #debug
                #rotate spade
                qpos, _, _ = r2.get_observation()
                last_joint_index = r2.dof - 1  
                steps = 36
                for step in range(steps):
                    qpos[last_joint_index] += (2 * np.pi) / steps #rotate joint at angle of 2pi / steps
                    drive_target = qpos
                    drive_velocity = [0] * r2.dof  
                    additional_force = [0] * r2.dof 
                    
                    r2.set_action(drive_target, drive_velocity, additional_force)

            else:
                print("Phase 4e") #debug
                self.phase = 0
                # return False



    def jacobian_drive(self, robot, index, target_pose, speed=0.3):
            """
            This function aims to move the robot's end effector to a target pose based on Jacobian matrices,
            which relate joint velocities to end effector velocities

            para: similar to above
            """
            # ee_pose to matrix
            passive_force = robot.get_compute_functions()['passive_force'](True, True, False)
            q_position, q_velocity, poses = robot.get_observation()
            current_pose: Pose = poses[index]
            current_pose = pose2mat(current_pose)
            current_rotation = current_pose[:3, :3]
            current_position = current_pose[:3, 3]
            target_pose = pose2mat(target_pose)

            # transformation from current to target
            pose_difference = np.linalg.inv(current_pose) @ target_pose
            twist_difference, theta_difference = pose2exp_coordinate(pose_difference)
            twist_body_difference = twist_difference * speed

            # compute v with twist
            my_adjoint_matrix = np.zeros((6, 6))
            my_adjoint_matrix[0:3, 0:3] = current_rotation
            my_adjoint_matrix[3:6, 3:6] = current_rotation
            my_adjoint_matrix[3:6, 0:3] = skew_symmetric_matrix(current_position) @ current_rotation
            ee_twist_difference = my_adjoint_matrix @ twist_body_difference
            target_q_velocity = self.compute_joint_velocity_from_twist(robot, ee_twist_difference)

            robot.set_action(q_position, target_q_velocity, passive_force)

    def diff_drive(self, robot, index, target_pose):
        """
        this diff drive is very hacky
        it tries to transport the target pose to match an end pose
        by computing the pose difference between current pose and target pose
        then it estimates a cartesian velocity for the end effector to follow.
        It uses differential IK to compute the required joint velocity, and set
        the joint velocity as current step target velocity.
        This technique makes the trajectory very unstable but it still works some times.
        """
        pf = robot.get_compute_functions()['passive_force'](True, True, False)
        max_v = 0.1
        max_w = np.pi
        qpos, qvel, poses = robot.get_observation()
        current_pose: Pose = poses[index]
        delta_p = target_pose.p - current_pose.p
        delta_q = qmult(target_pose.q, qinverse(current_pose.q))

        axis, theta = quat2axangle(delta_q)
        if (theta > np.pi):
            theta -= np.pi * 2

        t1 = np.linalg.norm(delta_p) / max_v
        t2 = theta / max_w
        t = max(np.abs(t1), np.abs(t2), 0.001)
        thres = 0.1
        if t < thres:
            k = (np.exp(thres) - 1) / thres
            t = np.log(k * t + 1)
        v = delta_p / t
        w = theta / t * axis
        target_qvel = robot.get_compute_functions()['cartesian_diff_ik'](np.concatenate((v, w)), 9)
        robot.set_action(qpos, target_qvel, pf)

    def diff_drive2(self, robot, index, target_pose, js1, joint_target, js2):
        """
        This is a hackier version of the diff_drive
        It uses specified joints to achieve the target pose of the end effector
        while asking some other specified joints to match a global pose
        """
        pf = robot.get_compute_functions()['passive_force'](True, True, False)
        max_v = 0.1
        max_w = np.pi
        qpos, qvel, poses = robot.get_observation()
        current_pose: Pose = poses[index]
        delta_p = target_pose.p - current_pose.p
        delta_q = qmult(target_pose.q, qinverse(current_pose.q))

        axis, theta = quat2axangle(delta_q)
        if (theta > np.pi):
            theta -= np.pi * 2

        t1 = np.linalg.norm(delta_p) / max_v
        t2 = theta / max_w
        t = max(np.abs(t1), np.abs(t2), 0.001)
        thres = 0.1
        if t < thres:
            k = (np.exp(thres) - 1) / thres
            t = np.log(k * t + 1)
        v = delta_p / t
        w = theta / t * axis
        target_qvel = robot.get_compute_functions()['cartesian_diff_ik'](np.concatenate((v, w)), 9)
        for j, target in zip(js2, joint_target):
            qpos[j] = target
        robot.set_action(qpos, target_qvel, pf)

    def get_global_position_from_camera(self, camera, depth, x, y):
        """
        camera: an camera agent
        depth: the depth obsrevation
        x, y: the horizontal, vertical index for a pixel, you would access the images by image[y, x]
        """
        cm = camera.get_metadata()
        proj, model = cm['projection_matrix'], cm['model_matrix']
        w, h = cm['width'], cm['height']

        # get 0 to 1 coordinate for (x, y) coordinates
        xf, yf = (x + 0.5) / w, 1 - (y + 0.5) / h

        # get 0 to 1 depth value at (x,y)
        zf = depth[int(y), int(x)]

        # get the -1 to 1 (x,y,z) coordinates
        ndc = np.array([xf, yf, zf, 1]) * 2 - 1

        # transform from image space to view space
        v = np.linalg.inv(proj) @ ndc
        v /= v[3]

        # transform from view space to world space
        v = model @ v

        return v

    def pick_box(self, c):
        color, depth, segmentation = c.get_observation()

        np.random.shuffle(self.box_ids)
        for i in self.box_ids:
            m = np.where(segmentation == i)
            if len(m[0]):
                min_x = 10000
                max_x = -1
                min_y = 10000
                max_y = -1
                for y, x in zip(m[0], m[1]):
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
                return self.get_global_position_from_camera(c, depth, x, y)

        return False
    
    def pick_box1(self, c, thres_x=0.25, thres_y=0.4):
        color, depth, segmentation = c.get_observation()
        np.random.shuffle(self.box_ids)
        for i in self.box_ids:
            m = np.where(segmentation == i)
            if len(m[0]):
                min_x = 10000
                max_x = -1
                min_y = 10000
                max_y = -1
                for y, x in zip(m[0], m[1]):
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
                box_pos = self.get_global_position_from_camera(c, depth, x, y)
                if np.abs(box_pos[0] + 0.05) < thres_x and np.abs(box_pos[1]) < thres_y:
                    return box_pos, True
                else:
                    continue

        return [0, 0], False
    
    def locate_bin(self, camera):
        """
        Locate the bin's global position using the camera's segmentation and depth data.
        """
        _, depth, segmentation = camera.get_observation()
        bin_pixels = np.where(segmentation == self.bin_id)
        if len(bin_pixels[0]) == 0:
            raise ValueError("Bin not found in the camera's view.")

        # Calculate the center pixel of the bin
        x_center = int((bin_pixels[1].min() + bin_pixels[1].max()) / 2)
        y_center = int((bin_pixels[0].min() + bin_pixels[0].max()) / 2)

        # Convert the pixel to a global position
        bin_global_position = self.get_global_position_from_camera(camera, depth, x_center, y_center)
        return bin_global_position
    
    def compute_joint_velocity_from_twist(self, robot, twist: np.ndarray) -> np.ndarray:
        """
        This function calculates the joint velocities needed to achieve a given spatial twist at the end effector.

        robot: The robot being controlled
        twist: A 6-dimensional vector representing the spatial twist,(linear&angular velocity)
        """
        assert twist.size == 6
        # compute dense jacobian matrix
        dense_jacobian = robot.get_compute_functions()['spatial_twist_jacobian']()
        end_effector_jacobian = np.zeros([6, robot.dof])
        end_effector_index = 9
        end_effector_jacobian[:3, :] = dense_jacobian[end_effector_index * 6 - 3:end_effector_index * 6, :7]
        end_effector_jacobian[3:6, :] = dense_jacobian[(end_effector_index - 1) * 6:end_effector_index * 6 - 3, :7]
        # pseudo inverse of jacobian
        ee_jacobian_inverse = np.linalg.pinv(end_effector_jacobian)
        # twist to joint velocity
        joint_velocity = ee_jacobian_inverse @ twist
        return joint_velocity
        raise NotImplementedError


if __name__ == '__main__':
    np.random.seed(0)
    env = FinalEnv()
    env.run(Solution(), render=True, render_interval=5, debug=True)
    env.close()
