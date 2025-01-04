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
        self.env = env
        meta = env.get_metadata()
        self.box_ids = meta['box_ids']
        r1, r2, c1, c2, c3, c4 = env.get_agents()

        self.ps = [1000, 800, 600, 600, 200, 200, 100]
        self.ds = [1000, 800, 600, 600, 200, 200, 100]
        r1.configure_controllers(self.ps, self.ds)
        r2.configure_controllers(self.ps, self.ds)
        
        #determine target bin coordinates
        self.target_bin_position = self.locate_bin(z_offset=0.5)
        
        self.total_box_picked = 0 #placeholder for boxed picked
        self.fail_chances = 1 #chances allowed for empty catches
        
    

    def act(self, env: FinalEnv, current_timestep: int):
        r1, r2, c1, c2, c3, c4 = env.get_agents()

        pf_left = f = r1.get_compute_functions()['passive_force'](True, True, False)
        pf_right = f = r2.get_compute_functions()['passive_force'](True, True, False)

        if self.phase == 0:
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
                selected = self.pick_box(c4)
                self.selected_x = selected[0]
                if self.selected_x is None:
                    return False

            target_pose_left = Pose([self.selected_x, 0.5, 0.67], euler2quat(np.pi, -np.pi / 3, -np.pi / 2))
            self.diff_drive(r1, 9, target_pose_left)

            target_pose_right = Pose([self.selected_x, -0.5, 0.6], euler2quat(np.pi, -np.pi / 3, np.pi / 2))
            self.diff_drive(r2, 9, target_pose_right)

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
            
            if self.counter == 2000 / 5:
                self.phase = 3
                
                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.1
                q = euler2quat(np.pi, -np.pi / 4, -np.pi / 2)
                self.pose_left = Pose(p, q)

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.0
                q = euler2quat(np.pi, -np.pi / 1.8, np.pi / 2)
                self.pose_right = Pose(p, q)

                self.counter = 0

        if self.phase == 3:
            self.counter += 1
            if self.counter < 500 / 5:
                print("Phase 3a") #debug
                self.diff_drive(r1, 9, self.pose_left)
                self.diff_drive(r2, 9, self.pose_right)
            elif self.counter < 1500 / 5:
                print("Phase 3b") #debug
                t1 = [3, 1, 0, -1.5, -1, 1, -2]
                r1.set_action(t1, [0] * 7, pf_left)
                self.diff_drive(r2, 9, self.pose_right)
            else:
                self.phase = 4
                self.counter = 0


        if self.phase == 4:
            self.counter += 1
            if (self.counter < 1000 / 5):
                print("Phase 4a") #debug
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.jacobian_drive(r2, 9, Pose(p, q))
                
            elif (self.counter < 3000 / 5):
                print("Phase 4b") #debug
                p = self.target_bin_position.copy()
                p[2]+=1.0
                q = euler2quat(0, -np.pi / 3, 0)
                self.jacobian_drive(r2, 9, Pose(p, q))
                
            elif not self.is_spade_above_bin(r2, self.target_bin_position):
                print("Phase 4c")
                p = self.target_bin_position.copy()
                p[1] -= 0.1
                q = euler2quat(0, -np.pi / 2, 0)
                self.jacobian_drive(r2, 9, Pose(p, q))
            else:
                print("phase 4d")
                # Initialize rotation state if it doesn't exist
                if not hasattr(self, "_rotation_step"):
                    self._rotation_step = 0
                    self._total_steps = 36
                    self._delay_multiplier = 10
                    self._delay_counter = 0
                    self._initial_qpos = r2.get_observation()[0]  
                    
                if self._delay_counter < self._delay_multiplier:
                    self._delay_counter += 1
                else:
                    self._delay_counter = 0
                    self._rotation_step += 1
                    
                    self.rotate_spade(r2, self._initial_qpos, self._rotation_step, self._total_steps)
                
                print(f'Rotation step: {self._rotation_step}')
                    
                if self._rotation_step >= self._total_steps:
                    del self._rotation_step
                    del self._total_steps
                    del self._delay_multiplier
                    del self._delay_counter
                    del self._initial_qpos
                    self.phase += 1  
                                        
        if self.phase == 5: #End phase: either continue simulation or end scene
            print('Phase 5')
            self.phase = 0
            
            score = env.get_reward()
            if score > self.total_box_picked: #current score is greater than previous score
                self.total_box_picked = score #copies current score
            else: #no boxes were picked
                print(f'Failed to place boxes in bin. Remaining chances: {self.fail_chances}') #debug
                self.fail_chances -= 1
                if self.fail_chances < 0:
                    return False #end scene
            

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
    
    def locate_bin(self, z_offset=0.0):
        """
        Locate the bin's global position using the camera's segmentation and depth data.
        """
        if self.env.bin is None:
            raise ValueError("Bin has not been created yet.")
    
        # Get the bin's global position
        bin_pose = self.env.bin.get_pose()
        bin_center = bin_pose.p  # Center of the bin (x, y, z)

        # Apply z-offset
        bin_center[2] += z_offset

        return bin_center

    def move_to_bin(self, robot, target_bin_position):
        """
        Move the robot arm close to the bin without touching it.
        """
        approach_orientation = euler2quat(0, -np.pi / 2, 0)
        target_pose = Pose(target_bin_position, approach_orientation)
        self.diff_drive(robot, index=9, target_pose=target_pose)

    def is_spade_above_bin(self, robot, target_bin_position):
        """
        Checks if the spade (last joint of the robot) is directly above the bin
        based on the x and y coordinates.

        Parameters:
            robot: The robot agent (e.g., r1 or r2).
            target_bin_position: bin position that the robot should arrive at
        Returns:
            bool: True if the spade is above the bin, otherwise False.
        """
        # Get the pose of the last joint
        last_joint_index = robot.dof - 1
        qpos, _, poses = robot.get_observation()
        last_joint_pose: Pose = poses[last_joint_index]

        # Extract the x, y positions
        spade_position = last_joint_pose.p[:2]  # (x, y)
        bin_position = target_bin_position[:2]    # (x, y)

        # Define a small tolerance for overlap
        tolerance = 0.15
        in_position = np.allclose(spade_position, bin_position, atol=tolerance)
        return in_position


    def rotate_spade(self, robot, initial_qpos, rotation_step, total_steps, clockwise=True):
        # Rotate the joint
        direction = 1 if clockwise else -1
        qpos, _, _ = robot.get_observation()
        last_joint_index = robot.dof - 1  
        qpos[last_joint_index] = initial_qpos[last_joint_index] + direction * (2 * np.pi) * (rotation_step / total_steps)

        drive_target = qpos
        drive_velocity = [0] * robot.dof  
        additional_force = [0] * robot.dof 
        robot.set_action(drive_target, drive_velocity, additional_force)


    def jacobian_drive1(self, robot, index, target_pose):
        """
        Implementation of differential drive using twist differential IK.
        This approach constructs a body twist to move from current pose to target pose,
        then uses the robot's twist differential IK to compute joint velocities.
        
        Args:
            robot: Robot instance with compute functions
            index: Index of the end effector
            target_pose: Target Pose object containing position and orientation
        """
        # Get passive force compute function
        pf = robot.get_compute_functions()['passive_force'](True, True, False)
        
        # Maximum linear and angular velocities
        max_v = 0.1  # m/s
        max_w = np.pi  # rad/s
        
        # Get current robot state
        qpos, qvel, poses = robot.get_observation()
        current_pose: Pose = poses[index]
        
        # Compute position error in world frame
        delta_p = target_pose.p - current_pose.p
        
        # Get transformation matrices
        current_T = pose2mat(current_pose)
        target_T = pose2mat(target_pose)
        
        # Compute relative transformation
        # T_rel = inv(current_T) @ target_T
        T_rel = np.linalg.inv(current_T) @ target_T
        
        # Extract rotation matrix and position from relative transform
        R_rel = T_rel[:3, :3]
        p_rel = T_rel[:3, 3]
        
        # Convert rotation matrix to axis-angle
        # Using the fact that: tr(R) = 1 + 2cos(θ)
        theta = np.arccos((np.trace(R_rel) - 1) / 2)
        
        if abs(theta) < 1e-10:
            axis = np.array([0., 0., 1.])  # Default axis if no rotation
        else:
            # axis = [R32-R23, R13-R31, R21-R12] / (2sin(θ))
            axis = np.array([
                R_rel[2,1] - R_rel[1,2],
                R_rel[0,2] - R_rel[2,0],
                R_rel[1,0] - R_rel[0,1]
            ]) / (2 * np.sin(theta))
            axis = axis / np.linalg.norm(axis)
        
        # Normalize angle to [-π, π]
        if theta > np.pi:
            theta -= np.pi * 2
        
        # Compute time scaling based on maximum velocities
        t1 = np.linalg.norm(p_rel) / max_v
        t2 = np.abs(theta) / max_w
        t = max(t1, t2, 0.001)
        
        # Apply smoothing for small time values
        thres = 0.1
        if t < thres:
            k = (np.exp(thres) - 1) / thres
            t = np.log(k * t + 1)
        
        # Compute linear and angular velocities in body frame
        v_body = p_rel / t
        w_body = axis * (theta / t)
        
        # Construct body twist vector [ω, v]
        twist = np.concatenate((w_body, v_body))
        
        # Compute joint velocities using twist differential IK
        target_qvel = robot.get_compute_functions()['twist_diff_ik'](twist, index)
        
        # Set robot action
        robot.set_action(qpos, target_qvel, pf)

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
    
    
    def is_collision_detected(current_pose, bin_center, threshold=0.05):
        """
        Check if the spade is too close to the bin.
        Args:
            current_pose: Current pose of the spade.
            bin_center: Center of the bin.
            threshold: Minimum safe distance (default: 0.05 meters).
        Returns:
            True if a collision is detected, False otherwise.
        """
        distance = np.linalg.norm(current_pose.p - bin_center)
        return distance < threshold
    
    def get_offset(self,current_pose, bin_center):
        offset_x = bin_center[0] - current_pose[0]
        offset_y = bin_center[1] - current_pose[1]
        print("offset_y",offset_y)
        target_position = [
            current_pose[0],
            current_pose[1] + offset_y,
            current_pose[2]
        ]

        return target_position
    
    def locate_spade_length(self, camera):
        """
        Calculate the length of the spade using the camera's segmentation and depth data.
        """
        _, depth, segmentation = camera.get_observation()

        # Get the spade's ID from the robot's metadata
        r_meta = env.get_agents()[1].get_metadata()  # Assuming r2 is the robot with the spade
        spade_id = r_meta['link_ids'][-1]  # Spade is typically the last link

        # Find the pixels belonging to the spade
        spade_pixels = np.where(segmentation == spade_id)
        if len(spade_pixels[0]) == 0:
            raise ValueError("Spade not found in the camera's view.")

        # Find the base and tip of the spade
        # Base: Pixel closest to the robot's arm (assume the base is at the top of the image)
        base_pixel = (spade_pixels[0].min(), spade_pixels[1][np.argmin(spade_pixels[0])])
        
        # Tip: Pixel farthest from the base (assume the tip is at the bottom of the image)
        tip_pixel = (spade_pixels[0].max(), spade_pixels[1][np.argmax(spade_pixels[0])])

        # Convert the base and tip pixels to global coordinates
        base_position = self.get_global_position_from_camera(camera, depth, base_pixel[1], base_pixel[0])
        tip_position = self.get_global_position_from_camera(camera, depth, tip_pixel[1], tip_pixel[0])

        # Compute the Euclidean distance between the base and tip
        spade_length = np.linalg.norm(base_position - tip_position)
        return spade_length
        

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

if __name__ == '__main__':
    np.random.seed(0)
    env = FinalEnv()
    env.run(Solution(), render=True, render_interval=5, debug=True)
    env.close()
