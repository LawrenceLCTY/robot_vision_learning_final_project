"""
Robot Vision and Learning: Final Project

Group members:
Lawrence Leroy Chieng Tze Yao 2401213369
Shui Jie 2401112104
Peterson Co 2401213365
"""

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
        self.fail_chances = 3 #chances allowed for empty catches
        
        self.useless_time = 0
        self.time_limit = 10000 #inefficient/useless time allowed
        self.sweep_start_pos = 0
        
    

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
                self.selected_x = None
                self.counter = 0

        if self.phase == 1:
            self.counter += 1

            if (self.counter == 1):
                selected = self.pick_box(c4)
                self.selected_x = selected[0]
                if self.selected_x is None: #No box to select
                    return False

            target_pose_left = Pose([self.selected_x, 0.5, 0.67], euler2quat(np.pi, -np.pi / 3, -np.pi / 2))
            self.diff_drive(r1, 9, target_pose_left)

            target_pose_right = Pose([self.selected_x, -0.5, 0.6], euler2quat(np.pi, -np.pi / 3, np.pi / 2))
            self.diff_drive(r2, 9, target_pose_right)

            if self.counter == 2000 / 5:
                self.phase = 2

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
            
            if self.counter == 4000 / 5:
                self.phase = 3
                
                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.1
                q = euler2quat(np.pi, -np.pi / 4.2, -np.pi / 2)
                self.pose_left = Pose(p, q)

                self.sweep_start_pos = np.copy(p)

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                self.start_pos = np.copy(p)
                p[1] += 0.01
                q = euler2quat(np.pi, -np.pi / 2.0, np.pi / 2)
                self.pose_right = Pose(p, q)
                
        if self.phase == 3:
            self.counter += 1
            if self.counter < 5000 / 5:  # Initial orientation phase
                #print("Phase 3a - Orienting") #debug
                self.diff_drive(r1, 9, self.pose_left)
                self.diff_drive(r2, 9, self.pose_right)
            elif self.counter < 5100/5:
                
                q = euler2quat(np.pi, -np.pi / 1.9, np.pi / 2)
                self.pose_right = Pose(self.start_pos, q)
                self.diff_drive(r2, 9, self.pose_right)
            
            elif self.counter < 5500 / 5:
                #print("Phase 3b - Sweeping Forward") #debug
                sweep_pose = Pose(
                    p=np.array([
                        self.sweep_start_pos[0],
                        self.sweep_start_pos[1] - 0.1,  
                        self.sweep_start_pos[2] - 0.1
                    ]),
                    q=self.pose_left.q 
                )
                self.diff_drive(r1, 9, sweep_pose)
                self.diff_drive(r2, 9, self.pose_right)
            
            elif self.counter < 6000 / 5: 
                #print("Phase 3c - Safety Adjustment") #debug
                t1 = [3, 1, 0, -1.5, -1, 1, -2]
                r1.set_action(t1, [0] * 7, pf_left)
                self.diff_drive(r2, 9, self.pose_right)

            elif is_spade_empty(self, c4, r2):
                self.phase = 0

            else:
                self.phase = 4

        if self.phase == 4:
            self.counter += 1
            if (self.counter < 7000 / 5):
                #print("Phase 4a") #debug
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.jacobian_drive(r2, 9, Pose(p, q))
                
            elif (self.counter < 9000 / 5):
                #print("Phase 4b") #debug
                p = self.target_bin_position.copy()
                p[2]+=1.0
                q = euler2quat(0, -np.pi / 3, 0)
                self.jacobian_drive(r2, 9, Pose(p, q))
                
            elif not self.is_spade_above_bin(r2, self.target_bin_position):
                #print("Phase 4c")
                p = self.target_bin_position.copy()
                p[1] -= 0.1
                q = euler2quat(0, -np.pi / 2, 0)
                self.jacobian_drive(r2, 9, Pose(p, q))
            else:
                #print("phase 4d")
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
                #reset penalty metrics since we are measuring continuous useleness
                self.useless_time = 0 
                self.fail_chances = 3
            else: #no boxes were picked
                self.fail_chances -= 1
                self.useless_time += self.counter
                print(f'Failed to place boxes in bin. Remaining chances: {self.fail_chances}') #debug
                print(f'Current time: {self.useless_time}; Time limit: {self.time_limit}')
                if self.fail_chances <= 0 or self.useless_time > self.time_limit: #too many failed chances or out of time
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

        # Get the global positions of all boxes
        box_positions = {}
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
                global_position = self.get_global_position_from_camera(c, depth, x, y)
                box_positions[i] = global_position

        # If no boxes are found, return False
        if not box_positions:
            return [None,None]

        # Count neighbors for each box
        neighbor_counts = {}
        reachable_boxes = []
        for box_id, position in box_positions.items():
            # Check if the box is reachable (within the robot's workspace)
            if is_box_reachable(position):  # Implement this function
                reachable_boxes.append(box_id)
                neighbor_counts[box_id] = 0
                for other_box_id, other_position in box_positions.items():
                    if box_id != other_box_id:
                        distance = np.linalg.norm(position - other_position)
                        if distance < 0.2:  # Neighbor threshold (adjust as needed)
                            neighbor_counts[box_id] += 1

        # If no reachable boxes, return False
        if not reachable_boxes:
            return [None,None]

        # Prioritize boxes with the most neighbors
        prioritized_boxes = sorted(reachable_boxes, key=lambda x: neighbor_counts[x], reverse=True)

        # Return the position of the box with the most neighbors
        best_box_id = prioritized_boxes[0]
        return box_positions[best_box_id]
   
    
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

    def jacobian_drive(self, robot, end_effector_index, target_pose, velocity_scale=0.3):
        """
        Move the robot's end effector to a target pose using Jacobian-based control.
        
        Args:
            robot: Robot instance to control
            end_effector_index: Index of the end effector
            target_pose: Target pose to reach
            velocity_scale: Scaling factor for movement speed (default: 0.3)
        """
        # Get current robot state
        robot_state = self._get_robot_state(robot, end_effector_index)
        
        # Calculate pose transformation
        pose_transform = self._calculate_pose_transform(
            robot_state.current_pose, 
            target_pose
        )
        
        # Compute body twist
        body_twist = self._compute_body_twist(pose_transform, velocity_scale)
        
        # Transform to spatial twist
        spatial_twist = self._transform_to_spatial_twist(
            body_twist,
            robot_state.rotation,
            robot_state.position
        )
        
        # Compute and set joint velocities
        self._set_robot_velocities(robot, robot_state, spatial_twist)

    def _get_robot_state(self, robot, end_effector_index):
        """
        Get the current state of the robot.
        
        Returns:
            RobotState: Named tuple containing robot state information
        """
        from collections import namedtuple
        RobotState = namedtuple('RobotState', 
            ['joint_positions', 'joint_velocities', 'current_pose', 
             'rotation', 'position', 'passive_force'])
        
        # Get robot state
        passive_force = robot.get_compute_functions()['passive_force'](True, True, False)
        joint_positions, joint_velocities, poses = robot.get_observation()
        
        # Convert current pose to matrix form
        current_pose_matrix = pose2mat(poses[end_effector_index])
        
        return RobotState(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            current_pose=current_pose_matrix,
            rotation=current_pose_matrix[:3, :3],
            position=current_pose_matrix[:3, 3],
            passive_force=passive_force
        )

    def _calculate_pose_transform(self, current_pose, target_pose):
        """
        Calculate the transformation between current and target poses.
        """
        target_pose_matrix = pose2mat(target_pose)
        return np.linalg.inv(current_pose) @ target_pose_matrix

    def _compute_body_twist(self, pose_transform, velocity_scale):
        """
        Compute the body twist from pose transformation.
        """
        twist, _ = pose2exp_coordinate(pose_transform)
        return twist * velocity_scale

    def _transform_to_spatial_twist(self, body_twist, rotation, position):
        """
        Transform body twist to spatial twist using adjoint matrix.
        """
        adjoint = self._create_adjoint_matrix(rotation, position)
        return adjoint @ body_twist

    def _create_adjoint_matrix(self, rotation, position):
        """
        Create the adjoint matrix for twist transformation.
        """
        adjoint = np.zeros((6, 6))
        adjoint[0:3, 0:3] = rotation
        adjoint[3:6, 3:6] = rotation
        adjoint[3:6, 0:3] = skew_symmetric_matrix(position) @ rotation
        return adjoint

    def _set_robot_velocities(self, robot, robot_state, spatial_twist):
        """
        Compute and set joint velocities based on spatial twist.
        """
        target_velocities = self.compute_joint_velocity_from_twist(robot, spatial_twist)
        robot.set_action(
            robot_state.joint_positions,
            target_velocities,
            robot_state.passive_force
        )

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
    
def is_spade_empty(self, camera, robot):
    color, depth, segmentation = camera.get_observation()
    
    # First get the global positions of all boxes
    box_positions = {}
    for box_id in self.box_ids:
        #print(box_id)
        m = np.where(segmentation == box_id)
        if len(m[0]):
            min_x = np.min(m[1])
            max_x = np.max(m[1])
            min_y = np.min(m[0])
            max_y = np.max(m[0])
            x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
            global_pos = self.get_global_position_from_camera(camera, depth, x, y)
            if global_pos is not None:
                print(box_id)
                box_positions[box_id] = global_pos

    # Check for elevated boxes
    surface_height = 0.61  # Known surface height
    threshold = 0.03
    print(len(box_positions.items()))
    for box_id, pos in box_positions.items():
        if (is_box_reachable(pos) and pos[2] > (surface_height + threshold)):  # z-axis check
            print("elevated and reachable: ",box_id)
            print(pos)
            return False
            
    return True

def is_box_reachable(box_position):
        """
        Check if the box is within the robot's workspace.
        Args:
            box_position: 3D global position of the box.
        Returns:
            bool: True if the box is reachable, False otherwise.
        """
        # Define the robot's workspace limits (adjust as needed)
        workspace_limits = {
            'x': [-0.30, 0.19],  # X-axis limits
            'y': [-0.40, 0.40],  # Y-axis limits
            'z': [0.0, 0.64]    # Z-axis limits
        }

        # Check if the box is within the workspace limits
        within_x = workspace_limits['x'][0] <= box_position[0] <= workspace_limits['x'][1]
        within_y = workspace_limits['y'][0] <= box_position[1] <= workspace_limits['y'][1]
        #within_z = workspace_limits['z'][0] <= box_position[2] <= workspace_limits['z'][1]

        #return within_x and within_y and within_z
        return within_x and within_y
    
if __name__ == '__main__':
    np.random.seed(100)
    env = FinalEnv()
    env.run(Solution(), render=True, render_interval=5, debug=True)
    env.close()
