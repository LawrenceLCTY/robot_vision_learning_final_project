# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang
import math

from final_env import FinalEnv, SolutionBase
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2axangle, qmult, qinverse
from scipy.spatial.distance import cdist
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
from sapien.core import Pose, PxrMaterial, OptifuserConfig, SceneConfig
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2axangle, qmult, qinverse
import time
import psutil
import os
import datetime
from sac_final_env import FinalEnv
# from baseline import Solution
# from solution import Solution
from SAC.agent import SAC
import time
import psutil
import numpy as np
# from solution_SAC import Solution


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
    term1 = theta_norm * np.eye(3)  # Iθ
    term2 = (1 - np.cos(theta_norm)) * omega_skew  # (1-cos θ)[ω]
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
        magnitude = np.sqrt(np.sum(v ** 2))  # different way to compute norm
        return np.concatenate([w, v / magnitude]), magnitude

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
    result[:3] = w / magnitude
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
    q = pose.q  # Quaternion [w,x,y,z]

    # Extract quaternion components
    w, x, y, z = q
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])

    # Ensure orthogonality (clean up numerical errors)
    U, _, Vh = np.linalg.svd(R)
    R = U @ Vh

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    return T


global_running_reward = 0


class SAC_agent:
    def __init__(self, env:FinalEnv):

        # to keep track of states for training
        self.prev_state = None
        self.cur_state = None
        self.episode = 0
        self.start_time = time.time()
        self.action = None



        self.box_id_on_left_spade = []
        self.box_id_on_right_spade = []


        # initialise robot
        MAX_EPISODES = 20000
        memory_size = 1e+6
        batch_size = 64
        gamma = 0.99
        alpha = 1
        lr = 3e-4

        n_actions = 6  # 7 values per arm (3 position)
        # Action bounds
        position_bounds = [0, 1]  # Normalized position bounds

        # Create bounds array for all actions
        action_bounds = np.array([
            # Lower bounds (first row)
            [position_bounds[0]] * 3 +  # Left arm
            [position_bounds[0]] * 3  # Right arm
            ,

            # Upper bounds (second row)
            [position_bounds[1]] * 3 +  # Left arm
            [position_bounds[1]] * 3  # Right arm
            ,
        ])

        self.sac_agent = SAC(env_name="SAPIEN",
                             n_states=self.get_env_robot_state(env).shape[0],
                             n_actions=6,
                             memory_size=memory_size,
                             batch_size=batch_size,
                             gamma=gamma,
                             alpha=alpha,
                             lr=lr,
                             action_bounds=action_bounds,
                             reward_scale=1.0)

        log_path = "sapien_logs"
        new_log_path = log_path
        counter = 0
        while os.path.exists(new_log_path):
            new_log_path = log_path + str(counter) + "/"
            counter += 1

        self.writer = SummaryWriter(new_log_path)

        # Other initialization code...

    def update_box_on_spade(self, box_id_on_left_spade, box_id_on_right_spade):
        # to keep track of current box id on spade
        self.box_id_on_left_spade = box_id_on_left_spade
        self.box_id_on_right_spade = box_id_on_right_spade

    def load_weight(self, path):
        # not training, load weights
        # self.sac_agent.load_weights("SAPIEN_02_box_smallmove_collision_near_to_bin.pth")
        self.sac_agent.load_weights(path)

    def log(self, episode, start_time, episode_reward, value_loss, q_loss, policy_loss):
        """
        Log training metrics to TensorBoard
        """
        # Add scalar values to tensorboard
        self.writer.add_scalar("Value Loss", value_loss, episode)
        self.writer.add_scalar("Q-Value Loss", q_loss, episode)
        self.writer.add_scalar("Policy Loss", policy_loss, episode)
        self.writer.add_scalar("Episode Reward", episode_reward, episode)

        # Optional: Calculate and log training time
        elapsed_time = time.time() - start_time
        self.writer.add_scalar("Training Time (s)", elapsed_time, episode)

        # Optional: Flush to ensure writing to disk
        self.writer.flush()

    #################  REWARD FUNCTIONS   #######################################
    def get_similarity_score(self, arr1, arr2):
        """
        Calculate similarity score between two numpy arrays on a scale of 0-5.
        5: Arrays are identical
        0: Arrays are very different

        Parameters:
        arr1, arr2: numpy arrays to compare

        Returns:
        float: similarity score from 0 to 5
        """
        # Check if arrays have same shape
        if arr1.shape != arr2.shape:
            return 0.0

        # Calculate normalized mean absolute difference
        max_val = max(np.max(np.abs(arr1)), np.max(np.abs(arr2)))
        if max_val == 0:
            return 5.0 if np.array_equal(arr1, arr2) else 0.0

        abs_diff = np.abs(arr1 - arr2)
        mean_diff = np.mean(abs_diff) / max_val

        # Convert to 0-5 scale using exponential decay
        # The factor -3 makes it more sensitive to small differences
        score = 5.0 * np.exp(-3 * mean_diff)

        return np.round(score, 2)


    def check_robot_collisions(self, robot_states, collision_threshold=0.1):
        """
        Check for collisions between robot arms using their SAC input states.

        Args:
            robot_states: List of SAC input states for each robot
            collision_threshold: Minimum distance threshold for collision detection

        Returns:
            int: Number of collision points detected
        """
        num_robots = len(robot_states)
        collision_count = 0

        # Extract positions and rotations from each robot state
        # SAC input format: [qpos(7), qvel(7), position(3), rotation_flat(9)]
        positions = []
        rotations = []

        for state in robot_states:
            # Position starts at index 14 (after qpos and qvel)
            pos = state[14:17]
            # Rotation matrix starts at index 17
            rot = state[17:].reshape(3, 3)
            positions.append(pos)
            rotations.append(rot)

        # Define key points along robot arm (simplified representation)
        # Using joint positions relative to base
        relative_points = np.array([
            [0, 0, 0],  # Base
            [0, 0, 0.333],  # Shoulder
            [0, 0, 0.666],  # Elbow
            [0, 0, 1.0]  # End effector
        ])

        # Calculate absolute positions of key points for each robot
        robot_points = []
        for i in range(num_robots):
            # Transform relative points to global coordinates
            global_points = np.dot(relative_points, rotations[i].T) + positions[i]
            robot_points.append(global_points)

        # Check distances between all pairs of robots
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                # Calculate distances between all points of robot i and robot j
                distances = cdist(robot_points[i], robot_points[j])

                # Count points that are closer than the threshold
                collisions = np.sum(distances < collision_threshold)
                collision_count += collisions

        return collision_count


    def check_multiple_robot_collisions(self, robots):
        """
        Check collisions between multiple robots.

        Args:
            robots: List of robot objects
        Returns:
            int: Total number of collision points
        """
        # Get SAC input states for all robots
        robot_states = []
        for robot in robots:
            sac_input = self.sapien_robot_to_sac_input(robot)
            robot_states.append(sac_input)

        # Check for collisions
        return self.check_robot_collisions(robot_states)


    def get_collision_punishment(self, env: FinalEnv):
        # punish for colliding
        r1, r2, _, _, _, _ = env.get_agents()

        robots = [r1, r2]  # However you access your robots
        collision_points = self.check_multiple_robot_collisions(robots)
        return collision_points


    def get_small_move_punishment(self):
        # puunish for making too small movement
        return self.get_similarity_score(self.prev_state, self.cur_state)

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

    def get_box_positions(self, c, box_ids):
        color, depth, segmentation = c.get_observation()

        # Get the global positions of all boxes
        box_positions = {}
        for i in box_ids:
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

        return box_positions

    def locate_bin(self, env:FinalEnv, z_offset=0.0):
        """
        Locate the bin's global position using the camera's segmentation and depth data.
        """
        if env.bin is None:
            raise ValueError("Bin has not been created yet.")

        # Get the bin's global position
        bin_pose = env.bin.get_pose()
        bin_center = bin_pose.p  # Center of the bin (x, y, z)

        # Apply z-offset
        bin_center[2] += z_offset

        return bin_center

    def dist_to_bin(self, env: FinalEnv):
        r1, r2, _, _, _, camera = env.get_agents()

        bin_location = self.locate_bin(env)
        dist = 0
        # print(f"bin_location {bin_location}")
        for i in range(3):
            dist += math.pow(r1.get_observation()[2][9].p[i] - bin_location[i], 2)
            dist += math.pow(r2.get_observation()[2][9].p[i] - bin_location[i], 2)
        return dist

    def get_box_on_spade(self, env:FinalEnv):
        r1, r2, c1, c2, c3, c4 = env.get_agents()
        meta = env.get_metadata()
        box_ids = meta['box_ids']
        left_end_effector_pos = r1.get_observation()[2][9].p
        right_end_effector_pos = r2.get_observation()[2][9].p

        box_positions = self.get_box_positions(c4, box_ids)

        box_count = 0

        for box_id, position in box_positions.items():
            if box_id in self.box_id_on_left_spade:
                dist = 0
                for i in range(3):
                    dist += (position[i] - left_end_effector_pos[i]) ** 2
                # print(f"id: {box_id} is dist {dist} from left arm ")
                if dist < 0.02:
                    box_count += 1

            if box_id in self.box_id_on_right_spade:
                dist = 0
                for i in range(3):
                    dist += (position[i] - right_end_effector_pos[i]) ** 2
                # print(f"id: {box_id} is dist {dist} from left arm ")
                if dist < 0.02:
                    box_count += 1
        return box_count

    def get_close_to_bin_reward(self, env: FinalEnv):

        return - self.dist_to_bin(env) * 0.5


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
        bin_position = target_bin_position[:2]  # (x, y)

        # Define a small tolerance for overlap
        tolerance = 0.15
        in_position = np.allclose(spade_position, bin_position, atol=tolerance)
        return in_position


    def get_on_bin_reward(self, env: FinalEnv):
        r1, r2, _, _, _, _ = env.get_agents()
        target_bin_position = self.locate_bin(env)
        reward = 0
        if self.is_spade_above_bin(r1, target_bin_position):
            reward += 10
        if self.is_spade_above_bin(r2, target_bin_position):
            reward += 10

        return reward

    def get_reward(self, env: FinalEnv):
        """
        Change funciton to test different rewards
        Need to promote the sapien to move towards the bin
        Need to promote box stay on spade
        need to punish if movement is small
        need to punish if coll
        """
        return self.get_box_on_spade(env) + \
            self.get_close_to_bin_reward(env) - \
            self.get_collision_punishment(env) - \
            self.get_small_move_punishment() + \
            self.get_on_bin_reward(env)

    ########################################################################


    ################# STATE FUNCTIONS ########################################

    def sapien_robot_to_sac_input(self, robot):

        qpos, qvel, poses = robot.get_observation()

        # Convert qpos and qvel to numpy arrays if they aren't already
        qpos = np.array(qpos)
        qvel = np.array(qvel)

        # Extract the final pose (end effector)
        final_pose = poses[-1]
        position = np.array(final_pose.p)  # Global position vector

        # Convert quaternion [w,x,y,z] to rotation matrix
        rotation_matrix = R.from_quat(final_pose.q).as_matrix()

        # Flatten the rotation matrix
        rotation_flat = rotation_matrix.flatten()

        # Combine all components into a single observation vector for SAC
        sac_input = np.concatenate([
            qpos,  # Joint positions (7)
            qvel,  # Joint velocities (7)
            position,  # Global position (3)
            rotation_flat,  # Flattened rotation matrix (9)
        ])

        return sac_input

    def box_to_sac_input(self, env: FinalEnv):
        """
        return a 1D dim 30 array to represent 10 boxes location
        """
        r1, r2, c1, c2, c3, c4 = env.get_agents()
        meta = env.get_metadata()
        box_ids = meta['box_ids']
        box_location = self.get_box_positions(c4, box_ids)
        flattened_arrays = []
        for key, arr in box_location.items():
            flattened_arrays.extend(arr.flatten()[:3])

        while len(flattened_arrays) < 30:
            flattened_arrays.append(0)

        return flattened_arrays

    def get_env_robot_state(self, env: FinalEnv):
        left_robot, right_robot, _, _, _, camera = env.get_agents()
        left_input = self.sapien_robot_to_sac_input(left_robot)
        right_input = self.sapien_robot_to_sac_input(right_robot)
        box_input = self.box_to_sac_input(env)


        # Combine all components
        combined_input = np.concatenate([
            left_input,  # 26 dims (7 qpos + 7 qvel + 3 pos + 9 rot)
            right_input,  # 26 dims (7 qpos + 7 qvel + 3 pos + 9 rot)
            self.locate_bin(env),  # 3 dims (x,y,z)
            box_input #30 dim: 10 boxes 3d coordinates
        ])

        # Final shape should be (64,): 26 + 26 + 9 + 3
        return combined_input

    #########################################################################

    #################  ACTION FUNCTIONS #####################################

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

    def action_to_poses(self, action):
        """Convert SAC action to target poses for both arms
        SAC output 6 values for x,y,z of end effector for both arms
        """
        # Denormalize actions if they were normalized
        # Split action into left and right arm components
        left_action = action[:3]
        right_action = action[3:]

        # Convert to poses
        left_pose = Pose(
            p=left_action,  # position
            q=euler2quat(0, -np.pi / 3, 0)  # quaternion
        )
        right_pose = Pose(
            p=right_action,
            q=euler2quat(0, -np.pi / 3, 0)
        )

        return left_pose, right_pose

    def apply_action(self, env: FinalEnv, action):
        """Apply SAC action to both robots"""

        left_target_pose, right_target_pose = self.action_to_poses(action)

        left_robot, right_robot, _, _, _, _ = env.get_agents()

        # Apply diff_drive to both arms
        self.jacobian_drive(left_robot, 9, left_target_pose)
        self.jacobian_drive(right_robot, 9, right_target_pose)

    def is_goal(self, env: FinalEnv):
        # both spade on bin
        return self.get_on_bin_reward(env) == 20

    def update_tajectory(self, env: FinalEnv, action):
        """
        Generate prexisting trajectories into the agent as guide
        """
        # update current state
        self.prev_state = self.cur_state
        self.cur_state = self.get_env_robot_state(env)

        if self.prev_state is not None:
            reward = self.get_reward(env)

            self.sac_agent.store(self.prev_state, reward, self.is_goal(env), self.action, self.cur_state)

            print(f"current trajectory reward: {reward} for action {action}")

        self.action =  action

        self.episode += 1

        if self.episode % 32 == 0:
            print(f"train model at ep: {self.episode}")
            self.sac_agent.train()
            self.sac_agent.save_weights()


    def run(self, env: FinalEnv, is_train=False):

        # update current state
        self.prev_state = self.cur_state
        self.cur_state = self.get_env_robot_state(env)

        if self.prev_state is not None:
            reward = self.get_reward(env)
            self.sac_agent.store(self.prev_state, reward, self.is_goal(env), self.action, self.cur_state)

            if is_train and self.episode % 64 == 0:
                value_loss, q_loss, policy_loss = self.sac_agent.train()

                if self.prev_state is not None:

                    print(f"episode: {self.episode}  current reward: {reward} value_loss {value_loss} policy_loss {policy_loss} q_loss {q_loss}")

                    self.log(self.episode, self.start_time, reward, value_loss, q_loss, policy_loss)


        if self.prev_state is None:
            self.prev_state = self.cur_state

        # new action
        self.action = self.sac_agent.choose_action(self.cur_state)

        # apply
        self.apply_action(env, self.action)

        self.episode += 1
        if self.episode % 400 == 0:
            print(
                f"EP:{self.episode}| Saved weights")
            self.sac_agent.save_weights()

        return self.episode





