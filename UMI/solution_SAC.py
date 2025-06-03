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
from solution import Solution
from SAC.agent import SAC
import time
import psutil
import numpy as np

global_running_reward = 0

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
        
        #set target bin position for r1
        self.r1_target_bin_position = self.bin_position.copy()
        self.r1_target_bin_position[0] += 0.1  # Offset X-axis towards center of bin
        self.r1_target_bin_position[1] += 0.15 # Offset Y-axis towards center of bin
        self.r1_target_bin_position[2] += 0.5  # Offset in Z-axis for safety
        
        #set target bin position for r2
        self.r2_target_bin_position = self.bin_position.copy()
        self.r2_target_bin_position[0] += 0.1  # Offset X-axis towards center of bin
        self.r2_target_bin_position[1] -= 0.1  # Offset Y-axis towards center of bin
        self.r2_target_bin_position[2] += 0.4  # Offset in Z-axis for safety

        MAX_EPISODES = 20000
        memory_size = 1e+6
        batch_size = 256
        gamma = 0.99
        alpha = 1
        lr = 3e-4

        n_actions = 15  # 7 values per arm (3 position + 4 quaternion) + 1 plus if decide to hard action
        # Action bounds
        position_bounds = [0, 1]  # Normalized position bounds
        quaternion_bounds = [-1, 1]  # Normalized quaternion bounds

        # Create bounds array for all actions
        action_bounds = np.array([
            # Lower bounds (first row)
            [position_bounds[0]] * 3 + [quaternion_bounds[0]] * 4 +  # Left arm
            [position_bounds[0]] * 3 + [quaternion_bounds[0]] * 4 +  # Right arm
            [-1],

            # Upper bounds (second row)
            [position_bounds[1]] * 3 + [quaternion_bounds[1]] * 4 +  # Left arm
            [position_bounds[1]] * 3 + [quaternion_bounds[1]] * 4 +  # Right arm
            [1],
        ])

        self.sac_agent = SAC(env_name="SAPIEN",
                    n_states=self.get_env_robot_state(env).shape[0],
                    n_actions=n_actions,
                    memory_size=memory_size,
                    batch_size=batch_size,
                    gamma=gamma,
                    alpha=alpha,
                    lr=lr,
                    action_bounds=action_bounds,
                    reward_scale=1.0)

        # use weights
        self.sac_agent.load_weights()


        self.box_counter = 0
        self.initiate_training = True
        self.state = None
        self.action = None
        self.next_state = None
        self.episode_reward = 0
        self.start_time = None
        self.episode, self.value_loss, self.q_loss, self.policy_loss, self.distance_to_bin = 0, 0, 0, 0, 0



    ###################### SAC related code #########################################

    def log(self, episode, start_time, episode_reward, value_loss, q_loss, policy_loss, memory_length):
        with SummaryWriter("sapien_logs/") as writer:
            writer.add_scalar("Value Loss", value_loss, episode)
            writer.add_scalar("Q-Value Loss", q_loss, episode)
            writer.add_scalar("Policy Loss", policy_loss, episode)

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

    def get_env_robot_state(self, env: FinalEnv):
        left_robot, right_robot, _, _, _, camera = env.get_agents()
        left_input = self.sapien_robot_to_sac_input(left_robot)
        right_input = self.sapien_robot_to_sac_input(right_robot)


        # Combine all components
        combined_input = np.concatenate([
            left_input,  # 26 dims (7 qpos + 7 qvel + 3 pos + 9 rot)
            right_input,  # 26 dims (7 qpos + 7 qvel + 3 pos + 9 rot)
            self.bin_position,  # 3 dims (x,y,z)
        ])

        # Final shape should be (64,): 26 + 26 + 9 + 3
        return combined_input

    def action_to_poses(self, action):
        """Convert SAC action to target poses for both arms"""
        # Denormalize actions if they were normalized
        # Split action into left and right arm components
        left_action = action[:7]
        right_action = action[7:14]

        # Convert to poses
        left_pose = Pose(
            p=left_action[:3],  # position
            q=left_action[3:]  # quaternion
        )
        right_pose = Pose(
            p=right_action[:3],
            q=right_action[3:]
        )

        return left_pose, right_pose

    def apply_action(self, env: FinalEnv, action):
        """Apply SAC action to both robots"""

        if action[-1] > 0:
            return True

        left_target_pose, right_target_pose = self.action_to_poses(action)

        left_robot, right_robot, _, _, _, _ = env.get_agents()

        # Apply diff_drive to both arms
        self.diff_drive(left_robot, 9, left_target_pose)  # Assuming index 9 is end-effector
        self.diff_drive(right_robot, 9, right_target_pose)

        return False

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

    import numpy as np
    from scipy.spatial.distance import cdist

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

    # Example usage:
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
        return self.get_similarity_score(self.state, self.next_state)


    def get_reward_box(self, env: FinalEnv):
        """
        If more box in bin, reward 100 for new box, else -1 for using time
        """
        reward = 0
        current_box = env.get_reward()
        if current_box > self.box_counter:
            new_box = current_box - self.box_counter
            self.box_counter = current_box
            reward += new_box * 100
        else:
            reward -= 0.1

        return reward

    def dist_to_bin(self, env: FinalEnv):
        r1, r2, _, _, _, camera = env.get_agents()

        bin_location = self.locate_bin(camera)
        dist = 0
        print(f"bin_location {bin_location}")
        for i in range(3):
            dist += math.pow(r1.get_observation()[2][9].p[i] - bin_location[i], 2)
            dist += math.pow(r2.get_observation()[2][9].p[i] - bin_location[i], 2)
        return dist

    def get_close_to_bin_reward(self, env: FinalEnv):

        return (1 - self.dist_to_bin(env) / self.distance_to_bin) * 10



    def get_reward(self, env: FinalEnv):
        """
        Change funciton to test different rewards
        Need to promote the sapien to move towards the bin
        Needs to decrease the punishment for time
        """




        return self.get_reward_box(env) - 0.1 * self.get_small_move_punishment()\
            - 0.1 * self.get_collision_punishment(env)

    ############################################################################
        
    

    def act(self, env: FinalEnv, current_timestep: int):
        r1, r2, c1, c2, c3, c4 = env.get_agents()

        pf_left = f = r1.get_compute_functions()['passive_force'](True, True, False)
        pf_right = f = r2.get_compute_functions()['passive_force'](True, True, False)

        if self.phase == 0:

            # phrase zero re initiate
            if not self.initiate_training:
                self.initiate_training = True

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
            # print(f"arm first pose: {target_pose_left.p}")
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

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.2
                self.pose_right = Pose(p, q)

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = 0.5
                q = euler2quat(np.pi, -np.pi / 2, -np.pi / 2)



                self.pose_left = Pose(p, q)

                self.counter = 0

        if self.phase == 3:
            self.counter += 1
            self.diff_drive(r1, 9, self.pose_left)
            self.diff_drive(r2, 9, self.pose_right)
            if self.counter == 200 / 5:
                self.phase = 4
                self.counter = 0

        if self.phase == 4:
            self.counter += 1
            if (self.counter < 1000):

                # use this to train SAC agent
                # new state and action



                # new state given previous action
                self.next_state = self.get_env_robot_state(env)

                if self.initiate_training:
                    self.episode += 1
                    self.start_time = time.time()
                    self.initiate_training = False
                    # first get current state
                    self.state = self.get_env_robot_state(env)
                    self.action = self.sac_agent.choose_action(self.state)
                    self.apply_action(env, self.action)
                    self.episode_reward = 0
                    self.distance_to_bin = self.dist_to_bin(env)
                    # end here cause we just started, need to render the environment
                    return

                # already use the agent to run one step
                reward = self.get_reward(env)

                self.sac_agent.store(self.state, reward, False, self.action, self.next_state)

                # print(f"episode_reward {self.episode_reward}")


                # update current state
                self.state = self.next_state
                # new action
                self.action = self.sac_agent.choose_action(self.next_state)

                if self.action[-1] < 0:
                    # self.action less than 1 indicate no chance of return
                    self.phase = 5
                    return

                # apply
                self.apply_action(env, self.action)

                self.value_loss, self.q_loss, self.policy_loss = self.sac_agent.train()
                self.episode_reward += reward

            else:
                # record the episode
                self.log(self.episode, self.start_time, self.episode_reward, self.value_loss, self.q_loss, self.policy_loss,
                         len(self.sac_agent.memory))
                self.phase = 5

            global global_running_reward
            global_running_reward = self.episode_reward if self.episode == 0 else \
                0.99 * global_running_reward + 0.01 * self.episode_reward

            ram = psutil.virtual_memory()
            if self.episode % 400 == 0:
                print(f"EP:{self.episode}| EP_r:{self.episode_reward:3.3f}| EP_running_r:{global_running_reward:3.3f}...")
                self.sac_agent.save_weights()
            self.episode += 1
                                        
        if self.phase == 5: #set an independent phase for return to start
            print('Phase 5')
            self.phase = 0




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

if __name__ == '__main__':
    np.random.seed(0)
    env = FinalEnv()
    env.run(Solution(), render=False, render_interval=5, debug=True)
    env.close()
