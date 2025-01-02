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
            if (self.counter < 3000 / 5):
                print("Phase 4a") #debug
                #TODO: collision-free trajectory to get r2 off the ground
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.diff_drive2(r2, 9, Pose(p, q), [4, 5, 6], [0, 0, 0, -1, 0], [0, 1, 2, 3, 4])
                
            elif (self.counter < 6000 / 5):
                print("Phase 4b") #debug
                #TODO: collision-free trajectory to get r2 away from r1
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.diff_drive2(r2, 9, Pose(p, q), [4, 5, 6], [0, 0, 1, -1, 0], [0, 1, 2, 3, 4])
                                
            elif not self.is_spade_above_bin(r2, self.r2_target_bin_position): #counter-independent condition
                print("Phase 4c(i)") #debug
                # move r2 to bin
                self.move_to_bin(r2, self.r2_target_bin_position)
                
            elif not self.is_spade_above_bin(r1, self.r1_target_bin_position): #counter-independent condition
                print("Phase 4c(ii)") #debug
                # move r1 to bin
                self.move_to_bin(r1, self.r1_target_bin_position)
                
            else:
                print("Phase 4d")  # Debug
                # Initialize rotation state if it doesn't exist
                if not hasattr(self, "_rotation_step"):
                    self._rotation_step = 0
                    self._total_steps = 36
                    self._delay_multiplier = 30
                    self._delay_counter = 0
                    self._initial_qpos = r2.get_observation()[0]  
                    
                if self._delay_counter < self._delay_multiplier:
                    self._delay_counter += 1
                else:
                    self._delay_counter = 0
                    self._rotation_step += 1
                    
                    self.rotate_spade(r2, self._initial_qpos, self._rotation_step, self._total_steps)
                    self.rotate_spade(r1, self._initial_qpos, self._rotation_step, self._total_steps, clockwise=False)

                # If rotation is complete, clean up private variables
                if self._rotation_step >= self._total_steps:
                    del self._rotation_step
                    del self._total_steps
                    del self._delay_multiplier
                    del self._delay_counter
                    del self._initial_qpos
                    self.phase += 1  
                                        
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
    env.run(Solution(), render=True, render_interval=5, debug=True)
    env.close()
