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
import cv2


def skew(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def adjoint_matrix(pose):
    adjoint = np.zeros([6, 6])
    adjoint[:3, :3] = pose[:3, :3]
    adjoint[3:6, 3:6] = pose[:3, :3]
    adjoint[3:6, 0:3] = skew(pose[:3, 3]) @ pose[:3, :3]
    return adjoint


def so32rot(rotation):
    assert rotation.shape == (3, 3)
    if np.isclose(rotation.trace(), 3):
        return np.zeros(3), 1
    if np.isclose(rotation.trace(), -1):
        # omega, theta = mat2axangle(rotation)
        theta = np.arccos((np.max([-1 + 1e-7, rotation.trace()]) - 1) / 2)
        omega = 1 / 2 / np.sin(theta) * np.array(
            [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
        return omega, theta
    theta = np.arccos((rotation.trace() - 1) / 2)
    omega = 1 / 2 / np.sin(theta) * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
    return omega, theta


def pose2exp_coordinate(pose):
    # ref solution
    omega, theta = so32rot(pose[:3, :3])
    ss = skew(omega)
    inv_left_jacobian = np.eye(3, dtype=np.float) / theta - 0.5 * ss + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
    v = inv_left_jacobian @ pose[:3, 3]
    return np.concatenate([omega, v]), theta


def compute_pose_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """You need to implement this function

    A distance function in SE(3) space
    Args:
        pose1: transformation matrix
        pose2: transformation matrix

    Returns:
        Distance scalar

    """
    relative_rotation = pose1[:3, :3].T @ pose2[:3, :3]
    rotation_term = np.arccos((np.trace(relative_rotation) - 1) / 2)
    translation_term = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    # print("Rotation term is: {}\nTranslation Term is: {}".format(rotation_term, translation_term))
    return rotation_term + translation_term


def pose2mat(pose):
    # ref solution
    mat44 = np.eye(4)
    mat44[:3, 3] = pose.p

    quat = np.array(pose.q).reshape([4, 1])
    if np.linalg.norm(quat) < np.finfo(np.float).eps:
        return mat44
    quat /= np.linalg.norm(quat, axis=0, keepdims=False)
    img = quat[1:, :]
    w = quat[0, 0]

    Eq = np.concatenate([-img, w * np.eye(3) + skew(img)], axis=1)  # (3, 4)
    Gq = np.concatenate([-img, w * np.eye(3) - skew(img)], axis=1)  # (3, 4)
    mat44[:3, :3] = np.dot(Eq, Gq.T)

    # mat44[:3, :3] = Eq @ Gq.T
    return mat44


global_time_left = 200000.


class Solution(SolutionBase):
    """
    Implement the init function and act functions to control the robot
    Your task is to transport all cubes into the blue bin
    You may only use the following functions

    FinalEnv class:
    - get_agents
    - get_metadata

    Robot class:
    - get_observation
    - configure_controllers
    - set_action
    - get_metadata
    - get_compute_functions

    Camera class:
    - get_metadata
    - get_observation

    All other functions will be marked private at test time, calling those
    functions will result in a runtime error.

    How your solution is tested:
    1. new testing environment is initialized
    2. the init function gets called
    3. the timestep  is set to 0
    4. every 5 time steps, the act function is called
    5. when act function returns False or 200 seconds have passed, go to 1
    """

    def init(self, env: FinalEnv):
        """called before the first step, this function should also reset the state of
        your solution class to prepare for the next run

        """
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

        # get the box location from the overhead camera

        # measure the bin
        self.bin_id = meta['bin_id']
        self.basic_info = {}
        self.locate_bin_bbox(c4)

        self.measured = False

    def act(self, env: FinalEnv, current_timestep: int):
        """called at each (actionable) time step to set robot actions. return False to
        indicate that the agent decides to move on to the next environment.
        Returning False early could mean a lower success rate (correctly placed
        boxes / total boxes), but it can also save you some time, so given a
        fixed total time budget, you may be able to place more boxes.

        """
        global global_time_left

        r1, r2, c1, c2, c3, c4 = env.get_agents()

        pf_left = f = r1.get_compute_functions()['passive_force'](True, True, False)
        pf_right = f = r2.get_compute_functions()['passive_force'](True, True, False)

        '''
        Phase 0: Initialization and Preparation
        Objective: Prepare the robots (r1 and r2) for the task by moving their arms to predefined initial positions.
        '''
        if self.phase == 0:
            if 20000 - (current_timestep // 5) < 4100:
                return False

            #target joint positions
            t1 = [2, 1, 0, -1.5, -1, 1, -2]
            t2 = [-2, 1, 0, -1.5, 1, 1, -2]

            #set target joint positions for r1 and r2
            r1.set_action(t1, [0] * 7, pf_left)
            r2.set_action(t2, [0] * 7, pf_right)

            pose = r1.get_observation()[2][9]
            p, q = pose.p, pose.q
            # print("init r1 position ",p, quat2euler(q))

            #check if robots reach target position and move to phase 1
            if np.allclose(r1.get_observation()[0], t1, 0.05, 0.05) and np.allclose(
                    r2.get_observation()[0], t2, 0.05, 0.05):
                self.phase = 1
                self.counter = 0
                self.selected_x = None

        '''
        Phase 1: Picking a Box
        Objective: Use the overhead camera to locate a box and move the robots' end effectors to pick it up.
        '''
        if self.phase == 1:
            self.counter += 1

            if (self.counter == 1):
                #call pick_box to identify a box within the workspace using camera c4 (top)
                selected, flag = self.pick_box(c4)

                if 20000 - current_timestep // 5 < global_time_left:
                    if flag == False:
                        return False

                #coordinates of chosen box
                self.selected_x = selected[0]
                self.selected_y = selected[1]

            #get target pose of end effector, move to box location
            target_pose_left = Pose([self.selected_x, 0.5, 0.67], euler2quat(np.pi, -np.pi / 6, -np.pi / 2))

            #move r1 to target pose
            self.diff_drive(r1, 9, target_pose_left)
            # self.jacobian_drive(r1, 9, target_pose_left)
            pose = r1.get_observation()[2][9]
            p, q = pose.p, pose.q

            #same
            target_pose_right = Pose([self.selected_x, -0.5, 0.6], euler2quat(np.pi, -np.pi / 4, np.pi / 2))
            self.diff_drive(r2, 9, target_pose_right)
            # self.jacobian_drive(r2, 9, target_pose_right)

            #transition to phase 2 after a fixed duration
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
        '''
        Phase 2: Scooping the Box
        Adjust the robots' end effectors to scoop the box
        '''
        if self.phase == 2:
            self.counter += 1
            self.diff_drive(r1, 9, self.pose_left)
            self.diff_drive(r2, 9, self.pose_right)

            if self.counter == 3000 / 5:
                self.phase = 3

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                # p[1] = 0.5
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
            print("3")
            if self.counter < 500 / 5:
                self.counter += 1
                self.diff_drive(r1, 9, self.pose_left)
                self.diff_drive(r2, 9, self.pose_right)
                print("3.1")

            elif self.counter < 1500 / 5:
                self.counter += 1
                t1 = [2, 1, 0, -1.5, -1, 1, -2]
                r1.set_action(t1, [0] * 7, pf_left)
                self.diff_drive(r2, 9, self.pose_right)
                print("3.2")

            else:
                self.phase = 4
                self.counter = 0

                if not self.measured:
                    self.measure_spade(c4, r2)

                if self.spade_is_empty(c4, r2):
                    self.phase = 0
                    global_time_left -= 1.
                    return True

        if self.phase == 4:
            self.counter += 1
            # middle point 1
            if (self.counter < 3000 / 5):
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.5
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.jacobian_drive(r2, 9, Pose(p, q))
            elif (self.counter < 9000 / 5):
                # p = [-1, -0.15, 1.18] #a milestone to control the trajectory for avoiding collision
                p = [-1, -0., 1.2]  # a milestone to control the trajectory for avoiding collision
                q = euler2quat(0, -np.pi / 3, 0)
                self.jacobian_drive(r2, 9, Pose(p, q), speed=0.4)
            elif (self.counter < 10000 / 5):
                cent = self.basic_info['bin_center']
                length = self.basic_info['spade_length']
                p = cent.copy()
                # p[2] += 0.1
                p[0] += length * 2.
                # p = [-1, -0.1, 1.2]
                q = euler2quat(0, -np.pi / 1.2, 0)
                self.jacobian_drive(r2, 9, Pose(p, q), speed=0.4)

            elif (self.counter < 11000 / 5):
                cent = self.basic_info['bin_center']
                length = self.basic_info['spade_length']
                p = cent.copy()
                # p[2] += 0.15
                p[0] += length * 2.
                # p = [-1, -0.1, 1.2]
                q = euler2quat(0, -np.pi / 1., 0)
                self.jacobian_drive(r2, 9, Pose(p, q), speed=0.4)
            else:
                # selected, flag = self.pick_box(c4)
                # if 20000 - current_timestep//5 < global_time_left:
                #     if flag == False:
                #         return False
                self.phase = 0

        global_time_left -= 1.

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
        my_adjoint_matrix[3:6, 0:3] = skew(current_position) @ current_rotation
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

    def get_global_position_from_camera(self, camera, depth, x, y):
        """
        This function is provided only to show how to convert camera observation to world space coordinates.
        It can be removed if not needed.

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

        # get the -1 to 1 (x,y,z) coordinate
        ndc = np.array([xf, yf, zf, 1]) * 2 - 1

        # transform from image space to view space
        v = np.linalg.inv(proj) @ ndc
        v /= v[3]

        # transform from view space to world space
        v = model @ v

        return v

    def pick_box(self, c, thres_x=0.25, thres_y=0.4):
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

    def measure_spade(self, c, r):
        '''
        use hough line transform to detect long spade side, and determine the size of spade
        :param c:
        :return:
        '''
        color, depth, segmentation = c.get_observation()

        r_meta = r.get_metadata()
        spade_id = r_meta['link_ids'][-1]

        h_s, w_s = np.where(segmentation == spade_id)
        points = np.stack([w_s, h_s], axis=1)
        rbox = cv2.minAreaRect(points)
        center, size, rot = rbox
        center = np.array(center, np.int)

        spade_cent_world = self.get_global_position_from_camera(c, depth, center[0], center[1])

        cur_pose = r.get_observation()[2][9]  # get spade position
        spade_root_world = cur_pose.p

        spade_length = 2 * (np.sum((spade_cent_world[:3] - spade_root_world) ** 2) ** 0.5)

        self.basic_info['spade_length'] = spade_length
        self.measured = True

    # Use top camera to get the box location
    # return False when cannot find it

    def locate_bin_bbox(self, c):

        color, depth, segmentation = c.get_observation()
        mask = segmentation == self.bin_id

        cm = c.get_metadata()
        proj, model = cm['projection_matrix'], cm['model_matrix']
        w, h = cm['width'], cm['height']

        xf = (np.arange(w) + 0.5) / w
        yf = 1 - (np.arange(h) + 0.5) / h

        gx, gy = np.meshgrid(xf, yf, )
        # get 0 to 1 coordinate for (x, y) coordinates

        ndc = np.stack([gx, gy, depth, np.ones(depth.shape)], axis=2) * 2 - 1
        # get the -1 to 1 (x,y,z) coordinates
        ndc = np.expand_dims(ndc, axis=3)

        # transform from image space to view space
        unproj = np.linalg.inv(proj)
        unproj = np.reshape(unproj, (1, 1, 4, 4))

        v = np.matmul(unproj, ndc)
        v = v / v[:, :, 3:4, 0:1]

        # transform from view space to world space
        model = np.reshape(model, (1, 1, 4, 4))
        points = np.matmul(model, v)[..., :3, 0]
        # points in the world coordinate

        # _, axs = plt.subplots(1, 3)
        # for i in range(3):
        #     axs[i].imshow(points[..., i])
        # plt.show()

        z_coord_masked = points[..., 2] * mask.astype(np.float32)
        max_height = z_coord_masked.max()
        # find the top regions

        canvas = np.zeros(depth.shape)
        canvas[np.abs(z_coord_masked - max_height) < 0.0001] = 255

        h_s, w_s = np.where(canvas == 255)
        bin_top_area = np.stack([w_s, h_s], axis=1)
        rbox = cv2.minAreaRect(bin_top_area)
        center, size, rot = rbox
        # (w_in_im, h_in_im), (), (height, width)
        corners = cv2.boxPoints(rbox)
        # (x, y), x = w_in_im, y = h_in_im

        center = np.array(center, np.int)
        h_in_im, w_in_im = center[1], center[0]

        bin_top_center = np.array([points[h_in_im, w_in_im, 0],
                                   points[h_in_im, w_in_im, 1],
                                   max_height])

        self.basic_info['bin_center'] = bin_top_center
        self.basic_info['bin_orientation'] = rot
        self.basic_info['bin_corner'] = corners

    def spade_is_empty(self, c, r):

        color, depth, segmentation = c.get_observation()
        r_meta = r.get_metadata()
        spade_id = r_meta['link_ids'][-1]

        h_s, w_s = np.where(segmentation == spade_id)
        points = np.stack([w_s, h_s], axis=1)
        rbox = cv2.minAreaRect(points)

        center, size, rot = rbox
        center = np.array(center, np.int)
        corners = cv2.boxPoints(rbox)

        ab = []
        for i in range(4):
            ab.append([corners[i], corners[(i + 1) % 4]])

        for box_id in self.box_ids:
            m = np.where(segmentation == box_id)
            if len(m[0]):
                min_x, max_x = np.min(m[1]), np.max(m[1])
                min_y, max_y = np.min(m[0]), np.max(m[0])
                x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
                box_cent = np.array([x, y])

                if self.in_spade(ab, center, box_cent):
                    # import matplotlib.pyplot as plt
                    # plt.imshow(color)
                    # for p in corners:
                    #     plt.plot(p[0], p[1], 'ro')
                    # plt.plot(center[0], center[1], marker='.', c=(0, 0, 1.))
                    # plt.plot(box_cent[0], box_cent[1], marker='.', c=(0, 1., 0.))
                    # plt.show()
                    return False
        return True

    def in_spade(self, ab, spade_center, box_center):
        for pair in ab:
            if not self.same_side(pair[0], pair[1], spade_center, box_center):
                return False
        return True

    def same_side(self, a, b, s_cen, b_cen):
        '''
        f1 = ab x a->s_cen
        f2 = ab x a->b_cen
        f1*f2 > 0: same side
        f1*f2 = 0: on border
        f1*f2 < 0: out of spade
        '''
        ab = b - a
        a_sc = s_cen - a
        a_bc = b_cen - a
        f1 = np.cross(ab, a_sc)
        f2 = np.cross(ab, a_bc)
        return f1 * f2 > 0
if __name__ == '__main__':
    np.random.seed(0)
    env = FinalEnv()
    env.run(Solution(), render=True, render_interval=5, debug=True)
    env.close()