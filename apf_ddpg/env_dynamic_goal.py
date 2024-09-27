import os
import cv2
import logging
import numpy as np

from gym import Space
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from pyrep import PyRep, objects

from catalyst_rl.rl.core import EnvironmentSpec
from catalyst_rl.rl.utils import extend_space
import math



logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

class CoppeliaSimEnvWrapper(EnvironmentSpec):
    def __init__(self, visualize=True,
                 mode="train",
                 **params):
        super().__init__(visualize=visualize, mode=mode)

        # Scene selection
        scene_file_path = os.path.join(os.getcwd(), 'simulation/baxter_v2.ttt')

        # Simulator launch
        self.env = PyRep()
        self.env.launch(scene_file_path, headless=(not visualize))
        self.env.stop()
        self.env.start()
        self.env.step()

        # Task related initialisations in Simulator
        self.tip = objects.dummy.Dummy("Baxter_rightArm_tip")
        self.tip_zero_pose = self.tip.get_pose()
        self.origin = objects.dummy.Dummy("origin")
        self.goal = objects.dummy.Dummy("goal")
        self.goal_STL = objects.shape.Shape("goal_visible")
        self.goal_STL_zero_pose = self.goal_STL.get_pose()

        # Seven joints to be controlled:
        self.right_joint1 = objects.joint.Joint("Baxter_rightArm_joint1")
        self.right_joint2 = objects.joint.Joint("Baxter_rightArm_joint2")
        self.right_joint3 = objects.joint.Joint("Baxter_rightArm_joint3")
        self.right_joint4 = objects.joint.Joint("Baxter_rightArm_joint4")
        self.right_joint5 = objects.joint.Joint("Baxter_rightArm_joint5")
        self.right_joint6 = objects.joint.Joint("Baxter_rightArm_joint6")
        self.right_joint7 = objects.joint.Joint("Baxter_rightArm_joint7")

        # Intervals of joints:
        self.right_joint1_interval = self.right_joint1.get_joint_interval()[1]
        self.right_joint2_interval = self.right_joint2.get_joint_interval()[1]
        self.right_joint3_interval = self.right_joint3.get_joint_interval()[1]
        self.right_joint4_interval = self.right_joint4.get_joint_interval()[1]
        self.right_joint5_interval = self.right_joint5.get_joint_interval()[1]
        self.right_joint6_interval = self.right_joint6.get_joint_interval()[1]
        self.right_joint7_interval = self.right_joint7.get_joint_interval()[1]

        self.rightArm_link2 = objects.shape.Shape("Baxter_rightArm_link2_visible")
        self.rightArm_link3 = objects.shape.Shape("Baxter_rightArm_link3_visible")
        self.rightArm_link4 = objects.shape.Shape("Baxter_rightArm_link4_visible")
        self.rightArm_link5 = objects.shape.Shape("Baxter_rightArm_link5_visible")
        self.rightArm_link6 = objects.shape.Shape("Baxter_rightArm_link6_visible")
        self.rightArm_link7 = objects.shape.Shape("Baxter_rightArm_link7_visible")
        self.rightArm_link8 = objects.shape.Shape("Baxter_rightArm_link8_visible")

        self.upperBody = objects.shape.Shape("Baxter_upperBody_visible")
        self.lowerBody = objects.shape.Shape("Baxter_lowerBody_visible")
        self.baxterBase = objects.shape.Shape("Baxter_base_visible")
        self.leftArm = objects.joint.Joint("Baxter_leftArm_joint1")

        self.step_counter = 0
        self.max_step_count = 100
        self.target_pose = None
        self.initial_distance = None
        self._history_len = 1

        self._observation_space = Box(-1, 1, (9,))#, dtype=np.uint8
        self._action_space = Box(-1, 1, (3,))
        self._state_space = extend_space(self._observation_space, self._history_len)

    @property
    def history_len(self):
        return self._history_len

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    @property
    def state_space(self) -> Space:
        return self._state_space

    @property
    def action_space(self) -> Space:
        return self._action_space

    def step(self, action):
        done = False
        info = {}

        # Make a step in simulation
        self.apply_controls(action)
        self.env.step()
        self.step_counter += 1

        logging.info(f'****** distance: {self.distance_to_goal()} *****')
        
        reward = -1
        if self.success_check():
            done = True
            reward = 1
            logging.info('--------Reset: Success--------')
            return self.get_observation(), reward, done, info
        if self.collision_check():
            done = True
            reward = -self.max_step_count
            logging.info('--------Reset: Collision-------')
            return self.get_observation(), reward, done, info
            #breakpoint()

        # Check reset conditions
        if self.step_counter > self.max_step_count:
            done = True
            logging.info('--------Reset: Timeout--------')
        elif self.distance_to_goal() > 1.5:
            reward = -5
            logging.info('--------Too far from target--------')
        elif self.distance_to_goal() < 0.5:
            reward = -0.5
            logging.info('--------Near target--------')

        return self.get_observation(), reward, done, info

    def reset(self):
        logging.info("Episode reset...")
        self.step_counter = 0
        self.env.stop()
        self.env.start()
        self.env.step()
        self.setup_scene()

        observation = self.get_observation()
        return observation
# -------------- all methods above are required for any Gym environment, everything below is env-specific --------------

    def distance_to_goal(self):
        goal_pos = self.goal.get_position(relative_to=self.origin)#[ 0.4714011 , -0.67175257,  0.69134951]#self.goal.get_position(relative_to=self.origin)
        tip_pos = self.tip.get_position(relative_to=self.origin)

        return np.linalg.norm(np.array(tip_pos) - np.array(goal_pos))

    def sample_goal_pose(self):
        x_range = [1.1, 1.2]
        y_range = [-1.2, -1.4]
        z_range = [1.6, 1.7]

        ranges = np.vstack([x_range, y_range, z_range])
        goal_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

        goal_quat = self.goal_STL_zero_pose[3:]
        
        print("********************************")
        print(f"goal pos: {goal_position}")
        return np.concatenate([goal_position, goal_quat])

    def setup_goal(self):

        # goal_position = self.goal_STL_zero_pose[:3]
        # import pdb; pdb.set_trace()
        # 2D goal randomization
       
        self.target_pose = self.sample_goal_pose()
        self.goal_STL.set_pose(self.target_pose)

        rgb_values_goal = list(np.random.rand(3,))
        rgb_values_plane = list(np.random.rand(3,))
        self.goal_STL.set_color(rgb_values_goal)

        self.initial_distance = self.distance_to_goal()

        print(f"setting goal: {self.goal.get_position(relative_to=self.origin)}")
        print(f"distance: {self.initial_distance}")


    def setup_scene(self):
        self.setup_goal()
        self.tip.set_pose(self.tip_zero_pose)

    def get_observation(self):
        
        goal_pos = self.goal.get_position(relative_to=self.origin)
        tip_pos = self.tip.get_position(relative_to=self.origin)

        right_joint1_angle = self.right_joint1.get_joint_position()
        right_joint2_angle = self.right_joint2.get_joint_position()
        right_joint4_angle = self.right_joint4.get_joint_position()

        obs_pos = np.concatenate((tip_pos, goal_pos, [right_joint1_angle, right_joint2_angle,\
                                            right_joint4_angle]), axis=0)
        return obs_pos

    def collision_check(self):
        return self.rightArm_link2.check_collision(self.upperBody) or \
                self.rightArm_link2.check_collision(self.lowerBody) or \
                self.rightArm_link2.check_collision(self.baxterBase) or \
                self.rightArm_link3.check_collision(self.upperBody) or \
                self.rightArm_link3.check_collision(self.lowerBody) or \
                self.rightArm_link3.check_collision(self.baxterBase) or \
                self.rightArm_link4.check_collision(self.upperBody) or \
                self.rightArm_link4.check_collision(self.lowerBody) or \
                self.rightArm_link4.check_collision(self.baxterBase) or \
                self.rightArm_link5.check_collision(self.upperBody) or \
                self.rightArm_link5.check_collision(self.lowerBody) or \
                self.rightArm_link5.check_collision(self.baxterBase) or \
                self.rightArm_link6.check_collision(self.upperBody) or \
                self.rightArm_link6.check_collision(self.lowerBody) or \
                self.rightArm_link6.check_collision(self.baxterBase) or \
                self.rightArm_link7.check_collision(self.upperBody) or \
                self.rightArm_link7.check_collision(self.lowerBody) or \
                self.rightArm_link7.check_collision(self.baxterBase) or \
                self.rightArm_link8.check_collision(self.upperBody) or \
                self.rightArm_link8.check_collision(self.lowerBody) or \
                self.rightArm_link8.check_collision(self.baxterBase)


    def success_check(self):
        success_reward = 0.
        if self.distance_to_goal() < 0.1:#1: 1 meter
            success_reward = 1
            logging.info('--------Success state--------')
            return True
        return False

    def apply_controls(self, action):

        action = action*(math.pi/16.)

        joints_target_po = [self.right_joint1.get_joint_position() + action[0], self.right_joint2.get_joint_position() + action[1], \
                            self.right_joint4.get_joint_position() + action[2]]

        joints_target_po[0] = np.clip(joints_target_po[0], self.right_joint1_interval[0], self.right_joint1_interval[1]).item()
        joints_target_po[1] = np.clip(joints_target_po[1], self.right_joint2_interval[0], self.right_joint2_interval[1]).item()
        joints_target_po[2] = np.clip(joints_target_po[2], self.right_joint4_interval[0], self.right_joint4_interval[1]).item()

        logging.debug(f"--- current pos: [{self.right_joint1.get_joint_position()}, {self.right_joint2.get_joint_position()}]")
        logging.debug(f"--- target pos: {joints_target_po}")
        
        self.right_joint1.set_joint_target_position(joints_target_po[0])
        self.right_joint2.set_joint_target_position(joints_target_po[1])
        self.right_joint4.set_joint_target_position(joints_target_po[2])
        
        self.env.step()
        self.env.step()
        self.env.step()
        self.env.step()
        self.env.step()
        self.env.step()


    def closeSim(self):
        self.env.stop()
        self.env.shutdown()

