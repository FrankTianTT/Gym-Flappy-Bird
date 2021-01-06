import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
import time
import random
import pygame
from gym_flappy_bird.envs import flappy_bird_utils
from gym_flappy_bird.envs.flappy_bird_env import FlappyBirdEnv
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

class FlappyBirdFeatureEnv(FlappyBirdEnv):
    def __init__(self, is_demo=False):
        super(FlappyBirdFeatureEnv, self).__init__(is_demo=is_demo)

        self.low = np.array([-52, -220, -52, 200, -52, -220, -52, 200, -8, 0])
        self.high = np.array([432, -140, 432, 280, 432, -140, 432, 280, 10, 512])
        self.observation_space = spaces.Box(low=np.full(shape=[10], fill_value=0),
                                            high=np.full(shape=[10], fill_value=1),
                                            dtype=np.float)

    def reset(self):
        self._pre_reset()
        obs, reward, terminal, info = self.step(0)
        return obs

    def step(self, action):
        image_data, reward, terminal, info = super(FlappyBirdFeatureEnv, self).step(action)
        # print(self.info2obs(info))

        return self.info2obs(info, self.low, self.high), reward, terminal, {'image_data': image_data}

    @staticmethod
    def info2obs(info, low, high):
        """
        info: upper_pipes, lower_pipes, player_vel_y, player_y
        lower_bound of pipes_x = - pipe_width = - 52
        upper_bound of pipes_x = 288 + 144 = 432
        lower_bound of upper_pipes_x = 20 + base_y * 0.2 - pipe_height = 20 + 80 - 320 = - 220
        upper_bound of upper_pipes_x = 90 + base_y * 0.2 - pipe_height = 90 + 81 - 320 = - 149
        lower_bound of lower_pipes_x = 20 + base_y * 0.2 + pipe_gap_size = 20 + 81 + 100 = 201
        upper_bound of lower_pipes_x = 90 + base_y * 0.2 + pipe_gap_size = 90 + 81 + 100 = 271
        """
        obs = []
        for i in range(2):
            obs.append(info['upper_pipes'][i]['x'])     # [-52, 432]
            obs.append(info['upper_pipes'][i]['y'])     # [-220, -149]
            obs.append(info['lower_pipes'][i]['x'])     # [-52, 432]
            obs.append(info['lower_pipes'][i]['y'])     # [201. 271]
        obs.append(info['player_vel_y'])                # [-8, 10]
        obs.append(info['player_y'])                    # [0, 512]

        return (np.array(obs) - low) / (high - low)
