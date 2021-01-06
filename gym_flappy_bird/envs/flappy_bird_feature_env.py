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
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-300, high=300, shape=([14]), dtype=np.uint8)

    def reset(self):
        self._pre_reset()
        obs, reward, terminal, info = self.step(0)
        return obs

    def step(self, action):
        image_data, reward, terminal, info = super(FlappyBirdFeatureEnv, self).step(action)
        # print(self.info2obs(info))

        return self.info2obs(info), reward, terminal, {'image_data': image_data}

    @staticmethod
    def info2obs(info):
        """
        info: upper_pipes, lower_pipes, player_vel_y, player_y
        lower_bound of pipes_x = - pipe_width = - 52
        upper_bound of pipes_x = 288 + 10 = 298
        lower_bound of upper_pipes_x = 20 + base_y * 0.2 - pipe_height = 20 + 80 - 320 = - 220
        upper_bound of upper_pipes_x = 90 + base_y * 0.2 - pipe_height = 90 + 81 - 320 = - 149
        lower_bound of lower_pipes_x = 20 + base_y * 0.2 + pipe_gap_size = 20 + 81 + 100 = 201
        upper_bound of lower_pipes_x = 90 + base_y * 0.2 + pipe_gap_size = 90 + 81 + 100 = 271
        """
        obs = []
        # number of upper_pipes and lower_pipes is between 2 and 3, so this 3 as placeholder
        for i in range(3):
            if len(info['upper_pipes']) <= i: # case of length = 2
                obs.extend([0, 0, 0, 0])
                break
            obs.append(info['upper_pipes'][i]['x'])
            obs.append(info['upper_pipes'][i]['y'])
            obs.append(info['lower_pipes'][i]['x'])
            obs.append(info['lower_pipes'][i]['y'])
        obs.append(info['player_vel_y'])
        obs.append(info['player_y'])
        return np.array(obs)
