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
        low_pipes_x = - self.pipe_width
        high_pipes_x = self.screenwidth + (self.screenwidth / 2)

        low_upper_pipes_y = 20 + self.base_y * 0.2 - self.pipe_height
        high_upper_pipes_y = 90 + self.base_y * 0.2 - self.pipe_height
        low_lower_pipes_y = 20 + self.base_y * 0.2 + self.pipe_gap_size
        high_lower_pipes_y = 90 + self.base_y * 0.2 + self.pipe_gap_size

        low_player_vel_y = self.player_min_vel_y
        high_player_vel_y = self.player_max_vel_y
        low_player_y = 0
        high_player_y = self.screenheight

        self.low_pipes = [
            low_pipes_x,
            low_upper_pipes_y,
            low_pipes_x,
            low_lower_pipes_y,
        ]
        self.low = self.low_pipes * 3 + [low_player_vel_y, low_player_y]
        self.low = np.array(self.low)
        self.high_pipes = [
            high_pipes_x,
            high_upper_pipes_y,
            high_pipes_x,
            high_lower_pipes_y,
        ]
        self.high = self.high_pipes * 3 + [high_player_vel_y, high_player_y]
        self.high = np.array(self.high)
        self.observation_space = spaces.Box(low=np.zeros(14),
                                            high=np.ones(14),
                                            dtype=np.float)

    def reset(self):
        self._pre_reset()
        obs, reward, terminal, info = self.step(0)
        return obs

    def step(self, action):
        image_data, reward, terminal, info = super(FlappyBirdFeatureEnv, self).step(action)
        # print(self.info2obs(info))

        return self.info2obs(info), reward, terminal, {'image_data': image_data}

    def info2obs(self, info):
        """
        info: upper_pipes, lower_pipes, player_vel_y, player_y
        """
        obs = []
        if info['upper_pipes'][0]['x'] > self.player_x:
            obs.append(self.low_pipes[0])
            obs.append((self.low_pipes[1] + self.high_pipes[1])/2)
            obs.append(self.low_pipes[2])
            obs.append((self.low_pipes[3] + self.high_pipes[3])/2)

        for i in range(len(info['upper_pipes'])):
            obs.append(info['upper_pipes'][i]['x'])     # [-52, 432]
            obs.append(info['upper_pipes'][i]['y'])     # [-220, -149]
            obs.append(info['lower_pipes'][i]['x'])     # [-52, 432]
            obs.append(info['lower_pipes'][i]['y'])     # [201. 271]

        if len(obs) == 8:
            obs.append(self.high_pipes[0])
            obs.append((self.low_pipes[1] + self.high_pipes[1])/2)
            obs.append(self.high_pipes[2])
            obs.append((self.low_pipes[3] + self.high_pipes[3])/2)

        obs.append(info['player_vel_y'])                # [-8, 10]
        obs.append(info['player_y'])                    # [0, 512]

        return (np.array(obs) - self.low) / (self.high - self.low)
