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
