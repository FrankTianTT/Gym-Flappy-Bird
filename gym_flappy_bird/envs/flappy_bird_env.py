import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
import time
import random
import pygame
from gym_flappy_bird.envs import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle


class FlappyBirdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_demo=False):
        # fix the fps when is_demo = True
        self.is_demo = is_demo
        self.fps = 30
        self.screenwidth = 288
        self.screenheight = 512

        pygame.init()
        self.fpsclock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.screenwidth, self.screenheight))
        pygame.display.set_caption('Flappy Bird')

        self.images, self.sounds, self.hitmasks = flappy_bird_utils.load()
        # gap between upper and lower part of pipe
        self.pipe_gap_size = 100
        self.base_y = self.screenheight * 0.79

        self.player_width = self.images['player'][0].get_width()
        self.player_height = self.images['player'][0].get_height()
        self.pipe_width = self.images['pipe'][0].get_width()
        self.pipe_height = self.images['pipe'][0].get_height()
        self.backgroud_width = self.images['background'].get_width()

        self.player_index_gen = cycle([0, 1, 2, 1])

        self.player_max_vel_y = 10      # max vel along Y, max descend speed
        self.player_min_vel_y = -8      # min vel along Y, max ascend speed

        # actions and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screenwidth, self.screenheight, 3),
                                            dtype=np.uint8)

    def step(self, input_actions):
        pygame.event.pump()

        reward = 0.1
        terminal = False

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions == 1:
            if self.player_y > -2 * self.player_height:
                self.player_vel_y = self.player_flap_acc
                self.player_flapped = True
                if self.is_demo:
                    self.sounds['wing'].play()

        # check for score
        player_mid_pos = self.player_x + self.player_width / 2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe['x'] + self.pipe_width / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self.score += 1
                if self.is_demo:
                    self.sounds['point'].play()
                reward = 1

        # playerIndex basex change
        if (self.loop_iter + 1) % 3 == 0:
            self.player_index = next(self.player_index_gen)
        self.loop_iter = (self.loop_iter + 1) % 30
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        # player's movement
        if self.player_vel_y < self.player_max_vel_y and not self.player_flapped:
            self.player_vel_y += self.player_acc_y
        if self.player_flapped:
            self.player_flapped = False
        self.player_y += min(self.player_vel_y, self.base_y - self.player_y - self.player_height)
        if self.player_y < 0:
            self.player_y = 0

        # move pipes to left
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            upper_pipe['x'] += self.pipe_vel_x
            lower_pipe['x'] += self.pipe_vel_x

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = self._get_random_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # remove first pipe if its out of the screen
        if self.upper_pipes[0]['x'] < -self.pipe_width:
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # check if crash here
        is_crash = self._check_crash({'x': self.player_x, 'y': self.player_y, 'index': self.player_index},
                                     self.upper_pipes,
                                     self.lower_pipes)
        if is_crash:
            if self.is_demo:
                self.sounds['hit'].play()
                self.sounds['die'].play()
            terminal = True
            self.__init__(self.is_demo)
            reward = -1

        # draw sprites

        self.screen.blit(self.images['background'], (0, 0))

        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            self.screen.blit(self.images['pipe'][0], (upper_pipe['x'], upper_pipe['y']))
            self.screen.blit(self.images['pipe'][1], (lower_pipe['x'], lower_pipe['y']))

        self.screen.blit(self.images['base'], (self.base_x, self.base_y))
        # print score so player overlaps the score
        if self.is_demo:
            self._show_score(self.score)
        self.screen.blit(self.images['player'][self.player_index],
                         (self.player_x, self.player_y))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        # feature info, which is use for FlappyBirdFeature-env
        info = {'upper_pipes': self.upper_pipes,
                'lower_pipes': self.lower_pipes,
                'player_vel_y': self.player_vel_y,
                'player_y': self.player_y}

        # print('before player_: x', sum([1 if i['x'] > self.player_x else 0 for i in self.upper_pipes]))
        # print('after  player_x: ', sum([0 if i['x'] > self.player_x else 1 for i in self.upper_pipes]))

        return image_data, reward, terminal, info

    def _pre_reset(self):
        self.score = self.player_index = self.loop_iter = 0
        # player is initialized in a fixed position, which is (screenwidth * 0.2, screenheight - player_height) / 2)
        self.player_x = int(self.screenwidth * 0.2)
        self.player_y = int((self.screenheight - self.player_height) / 2)
        # base is the land in the below of the window
        self.base_x = 0
        self.base_shift = self.images['base'].get_width() - self.backgroud_width

        # generate two pipe when reset the game
        new_pipe1 = self._get_random_pipe()
        new_pipe2 = self._get_random_pipe()
        self.upper_pipes = [
            {'x': self.screenwidth, 'y': new_pipe1[0]['y']},
            {'x': self.screenwidth + (self.screenwidth / 2), 'y': new_pipe2[0]['y']},
        ]
        self.lower_pipes = [
            {'x': self.screenwidth, 'y': new_pipe1[1]['y']},
            {'x': self.screenwidth + (self.screenwidth / 2), 'y': new_pipe2[1]['y']},
        ]

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.pipe_vel_x = -4
        self.player_vel_y = 0           # player's velocity along Y, default same as playerFlapped
        self.player_acc_y = 1           # players downward acceleration
        self.player_flap_acc = -7       # players speed on flapping
        self.player_flapped = False     # True when player flaps

    def reset(self):
        self._pre_reset()
        image_data, reward, terminal, info = self.step(0)

        return image_data

    def render(self, mode='human', close=False):
        pygame.display.update()
        if self.is_demo:
            self.fpsclock.tick(self.fps)

    def _get_random_pipe(self):
        """
        returns a randomly generated pipe, which is a list of dict of upper and lower pipe.
        """
        # y of gap between upper and lower pipe
        gap_y_list = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gap_y_list) - 1)
        gap_y = gap_y_list[index]

        gap_y += int(self.base_y * 0.2)
        pipe_x = self.screenwidth + 10
        return [
            {'x': pipe_x, 'y': gap_y - self.pipe_height},  # upper pipe
            {'x': pipe_x, 'y': gap_y + self.pipe_gap_size},  # lower pipe
        ]

    def _show_score(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0  # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.images['numbers'][digit].get_width()

        Xoffset = (self.screenwidth - totalWidth) / 2

        for digit in scoreDigits:
            self.screen.blit(self.images['numbers'][digit], (Xoffset, self.screenheight * 0.1))
            Xoffset += self.images['numbers'][digit].get_width()

    def _check_crash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.images['player'][0].get_width()
        player['h'] = self.images['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.base_y - 1:
            return True
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], self.pipe_width, self.pipe_height)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], self.pipe_width, self.pipe_height)

                # player and upper/lower pipe hitmasks
                pHitMask = self.hitmasks['player'][pi]
                uHitmask = self.hitmasks['pipe'][0]
                lHitmask = self.hitmasks['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self._pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self._pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False

    def _pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False