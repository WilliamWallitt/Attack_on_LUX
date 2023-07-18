import random

import numpy as np
import pygame

MAP_SIZE = 16
MAX_FOOD = 16
MAX_STEP = 32
SCALE = 34


class GridWorld:
    def __init__(self):
        self.movement = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
        self.agent_energy = -1
        self.agent_x = -1
        self.agent_y = -1
        self.game_step = -1
        self.block_surface = None
        self.display_surface = None
        self.player_surface = None
        self.running = False
        self.map = np.zeros((MAP_SIZE, MAP_SIZE))
        self.reset(0, 0)

    def reset(self, episode, score):
        self.agent_energy = MAP_SIZE
        self.agent_x = random.randint(0, MAP_SIZE - 1)
        self.agent_y = random.randint(0, MAP_SIZE - 1)
        self.game_step = 0
        self.block_surface = None
        self.display_surface = None
        self.player_surface = None
        self.running = False

        pygame.display.set_caption(f'episode {episode}   score {score:.2f}')

        self.map = np.zeros((MAP_SIZE, MAP_SIZE))
        for i in range(MAP_SIZE * 3):
            x = random.randint(0, MAP_SIZE - 1)
            y = random.randint(0, MAP_SIZE - 1)
            self.map[x, y] = random.randint(1, MAX_FOOD)

        self.init()
        return self.get_state(), ''

    def get_state(self):
        s = np.zeros((MAP_SIZE, MAP_SIZE, 4))
        s[self.agent_x, self.agent_y, 0] = 1
        s[:, :, 1] = self.map / MAX_FOOD
        s[:, :, 2] = self.agent_energy / MAX_FOOD
        s[:, :, 3] = self.game_step / MAX_STEP
        return s

    def step(self, action):
        reward = -1
        self.agent_energy -= 1
        self.game_step += 1
        dx, dy = self.movement[action]
        if 0 <= self.agent_x + dx < MAP_SIZE and 0 <= self.agent_y + dy < MAP_SIZE:
            self.agent_x, self.agent_y = self.agent_x + dx, self.agent_y + dy
        if action > 0:
            self.agent_energy -= 0.1
            reward -= 0.1
        for dx, dy in self.movement:
            xx, yy = self.agent_x + dx, self.agent_y + dy
            if 0 <= xx < MAP_SIZE and 0 <= yy < MAP_SIZE:
                if self.map[xx, yy] > 0:
                    self.agent_energy += 1
                    self.map[xx, yy] -= 1
                    reward += 1

        done = False
        if self.agent_energy <= 0:
            print('***** DEAD *****')
            reward = -50
            done = True
        elif self.game_step == MAX_STEP:
            print('##### WIN #####')
            reward = 1
            done = True

        self.render()
        return self.get_state(), reward, done, '', ''

    def init(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((MAP_SIZE * SCALE, MAP_SIZE * SCALE), pygame.HWSURFACE)
        self.player_surface = pygame.image.load('gridworld/player.png').convert()
        self.block_surface = pygame.image.load('gridworld/block.png').convert()
        self.running = True

    def render(self):
        self.display_surface.fill((124, 252, 0))
        for x in range(0, MAP_SIZE):
            for y in range(0, MAP_SIZE):
                if self.map[x, y] > 0:
                    self.display_surface.blit(self.block_surface, (x * SCALE, y * SCALE))
        self.display_surface.blit(self.player_surface, (self.agent_x * SCALE, self.agent_y * SCALE))
        pygame.display.flip()
