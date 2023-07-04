import numpy as np
import gym
import pygame as pygame
from gym import spaces
from typing import TypedDict, List
from enum import Enum
from pygame.locals import *


class Location(TypedDict):
    x: int
    y: int


class Movement(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Resource(Enum):
    EMPTY = 1,
    WATER = 2,
    SPICE = 3


class MOVEMENT:
    def __init__(self, location: Location, rows: int, cols: int):
        self.location: Location = location
        self.rows = rows
        self.cols = cols

    def move(self, movement: Movement):
        # Update the agent's position based on the given direction
        if movement == Movement.UP:
            self.location['y'] = max(self.location['y'] - 1, 0)
        elif movement == Movement.DOWN:
            self.location['y'] = min(self.location['y'] + 1, self.rows - 1)
        elif movement == Movement.LEFT:
            self.location['x'] = max(self.location['x'] - 1, 0)
        elif movement == Movement.RIGHT:
            self.location['x'] = min(self.location['x'] + 1, self.cols - 1)
        else:
            raise ValueError('Invalid direction.')

    def check_valid_move(self, movement: Movement):
        # Check if a move in the given direction is valid
        if movement == Movement.UP:
            return self.location['y'] > 0
        elif movement == Movement.DOWN:
            return self.location['y'] < self.rows - 1
        elif movement == Movement.LEFT:
            return self.location['x'] > 0
        elif movement == Movement.RIGHT:
            return self.location['x'] < self.cols - 1
        else:
            raise ValueError('Invalid direction.')

class STATS:
    def __init__(self, hp: int, attack: int, defence: int):
        self.DESTROYED: bool = False
        self.HP = hp
        self.attack = attack
        self.defence = defence


class WORKER(MOVEMENT):
    def __init__(self, location: Location, rows: int, cols: int):
        self.DESTROYED: bool = False
        self.HP = 10
        self.attack = 2
        self.defence = 10
        super().__init__(location, rows, cols)


class WARRIOR(MOVEMENT):
    def __init__(self, location: Location, rows: int, cols: int):
        self.DESTROYED: bool = False
        self.HP = 50
        self.attack = 20
        self.defence = 5
        super().__init__(location, rows, cols)


class SIETCH:
    def __init__(self, location: Location):
        self.location: Location = location
        self.SPICE: int = 10
        self.DESTROYED: bool = False
        self.HP = 200
        self.attack = 0
        self.defence = 50

    def step(self):
        self.SPICE -= 2
        if self.SPICE <= 0:
            self.DESTROYED = True


class RESOURCE:
    def __init__(self, resource: Resource):
        self.resource = resource


class Spice(RESOURCE):
    def __init__(self, quantity: int = 200, decay_per_turn: int = 5):
        self.quantity = quantity
        self.decay_per_turn = decay_per_turn
        super().__init__(Resource.SPICE)

    def step(self):
        self.quantity -= self.decay_per_turn


class Water(RESOURCE):
    def __init__(self, quantity: int = 200, decay_per_turn: int = 5):
        self.quantity = quantity
        self.decay_per_turn = decay_per_turn
        super().__init__(Resource.SPICE)

    def step(self):
        self.quantity -= self.decay_per_turn


class Empty(RESOURCE):
    def __init__(self):
        super().__init__(Resource.EMPTY)


class Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_agents: int = 2, size: int = 50):
        self.MAP = None
        self.N_DISCRETE_ACTIONS = 10
        self.NUM_AGENTS = num_agents

        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        self.size = size
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window_size = 800  # The size of the PyGame window
        self.window = None
        self.clock = None
        self.font = None

    def get_random_grid_coordinate(self) -> Location:
        return Location(x=np.random.choice(self.size),
                        y=np.random.choice(self.size))

    def step(self, action):

        if self.render_mode == "human":
            self._render_frame()

    def is_empty(self, x, y, class_instance):
        return isinstance(self.MAP[x, y], class_instance)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.MAP = np.zeros((self.size, self.size), dtype=object)

        for i in range(self.size):
            for j in range(self.size):
                self.MAP[i, j] = Empty()

        sietches_placed = 0
        min_agent_distance = 5
        while sietches_placed < self.NUM_AGENTS:
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            if self.is_empty(x, y, Empty):
                # Check the distance to other agents
                too_close = False
                for i in range(max(0, x - min_agent_distance), min(self.size, x + min_agent_distance + 1)):
                    for j in range(max(0, y - min_agent_distance), min(self.size, y + min_agent_distance + 1)):
                        if isinstance(self.MAP[i, j], SIETCH):
                            too_close = True
                            break
                    if too_close:
                        break
                if not too_close:
                    self.MAP[x, y] = SIETCH(location={'x': x, 'y': y})
                    sietches_placed += 1
                    # Spawn a warrior adjacent to the agent
                    placed = False
                    for i in range(max(0, x - 1), min(self.size, x + 2)):
                        for j in range(max(0, y - 1), min(self.size, y + 2)):
                            if self.is_empty(i, j, Empty) and not placed:
                                self.MAP[i, j] = WORKER(location={'x': x, 'y': y}, rows=self.size, cols=self.size)
                                placed = True

        for i in range(self.size):
            for j in range(self.size):
                if not self.is_empty(i, j, SIETCH) and not self.is_empty(i, j, WORKER):
                    state = np.random.choice(['Empty', 'Spice', 'Water'])
                    if state == 'Spice':
                        quantity = np.random.randint(100, 200)
                        self.MAP[i, j] = Spice(quantity)
                    elif state == 'Water':
                        quantity = np.random.randint(100, 200)
                        self.MAP[i, j] = Water(quantity)

        if self.render_mode == "human":
            self._render_frame()

    def render(self, mode='human'):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.font = pygame.font.SysFont('Arial', 12)
            pygame.display.set_caption('ATTACK ON LUX')
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target

        for i in range(self.size):
            for j in range(self.size):

                rect = pygame.Rect(pix_square_size * np.array([i, j]), (pix_square_size, pix_square_size))

                if self.is_empty(i, j, Empty):
                    pygame.draw.rect(
                        canvas,
                        (224, 224, 224),
                        rect,
                    )

                if self.is_empty(i, j, Water):
                    pygame.draw.rect(
                        canvas,
                        (102,178,225),
                        rect
                    )

                if self.is_empty(i, j, Spice):
                    pygame.draw.rect(
                        canvas,
                        (160, 160, 160),
                        rect
                    )

                if self.is_empty(i, j, WARRIOR):
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        rect
                    )

                if self.is_empty(i, j, WORKER):
                    pygame.draw.rect(
                        canvas,
                        (255, 160, 122),
                        rect
                    )

                if self.is_empty(i, j, SIETCH):
                    pygame.draw.rect(
                        canvas,
                        (102, 0, 51),
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


env = Env(render_mode="human", num_agents=2, size=50)
env.reset()
print(env.MAP)
while True:
    env.step(action=[])