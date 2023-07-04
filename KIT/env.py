import random
from typing import Optional, Union, List, Tuple, TypedDict
import numpy as np
from KIT.CLASSES.Resource import Resource, ResourceType
from KIT.CLASSES.Factory import Factory
from KIT.CLASSES.Unit import Unit
from KIT.UTILITIES.utilities import create_factory, produce_warrior_unit
import gym
from gym.core import RenderFrame, ActType, ObsType
from enum import Enum


class AttackOnLux(gym.Env):

    def __init__(self):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)


class AttackOnLuxEngine:

    def __int__(self):
        self.map_size: int = 50
        self.agents: int = 2


class Map:

    def __init__(self, size: int, num_agents: int):
        self.size = size
        self.num_agents = num_agents
        self.map = np.zeros((self.size, self.size), dtype=object)
        self.factories: List[Factory] = []
        self.agents: List[Unit] = []
        for i in range(self.size):
            for j in range(self.size):
                self.map[i, j] = Resource(resource_type=ResourceType.EMPTY)

    def ran

    def place_factories_and_agents(self):
        factories_placed = 0
        min_agent_distance = 5
        while factories_placed < self.num_agents:
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            # Check the distance to other agents
            too_close = False
            for i in range(max(0, x - min_agent_distance), min(self.size, x + min_agent_distance + 1)):
                for j in range(max(0, y - min_agent_distance), min(self.size, y + min_agent_distance + 1)):
                    if len([x for x in self.factories if x.position == np.ndarray([i, j])]) != 0:
                        too_close = True
                        break
                if too_close:
                    break
            if not too_close:
                self.factories.append(create_factory(np.ndarray([x, y]), self.size))
                factories_placed += 1
                # Spawn a worker adjacent to the agent
                placed = False
                for i in range(max(0, x - 1), min(self.size, x + 2)):
                    for j in range(max(0, y - 1), min(self.size, y + 2)):
                        if len([x for x in self.agents if x.position == np.ndarray([x, y])]) != 0 and not placed:
                            self.agents.append(produce_warrior_unit(np.ndarray([x, y])))
                            placed = True
