import random
from typing import Optional, Union, List, Tuple, TypedDict
import numpy as np
from gym import spaces

from KIT.CLASSES.Map import Map
import gym
from gym.core import RenderFrame, ActType, ObsType
from KIT.CLASSES.Player import Player

# action space
'''
[center, up, right, down, left, 
mine, 
transfer,
produce worker, produce warrior,
attack unit,
create factory
] -> 11 actions

'''


class AttackOnLux(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int, num_agents: int, render_mode=None):
        self.map = Map(size, num_agents)
        self.players = [
            Player(
                agents=[agent for agent in self.map.agents if agent.player == player_id],
                enemy_agents=[agent for agent in self.map.agents if agent.player != player_id],
                factories=[factory for factory in self.map.factories if factory.player == player_id],
                enemy_factories=[factory for factory in self.map.factories if factory.player != player_id],
                player_id=player_id, resources=self.map.resources)
            for player_id in range(num_agents)]
        self.window_size = 800  # The size of the PyGame window
        self.window = None
        self.clock = None
        self.font = None
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(11)

        pass

    def step(self, actions) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map = Map(size=self.map.size, num_agents=self.map.num_agents)


attackOnLux = AttackOnLux(50, 2)
attackOnLux.reset()
