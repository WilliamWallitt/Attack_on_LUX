import random
from typing import Optional, Union, List, Tuple, TypedDict
import numpy as np
from gym import spaces

from KIT.CLASSES.Map import Map

import gym
from gym.core import RenderFrame, ActType, ObsType

# action space
'''
[center, up, right, down, left, 
mine spice, mine water, 
transfer,
produce worker, produce warrior,
attack worker, attack warrior
] -> 12 actions

'''
class AttackOnLux(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 800  # The size of the PyGame window
        self.window = None
        self.clock = None
        self.font = None
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(12)

        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)


class AttackOnLuxWrapper(Map):

    def __init__(self, size: int, num_agents: int):

        super().__init__(size, num_agents)

    def step(self, actions):



        self.current_agent = self.current_agent + 1 if self.current_agent < self.num_agents else 0


