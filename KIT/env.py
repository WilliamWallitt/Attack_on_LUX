import random
from typing import Optional, Union, List, Tuple, TypedDict
import numpy as np
import pygame
from gym import spaces
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from KIT.CLASSES.Map import Map
import gym
from gym.core import RenderFrame, ActType, ObsType
from KIT.CLASSES.Player import Player
import pprint
from KIT.SPEC.Obs import AttackOnLuxObsSpec
from KIT.CLASSES.Resource import ResourceType
import pygame

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
        self.map = None
        self.size = size
        self.num_agents = num_agents
        self.num_actions = 11
        self.players = []
        self.window_size = 800  # The size of the PyGame window
        self.window = None
        self.clock = None
        self.font = None
        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = AttackOnLuxObsSpec.get_obs_spec(self.num_agents, self.size)

    def step(self, actions):
        rewards = []
        done = False
        for i, player in enumerate(self.players):
            agent_actions = actions[i * len(player.agents): (i + 1) * len(player.agents)]
            agent_rewards, agent_done = player.execute_actions(agent_actions)
            rewards.extend(agent_rewards)
            done = done or agent_done
        return self._get_observations(), rewards, done, {}

    def render(self, mode='human') -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self._render_frame()

    def _render_frame(self):

        colors = {
            'WATER': (0, 0, 255),  # Blue
            'SPICE': (255, 165, 0),  # Orange
            'FACTORY': (128, 128, 128),  # Gray
            'WARRIOR': (255, 0, 0),  # Red
            'WORKER': (0, 255, 0),  # Green
            'EMPTY': (255, 255, 255)  # White
        }

        cell_size = 20
        grid_width, grid_height = self.size, self.size

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

        resource_colors = [['Empty'] * grid_width for _ in range(grid_height)]

        for resource in self.map.resources:
            x, y = resource.position
            resource_colors[x][y] = colors[resource.resource_type.name]


        # # Finally, add some gridlines
        for row in range(grid_height):
            for col in range(grid_width):



                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                pygame.draw.rect(canvas, colors["FACTORY"], rect)

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

    def _get_observations(self):
        observation = {
            key: val.copy() if val.ndim == 2 else val[:, :, :self.size, :self.size].copy() * 0
            for key, val in self.observation_space.sample().items()
        }

        for player in self.players:
            player_observation = player.get_observation()
            for agent in player_observation[player.player]["worker"]:
                x, y = agent["position"][0], agent["position"][1]
                observation["worker"][0, int(player.player), x, y] = 1

                spice = next((item for item in agent["mining_options"] if item['type'] == ResourceType.SPICE), None)
                water = next((item for item in agent["mining_options"] if item['type'] == ResourceType.WATER), None)
                health = agent["health"]["amount"]
                hp = agent["stats"]["hp"]

                observation["worker_cargo_spice"][0, int(player.player), x, y] \
                    = spice["amount"] if spice is not None else 0
                observation["worker_cargo_water"][0, int(player.player), x, y] \
                    = water["amount"] if water is not None else 0
                observation["worker_health"][0, int(player.player), x, y] \
                    = health
                observation["worker_hp"][0, int(player.player), x, y] \
                    = hp

            for agent in player_observation[player.player]["warrior"]:
                x, y = agent["position"][0], agent["position"][1]

                observation["warrior"][0, int(player.player), x, y] = 1

                water = next((item for item in agent["mining_options"] if item['type'] == ResourceType.WATER), None)
                health = agent["health"]["amount"]
                hp = agent["stats"]["hp"]

                observation["warrior_cargo_water"][0, int(player.player), x, y] \
                    = water["amount"] if water is not None else 0
                observation["warrior_health"][0, int(player.player), x, y] \
                    = health
                observation["warrior_hp"][0, int(player.player), x, y] \
                    = hp

            for factory in player_observation[player.player]["factory"]:
                x, y = factory["position"][0], factory["position"][1]

                observation["factory"][0, int(player.player), x, y] = 1

                spice = factory["health"]["amount"]
                health = factory["stats"]["hp"]

                observation["factory_cargo_spice"][0, int(player.player), x, y] \
                    = spice
                observation["factory_health"][0, int(player.player), x, y] \
                    = health

        for resource in self.map.resources:

            if resource.resource_type == ResourceType.WATER:
                observation["water", 0, resource.position[0], resource.position[1]] \
                    = resource.amount

            if resource.resource_type == ResourceType.SPICE:
                observation["spice", 0, resource.position[0], resource.position[1]] \
                    = resource.amount

            if resource.resource_type == ResourceType.EMPTY:
                observation["empty", 0, resource.position[0], resource.position[1]] \
                    = resource.amount

        return observation

    def reset(self, seed=None, options=None):
        self.map = Map(size=self.size, num_agents=self.num_agents)
        self.players = [
            Player(
                agents=[agent for agent in self.map.agents if agent.player == str(player_id)],
                enemy_agents=[agent for agent in self.map.agents if agent.player != str(player_id)],
                factories=[factory for factory in self.map.factories if factory.player == str(player_id)],
                enemy_factories=[factory for factory in self.map.factories if factory.player != str(player_id)],
                player_id=str(player_id), resources=self.map.resources, map_size=self.size,
                num_actions=self.num_actions)
            for player_id in range(self.num_agents)]
        return self._get_observations()


attackOnLux = AttackOnLux(render_mode="human", size=50, num_agents=2)
observations = attackOnLux.reset()
done = False
while not done:
    actions = [agent.choose_action(observation) for agent, observation in zip(attackOnLux.players, observations)]
    new_observations, rewards, done, _ = attackOnLux.step(actions)
    observations = new_observations
    break

while True:

    attackOnLux.render("human")
# print(attackOnLux.observation_space)
