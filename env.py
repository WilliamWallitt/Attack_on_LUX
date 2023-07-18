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
import torch.nn as nn
import pygame
import time
import torch
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
        self.scale = 35
        self.turn = 0
        self.window = None
        self.clock = None
        self.font = None
        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = AttackOnLuxObsSpec.get_obs_spec(self.num_agents, self.size)
    
    def init(self):
        pygame.init()
        pygame.display.init()
        self.font = pygame.font.SysFont('Arial', 12)
        pygame.display.set_caption('ATTACK ON LUX')
        self.display_surface = pygame.display.set_mode((self.size * self.scale, self.size * self.scale), pygame.HWSURFACE)
        self.player_surface = pygame.image.load('gridworld/player.png').convert()
        self.block_surface = pygame.image.load('gridworld/block.png').convert()
        self.water_surface = pygame.image.load('gridworld/water.jpeg').convert()
        self.spice_surface = pygame.image.load('gridworld/spice.webp').convert()
        self.factory_surface = pygame.image.load('gridworld/factory.png').convert()

        self.running = True

    def render(self):
        self.display_surface.fill((124, 252, 0))
        
        for resource in self.map.resources:
            x, y = resource.position[0], resource.position[1]
            
            if resource.resource_type == ResourceType.EMPTY or resource.amount == 0:
                self.display_surface.blit(self.block_surface, (x * self.scale, y * self.scale))
            elif resource.resource_type == ResourceType.WATER:
                self.display_surface.blit(self.water_surface, (x * self.scale, y * self.scale))
            elif resource.resource_type == ResourceType.SPICE:
                self.display_surface.blit(self.spice_surface, (x * self.scale, y * self.scale))
            elif resource.resource_type == ResourceType.FACTORY:
                self.display_surface.blit(self.factory_surface, (x * self.scale, y * self.scale))
        
        for player in self.players:
            for agent in player.agents:
                x, y = agent.position[0], agent.position[1]
                self.display_surface.blit(self.player_surface, (x * self.scale, y * self.scale))
            
            for agent in player.factories:
                x, y = agent.position[0], agent.position[1]
                self.display_surface.blit(self.factory_surface, (x * self.scale, y * self.scale))
                
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

    def step(self, actions, player_id: int, agent_id: int):
        done = False
        rewards, agent_done = self.players[player_id].execute_actions(actions, agent_id)
        done = done or agent_done
        # if self.turn > 50:
        #     done = True
        self.turn += 1

        for resource in self.map.resources:
            resource.step()

        return self.get_observations(player_id, agent_id), rewards, done, False, {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_observations(self, player_id: int, agent_id: int):
        observations = {
            key: val.copy() #if val.ndim == 3 else val[:, :, :self.size, :self.size].copy() * 0
            for key, val in self.observation_space.sample().items()
        }

        # TODO custom features for each agent???

        player_observation = self.players[player_id].get_observation()
        for agent in player_observation[str(player_id)]["worker"]:
            x, y = agent["position"][0], agent["position"][1]
            observations["worker"][0, x, y] = 1

            spice = next((item for item in agent["mining_options"] if item['type'] == ResourceType.SPICE), None)
            water = next((item for item in agent["mining_options"] if item['type'] == ResourceType.WATER), None)
            health = agent["health"]["amount"]

            observations["worker_cargo_spice"][0, x, y] \
                = spice["amount"] if spice is not None else 0
            observations["worker_cargo_water"][0, x, y] \
                = water["amount"] if water is not None else 0
            observations["worker_health"][0, x, y] \
                = health
        
        for enemy_player in self.players:
            if enemy_player.player == str(player_id):
                continue

            enemy_observation = self.players[int(enemy_player.player)].get_observation()

            for agent in enemy_observation[enemy_player.player]["worker"]:
                x, y = agent["position"][0], agent["position"][1]
                observations["enemy_worker"][0, x, y] = 1

        for agent in player_observation[str(player_id)]["warrior"]:
            x, y = agent["position"][0], agent["position"][1]

            observations["warrior"][0, x, y] = 1

            water = next((item for item in agent["mining_options"] if item['type'] == ResourceType.WATER), None)
            health = agent["health"]["amount"]

            observations["warrior_cargo_water"][0, x, y] \
                = water["amount"] if water is not None else 0
            observations["warrior_health"][0, x, y] \
                = health
        
        for enemy_player in self.players:
            if enemy_player.player == str(player_id):
                continue

            enemy_observation = self.players[int(enemy_player.player)].get_observation()
            
            for agent in enemy_observation[enemy_player.player]["warrior"]:
                x, y = agent["position"][0], agent["position"][1]

                observations["enemy_warrior"][0, x, y] = 1

        for factory in player_observation[str(player_id)]["factory"]:
            x, y = factory["position"][0], factory["position"][1]

            observations["factory"][0, x, y] = 1

            spice = factory["health"]["amount"]

            observations["factory_cargo_spice"][0, x, y] \
                = spice
            observations["factory_health"][0, x, y] \
                = spice

        for resource in self.map.resources:

            if resource.resource_type == ResourceType.WATER:
                observations["water"][0, resource.position[0], resource.position[1]] \
                    = resource.amount

            if resource.resource_type == ResourceType.SPICE:
                observations["spice"][0, resource.position[0], resource.position[1]] \
                    = resource.amount

            if resource.resource_type == ResourceType.EMPTY:
                observations["empty"][0, resource.position[0], resource.position[1]] \
                    = resource.amount

        return observations

    def reset(self, player_id: int = 0, agent_id: int = 0):
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
        return self.get_observations(player_id, agent_id)

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = state.reshape(1, 15*8*8)
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.cpu().numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 256  # Nothing special with 16, feel free to change
        hidden_space2 = 128  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


attackOnLux = AttackOnLux(render_mode="human", size=8, num_agents=2)
attackOnLux.init()
done = False

total_num_episodes = 100  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
#obs_space_dims = attackOnLux.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
#action_space_dims = attackOnLux.action_space.shape[0]
rewards_over_seeds = []

for seed in [1, 5, 13]:  # Fibonacci seeds
    # set seed
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    # Reinitialize model every seed
    model = REINFORCE(64*15, 11)
    #neg_model = REINFORCE(64*15, 11)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment

        observations = None
        has_reset = False
        done = False

        pooled_rewards = []

        agent_id_idx = {str(player_id): 0 for player_id in attackOnLux.num_agents}

        while not done:
            for player_id_turn in range(attackOnLux.num_agents):
                if not has_reset:
                    observations = attackOnLux.reset(player_id=player_id_turn, agent_id=agent_id_idx[str(player_id_turn)])
                    has_reset = True
                else:
                    observations = attackOnLux.get_observations(player_id=player_id_turn, agent_id=agent_id_idx[str(player_id_turn)])
                
                np_observations = np.concatenate([x for x in observations.values()], axis=0)
                np_observations = np.expand_dims(np_observations, axis=0)

                action = model.sample_action(np_observations)
                #neg_action = neg_model.sample_action(np_observations)

                # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                # These represent the next observation, the reward from the step,
                # if the episode is terminated, if the episode is truncated and
                # additional info from the step
                observations, rewards, done, _, _ = attackOnLux.step(action, player_id_turn, agent_id_idx[str(player_id_turn)])
                
                model.rewards.append(rewards)
                pooled_rewards.append(rewards)

                if done:
                    break

                attackOnLux.render()
        
        model.update()
    
    rewards_over_seeds.append(pooled_rewards)
        
    
rewards_to_plot = rewards_over_seeds
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()
