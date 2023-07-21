import random
from typing import Optional, Union, List, Tuple, TypedDict
import numpy as np
import pygame
from gym import spaces
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from torch.nn.functional import softmax

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
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import keyboard
import torch.nn.functional as F
import wandb

# hparams
LEARNING_RATE = 1e-8  # Learning rate for policy optimization
GAMMA = 0.99  # Discount factor
EPS = 1e-4  # small number for mathematical stability
TOTAL_EPISODES = 100000
NUM_OBSERVATIONS = 14
NUM_ACTIONS = 11
NUM_PLAYERS = 2
SIZE = 8

wandb.init(
    project="Attack_on_LUX",
    config={
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "eps": EPS,
        "num_actions": NUM_ACTIONS,
        "num_observations": NUM_OBSERVATIONS,
        "num_players": NUM_PLAYERS,
        "size": SIZE,
        "architecture": "CNN",
        "total_episodes": TOTAL_EPISODES,
    }
)

# action space
'''
[center, up, right, down, left, 
mine, 
transfer,
produce worker, produce warrior,
attack unit,
create factory,
] -> 11 actions

'''

def _normalize_array(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class AttackOnLux(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int, num_agents: int, num_actions: int, render_mode=None):
        self.map = None
        self.size = size
        self.num_agents = num_agents
        self.num_actions = num_actions
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
        self.unit_red_surface = pygame.image.load('gridworld/player_red_2.png').convert()
        self.unit_blue_surface = pygame.image.load('gridworld/player_blue_2.png').convert()
        self.block_surface = pygame.image.load('gridworld/block_2.png').convert()
        self.water_surface = pygame.image.load('gridworld/water_2.png').convert()
        self.spice_surface = pygame.image.load('gridworld/spice_2.png').convert()
        self.factory_red_surface = pygame.image.load('gridworld/factory_red_2.png').convert()
        self.factory_blue_surface = pygame.image.load('gridworld/factory_blue_2.png').convert()

        self.running = True

    def render(self):
        self.display_surface.fill((255, 255, 255))
        
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
        
        for i, player in enumerate(self.players):
            if i % 2 == 0:
                unit_surface = self.unit_blue_surface
                factory_surface = self.factory_blue_surface
            else:
                unit_surface = self.unit_red_surface
                factory_surface = self.factory_red_surface
            
            for agent in player.agents:
                if not agent.health["alive"]:
                    continue
                    
                x, y = agent.position[0], agent.position[1]
                self.display_surface.blit(unit_surface, (x * self.scale, y * self.scale))
            
            for agent in player.factories:
                if not agent.health["alive"]:
                    continue

                x, y = agent.position[0], agent.position[1]
                self.display_surface.blit(factory_surface, (x * self.scale, y * self.scale))
            
            
                
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

    def step(self):
        self.turn += 1

        for resource in self.map.resources:
            resource.step()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_observations(self, player_id: int, agent_id: Optional[str] = None):
        observations = {
            key: val.copy() #if val.ndim == 3 else val[:, :, :self.size, :self.size].copy() * 0
            for key, val in self.observation_space.sample().items()
        }

        if agent_id is None:
            agent_id = self.players[player_id].agents[0].id

        # TODO custom features for each agent???

        observations["worker_cargo_water"][0, x, y]

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
        
        # Normalize features?
        observations["worker_cargo_spice"] = _normalize_array(observations["worker_cargo_spice"])
        observations["worker_cargo_water"] = _normalize_array(observations["worker_cargo_water"])
        observations["worker_health"] = _normalize_array(observations["worker_health"])
        observations["warrior_cargo_water"] = _normalize_array(observations["warrior_cargo_water"])
        observations["warrior_health"] = _normalize_array(observations["warrior_health"])
        observations["factory_health"] = _normalize_array(observations["factory_health"])
        observations["water"] = _normalize_array(observations["water"])
        observations["spice"] = _normalize_array(observations["spice"])
        observations["empty"] = _normalize_array(observations["empty"])

        return observations

    def reset(self, player_id: int = 0, agent_id: Optional[str] = None):
        self.map = Map(size=self.size, num_agents=self.num_agents)
        self.players = [
            Player(
                agents=[agent for agent in self.map.agents if agent.player == str(player_id)],
                enemy_agents=[agent for agent in self.map.agents if agent.player != str(player_id)],
                factories=[factory for factory in self.map.factories if factory.player == str(player_id)],
                enemy_factories=[factory for factory in self.map.factories if factory.player != str(player_id)],
                player_id=str(player_id), resources=self.map.resources, map_size=self.size,
                num_actions=self.num_actions)
            for player_id in range(self.num_agents)
        ]
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
        self.learning_rate = LEARNING_RATE  # Learning rate for policy optimization
        self.gamma = GAMMA  # Discount factor
        self.eps = EPS  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    # def sample_action(self, state: np.ndarray) -> float:
    #     """Returns an action, conditioned on the policy and observation.

    #     Args:
    #         state: Observation from the environment

    #     Returns:
    #         action: Action to be performed
    #     """
    #     state = torch.tensor(np.array([state]))
    #     action_means, action_stddevs = self.net(state)

    #     # create a normal distribution from the predicted
    #     #   mean and standard deviation and sample an action
    #     distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
    #     action = distrib.sample()
    #     prob = distrib.log_prob(action)

    #     action = action.cpu().numpy()

    #     self.probs.append(prob)

    #     return action
    
    def sample_action(self, state: np.ndarray) -> tuple:
        # Sample an action based on the action probabilities
        state = torch.from_numpy(np.array([state]))
        action_probs = self.net(state)
        m = Categorical(action_probs)

        action = m.sample()
        log_prob = m.log_prob(action)

        self.probs.append(log_prob)
        
        return action.item(), log_prob

    def update(self) -> float:
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        policy_loss = []
        for log_prob, delta in zip(self.probs, deltas):
            policy_loss.append(-log_prob * delta)  # negative sign for gradient ascent

        self.optimizer.zero_grad()
        policy_loss = torch.sum(torch.stack(policy_loss))  # used stack instead of cat
        policy_loss.backward()
        self.optimizer.step()
        
        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

        return policy_loss.item()

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

        self.conv1 = nn.Conv2d(obs_space_dims, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 20, kernel_size=1, stride=1)
        
        self.fc = nn.Linear(20*SIZE*SIZE, 256)  # flattening output of conv layers
        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 64)

        self.policy_mean_net = nn.Linear(64, action_space_dims)
        self.policy_stddev_net = nn.Linear(64, action_space_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size()[0], -1)  # flattening
        x = F.relu(self.fc(x))
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        
        action_means = self.policy_mean_net(x)
        # action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(x)))

        action_probs = softmax(action_means, dim=-1)

        # return action_means, action_stddevs

        return action_probs


attackOnLux = AttackOnLux(size=SIZE, num_agents=NUM_PLAYERS, num_actions=NUM_ACTIONS)
attackOnLux.init()
observations = attackOnLux.reset(player_id=0, agent_id=None)
done = False


model = torch.load("models/9000_cnn_1.ckpt") #REINFORCE(NUM_OBSERVATIONS, NUM_ACTIONS)

for episode in range(TOTAL_EPISODES):
    has_reset = False
    done = False

    while not done:
        for player_id_turn in range(attackOnLux.num_agents):
            for i, p in enumerate(attackOnLux.players):
                done = p.is_game_over()

                if done and i == player_id_turn and len(model.rewards) > 0:
                    model.rewards[-1] += 100
                elif done and len(model.rewards) > 0:
                    model.rewards[-1] -= 100
            
            start_idx = len(model.rewards)
            
            for agent in attackOnLux.players[player_id_turn].agents:
                # Update friendly+enemy agents before move.
                if not has_reset:
                    observations = attackOnLux.reset(player_id=player_id_turn, agent_id=agent.id)
                    has_reset = True
                else:
                    attackOnLux.players[player_id_turn].enemy_agents = [agent for agent in attackOnLux.map.agents if agent.player != str(player_id_turn)]
                    attackOnLux.players[player_id_turn].enemy_factories = [factory for factory in attackOnLux.map.factories if factory.player != str(player_id_turn)]

                    observations = attackOnLux.get_observations(player_id=player_id_turn, agent_id=agent.id)
                
                np_observations = np.concatenate([x for x in observations.values()], axis=0)

                action_idx = model.sample_action(np_observations)[0]

                rewards, done = attackOnLux.players[player_id_turn].execute_actions(action_idx, agent)

                attackOnLux.players[player_id_turn].enemy_agents = [agent for agent in attackOnLux.map.agents if agent.player != str(player_id_turn)]
                attackOnLux.players[player_id_turn].enemy_factories = [factory for factory in attackOnLux.map.factories if factory.player != str(player_id_turn)]

                if len(attackOnLux.players[player_id_turn].enemy_agents) == 0:
                    rewards += 100 # Win reward (loss penalty is on execute_actions)
                    done = True

                model.rewards.append(rewards)

                new_map_agents = []
                new_map_factories = []

                for player in attackOnLux.players:
                    new_map_agents.extend(player.agents)
                    new_map_factories.extend(player.factories)
                
                attackOnLux.map.agents = new_map_agents
                attackOnLux.map.factories = new_map_factories

                if done:
                    break
            
            end_idx = len(model.rewards)
            
            for agent in attackOnLux.players[player_id_turn].enemy_agents:
                agent.step()
            
            for factory in attackOnLux.players[player_id_turn].factories:
                was_alive = factory.health["alive"]

                factory.step()

                if not factory.health["alive"] and was_alive:
                    # 5 is the constant for factory breakage, apply to all cumulative moves
                    factory_break_penalty = 5 / (end_idx - start_idx)
                    for i in range(start_idx, end_idx):
                        model.rewards[i] -= factory_break_penalty
            
            for factory in attackOnLux.players[player_id_turn].enemy_factories:
                factory.step()
            
            attackOnLux.players[player_id_turn].agents = [agent for agent in attackOnLux.map.agents if agent.player == str(player_id_turn)]
            attackOnLux.players[player_id_turn].factories = [factory for factory in attackOnLux.map.factories if factory.player == str(player_id_turn)]
            
            attackOnLux.step()
            attackOnLux.render()

            if done:
                break
        
        if keyboard.is_pressed(keyboard.KEY_UP):
            time.sleep(0.15)
    
    avg_rewards = sum(model.rewards) / len(model.rewards)
    game_length = len(model.rewards) # len(model.rewards) also represents number of moves

    loss_item = model.update()

    wandb.log({
        "episode": episode,
        "avg_rewards": avg_rewards,
        "game_length": game_length, 
        "loss": loss_item
    })

    # if episode % 1000 == 0:
    #     print("saving model...")
    #     torch.save(model, f"./models/{episode}_cnn_1.ckpt")
