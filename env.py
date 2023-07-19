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
import copy
import keyboard
import torch.nn.functional as F

# action space
'''
[center, up, right, down, left, 
mine, 
transfer,
produce worker, produce warrior,
attack unit,
create factory,
select_next_agent
] -> 12 actions

'''


class AttackOnLux(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int, num_agents: int, render_mode=None):
        self.map = None
        self.size = size
        self.num_agents = num_agents
        self.num_actions = 12
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
                x, y = agent.position[0], agent.position[1]
                self.display_surface.blit(unit_surface, (x * self.scale, y * self.scale))
            
            for agent in player.factories:
                x, y = agent.position[0], agent.position[1]
                self.display_surface.blit(factory_surface, (x * self.scale, y * self.scale))
            
            
                
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

    def step(self, action_idx, player_id: int, agent_id: int, selectable_agents=[]):
        done = False
        rewards, agent_done = self.players[player_id].execute_actions(action_idx, agent_id)
        done = done or agent_done
        # if self.turn > 50:
        #     done = True
        self.turn += 1

        for resource in self.map.resources:
            resource.step()

        return self.get_observations(player_id, agent_id, selectable_agents), rewards, done, False, {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_observations(self, player_id: int, agent_id: Optional[str] = None, selectable_agents=[]):
        observations = {
            key: val.copy() #if val.ndim == 3 else val[:, :, :self.size, :self.size].copy() * 0
            for key, val in self.observation_space.sample().items()
        }

        if agent_id is None:
            agent_id = self.players[player_id].agents[0].id

        # TODO custom features for each agent???

        player_observation = self.players[player_id].get_observation()
        for agent in player_observation[str(player_id)]["worker"]:
            x, y = agent["position"][0], agent["position"][1]
            observations["worker"][0, x, y] = 1

            for i, selectable_agent in enumerate(selectable_agents):
                if agent["id"] == selectable_agent.id:
                    observations["selectable_worker"][0, x, y] = i

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

            for i, selectable_agent in enumerate(selectable_agents):
                if agent["id"] == selectable_agent.id:
                    observations["selectable_warrior"][0, x, y] = i

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

    def reset(self, player_id: int = 0, agent_id: Optional[str] = None, selectable_agents=[]):
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
        return self.get_observations(player_id, agent_id, selectable_agents)

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
        self.learning_rate = 1e-5  # Learning rate for policy optimization
        self.gamma = 0.8  # Discount factor
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

        self.conv1 = nn.Conv2d(obs_space_dims, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc = nn.Linear(128*8*8, 512)  # flattening output of conv layers
        self.policy_mean_net = nn.Linear(512, action_space_dims)
        self.policy_stddev_net = nn.Linear(512, action_space_dims)

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
        
        action_means = self.policy_mean_net(x)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(x)))

        return action_means, action_stddevs


attackOnLux = AttackOnLux(render_mode="human", size=8, num_agents=2)
attackOnLux.init()
observations = attackOnLux.reset(player_id=0, agent_id=None)
done = False

total_num_episodes = 100000  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
#obs_space_dims = attackOnLux.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
#action_space_dims = attackOnLux.action_space.shape[0]
rewards_over_seeds = []

# set seed
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)

# Reinitialize model every seed
model = REINFORCE(17, 12)
#neg_model = REINFORCE(64*15, 11)
cumulative_moves = []

for episode in range(total_num_episodes):
    # gymnasium v26 requires users to set seed while resetting the environment

    observations = None
    has_reset = False
    done = False
    

    while not done:
        for player_id_turn in range(attackOnLux.num_agents):
            selected_agent = False
            must_move_agent = False
            selectable_agents = attackOnLux.players[player_id_turn].agents
            selectable_agent_probs = []
            agent_idx = 0

            attackOnLux.players[player_id_turn].enemy_agents = [agent for agent in attackOnLux.map.agents if agent.player != str(player_id_turn)]
            attackOnLux.players[player_id_turn].enemy_factories = [factory for factory in attackOnLux.map.factories if factory.player != str(player_id_turn)]
            
            random.shuffle(selectable_agents) # remove bias

            while not selected_agent and len(selectable_agents) > 0:
                if not has_reset:
                    observations = attackOnLux.reset(player_id=player_id_turn, agent_id=None, selectable_agents=selectable_agents)
                    has_reset = True
                else:
                    observations = attackOnLux.get_observations(player_id=player_id_turn, agent_id=selectable_agents[agent_idx].id, selectable_agents=selectable_agents)
                
                np_observations = np.concatenate([x for x in observations.values()], axis=0)

                actions = model.sample_action(np_observations)
                actions = np.exp(actions) / np.exp(actions).sum() # softmax

                if must_move_agent:
                    actions = actions[:-1]
                
                action_idx = np.argmax(actions)
                
                # attackOnLuxClone = copy.copy(attackOnLux)
                # _, rewards_clone, _, _, _ = attackOnLuxClone.step(action_idx, player_id_turn, selectable_agents[agent_idx].id, selectable_agents)


                if len(selectable_agents) <= 1:
                    # Must move, don't bother recomputing
                    if action_idx == 11:
                        model.rewards.append(-100)
                    
                    actions = actions[:-1]
                    action_idx = np.argmax(actions)

                    observations, rewards, done, _, _ = attackOnLux.step(action_idx, player_id_turn, selectable_agents[agent_idx].id, selectable_agents)

                    attackOnLux.players[player_id_turn].enemy_agents = [agent for agent in attackOnLux.map.agents if agent.player != str(player_id_turn)]
                    attackOnLux.players[player_id_turn].enemy_factories = [factory for factory in attackOnLux.map.factories if factory.player != str(player_id_turn)]

                    if len(attackOnLux.players[player_id_turn].enemy_agents) == 0:
                        rewards += 1000

                    model.rewards.append(rewards)
                    selected_agent = True
                elif action_idx == 11:
                    model.rewards.append(5)

                    # selectable_agent_probs.append([agent_idx, actions, rewards_clone])

                    agent_idx += 1

                    if agent_idx >= len(selectable_agents):
                    #     selectable_agent_probs = sorted(selectable_agent_probs, key=lambda x: x[1][-1])
                    #     agent_idx = selectable_agent_probs[0][0] # agent with lowest probs of not being selected
                        agent_idx -= 1
                        must_move_agent = True
                else:
                    if len(selectable_agent_probs) > 0:
                        best_agent = max(selectable_agent_probs, key = lambda x: x[2])
                        if best_agent[0] == agent_idx:
                            rewards += 50
                        else:
                            rewards -= 15

                    
                    observations, rewards, done, _, _ = attackOnLux.step(action_idx, player_id_turn, selectable_agents[agent_idx].id, selectable_agents)

                    attackOnLux.players[player_id_turn].enemy_agents = [agent for agent in attackOnLux.map.agents if agent.player != str(player_id_turn)]
                    attackOnLux.players[player_id_turn].enemy_factories = [factory for factory in attackOnLux.map.factories if factory.player != str(player_id_turn)]

                    if len(attackOnLux.players[player_id_turn].enemy_agents) == 0:
                        rewards += 1000

                    model.rewards.append(rewards)
                    selected_agent = True

            if done:
                break

            attackOnLux.render()
        
        if keyboard.is_pressed(keyboard.KEY_UP):
            time.sleep(0.15)

    cumulative_moves.append(attackOnLux.players[0].moves * 2)
    
    if episode % 50 == 0:
        print("RWD", sum(model.rewards[-10:]) / 10, "AVERAGE MOVES", sum(cumulative_moves) / len(cumulative_moves))
        pygame.display.set_caption(f"RWDS {(sum(model.rewards[-10:]) / 10):.2f} MVS {(sum(cumulative_moves) / len(cumulative_moves)):.2f}")
        cumulative_moves = []

    model.update()

    if episode % 1000 == 0:
        print("saving model...")
        torch.save(model, f"./models/{episode}_3.ckpt")
        
    
rewards_to_plot = rewards_over_seeds
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()
