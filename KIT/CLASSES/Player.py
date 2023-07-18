import functools
import itertools
import random
from typing import Optional, Union, List, Tuple, TypedDict
import numpy as np
from KIT.CLASSES.Resource import Resource, ResourceType
from KIT.CLASSES.Factory import Factory
from KIT.CLASSES.Unit import Unit, produce_warrior_unit, UnitMovement, UnitType
from KIT.CLASSES.Factory import create_factory
import time
import torch


class Player:

    def __init__(self, *, player_id: str, agents: List[Unit], enemy_agents: List[Unit],
                 factories: List[Factory], enemy_factories: List[Factory],
                 resources: List[Resource], map_size: int, num_actions: int):
        self.player = player_id
        self.factories: List[Factory] = factories
        self.agents: List[Unit] = agents
        self.enemy_agents: List[Unit] = enemy_agents
        self.enemy_factories: List[Factory] = enemy_factories
        self.resources: List[Resource] = resources
        self.map_size = map_size
        self.num_actions = num_actions
        self.moves = 0

    def __get_factory(self, agent: Unit):
        return next((item for item in self.factories if np.array_equal(item.position, agent.position) and item.player == self.player),
                    None)

    def __get_enemy_factory(self, agent: Unit):
        return next(
            (item for item in self.enemy_factories if np.array_equal(item.position, agent.position) and item.player == self.player), None)

    def __get_resource(self, agent: Unit):
        return next((item for item in self.resources if np.array_equal(item.position, agent.position)), None)

    def __get_enemy_agent(self, agent: Unit):
        return next((item for item in self.enemy_agents if np.array_equal(item.position, agent.position)), None)
    
    def calculate_rewards(self):
        rewards = 0
        for agent in self.agents:
            rewards += agent.health["amount"]  # reward for survival
        
        for agent in self.enemy_agents:
            rewards += (100 - agent.health["amount"]) * 0.25
        
        # rewards += 100 * (1.25**self.moves)  # reward for survival
        
        return rewards

    def execute_actions(self, actions, agent_id: int):
        '''
        [center, up, right, down, left, 
        mine spice, 
        transfer,
        produce worker, produce warrior,
        attack unit
        ] -> 10 actions
        '''

        for factory in self.factories:
            factory.step()

        if True:
            agent = self.agents[agent_id]
            action_idx = np.argmax(actions)
            assert action_idx < self.num_actions, f"action out of range expected 0 < action < {self.num_actions} got {action_idx}"

            if action_idx < 5:
                # print(f"agent {agent.id} moving {UnitMovement(action_idx).name}")
                agent.move(action_idx)
            if action_idx == 5:
                resource = self.__get_resource(agent)
                if resource is not None:
                    # print(f"agent {agent.id} mining resource {resource.id} of type {resource.resource_type.name}")
                    agent.mine(resource)
            if action_idx == 6:
                factory = self.__get_factory(agent)
                if factory is not None:
                    # print(f"agent {agent.id} transferring resource to factory {factory.id}")
                    agent.transfer(factory)
            if action_idx == 7:
                factory = self.__get_factory(agent)
                if factory is not None:
                    # print(f"factory {factory.id} producing worker")
                    new_worker = factory.produce_unit(UnitType.WORKER)
                    if new_worker is not None:
                        self.agents.append(new_worker)
            if action_idx == 8:
                factory = self.__get_factory(agent)
                if factory is not None:
                    # print(f"factory {factory.id} producing warrior")
                    new_warrior = factory.produce_unit(UnitType.WARROR)

                    if new_warrior is not None:
                        self.agents.append(new_warrior)
            if action_idx == 9:
                enemy_agent = self.__get_enemy_agent(agent)
                if enemy_agent is not None:
                    # print(f"agent {agent.id} attacking enemy agent {enemy_agent.id}")
                    agent.attack(enemy_agent)
            if action_idx == 10:
                factory = self.__get_factory(agent)
                enemy_agent = self.__get_enemy_agent(agent)
                enemy_factory = self.__get_enemy_factory(agent)
                if factory is not None and enemy_agent is not None and enemy_factory is not None:
                    # print(f"agent creating factory @ position {agent.position}")
                    self.factories.append(
                        create_factory(self.player, agent.position, map_size=self.map_size)
                    )
            
        for agent in self.agents:
            agent.step()
        
        self.agents = [agent for agent in self.agents if agent.health["alive"]]
        self.factories = [factory for factory in self.factories if factory.health["alive"]]
        
        self.moves += 1
        
        rewards = self.calculate_rewards()
        
        return rewards, self.is_game_over()

    def get_observation(self) -> dict:
        # Return the current observation for the agent
        # Implement your own observation logic
        obs = {self.player: {}}

        workers = [agent for agent in self.agents if agent.unit_type == UnitType.WORKER]
        warriors = [agent for agent in self.agents if agent.unit_type == UnitType.WARROR]

        obs[self.player]['worker'] = [{
            'position': x.position,
            'stats': x.stats,
            'health': x.health,
            'mining_options': x.mining_options
        } for x in workers]

        obs[self.player]['warrior'] = [{
            'position': x.position,
            'stats': x.stats,
            'health': x.health,
            'mining_options': x.mining_options
        } for x in warriors]

        obs[self.player]['factory'] = [{
            'position': x.position,
            'stats': x.stats,
            'health': x.health
        } for x in self.factories]

        return obs

    def choose_action(self, observation) -> int:
        # Get the indices of the available actions (where value is 1)
        available_actions = [index for index, value in enumerate([None] * self.num_actions)]
        # Randomly sample an action from the available actions
        random_action = random.choices(available_actions)[0]
        return random_action

    def is_game_over(self) -> bool:
        # Check if the game is over for the agent
        # Implement your own game over logic

        num_alive_factories = sum(x.health["alive"] for x in self.factories)
        num_alive_agents = sum(x.health["alive"] for x in self.agents)

        if num_alive_agents == 0: #  sum(x.health["amount"] for x in self.factories if x.health["amount"] > 6) == 0) ????
            print("GAME OVER")
            print("TURNS SURVIVED", self.moves)
            return True

        return False
