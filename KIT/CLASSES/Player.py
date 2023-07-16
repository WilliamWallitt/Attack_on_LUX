import functools
import itertools
import random
from typing import Optional, Union, List, Tuple, TypedDict
import numpy as np
from KIT.CLASSES.Resource import Resource, ResourceType
from KIT.CLASSES.Factory import Factory
from KIT.CLASSES.Unit import Unit, produce_warrior_unit, UnitMovement, UnitType
from KIT.CLASSES.Factory import create_factory


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

    def __get_factory(self, agent: Unit):
        return next((item for item in self.factories if item.position == agent.position and item.player == self.id),
                    None)

    def __get_enemy_factory(self, agent: Unit):
        return next(
            (item for item in self.enemy_factories if item.position == agent.position and item.player == self.id), None)

    def __get_resource(self, agent: Unit):
        return next((item for item in self.resources if item.position == agent.position), None)

    def __get_enemy_agent(self, agent: Unit):
        return next((item for item in self.enemy_agents if item.position == agent.position), None)

    def execute_actions(self, actions):
        rewards = []
        '''
        [center, up, right, down, left, 
        mine spice, 
        transfer,
        produce worker, produce warrior,
        attack unit
        ] -> 10 actions
        '''
        for i, agent in enumerate(self.agents):
            action = actions[i]
            action_idx = np.argmax(action)
            if action_idx < 5:
                agent.move(UnitMovement(action_idx))
            if action_idx == 5:
                resource = self.__get_resource(agent)
                if resource is not None:
                    agent.mine(resource)
            if action_idx == 6:
                factory = self.__get_factory(agent)
                if factory is not None:
                    agent.transfer(factory)
            if action_idx == 7:
                factory = self.__get_factory(agent)
                if factory is not None:
                    factory.produce_unit(UnitType.WORKER)
            if action_idx == 8:
                factory = self.__get_factory(agent)
                if factory is not None:
                    factory.produce_unit(UnitType.WARROR)
            if action_idx == 9:
                enemy_agent = self.__get_enemy_agent(agent)
                if enemy_agent is not None:
                    agent.attack(enemy_agent)
            if action_idx == 10:
                factory = self.__get_factory(agent)
                enemy_agent = self.__get_enemy_agent(agent)
                enemy_factory = self.__get_enemy_factory(agent)
                if factory is not None and enemy_agent is not None and enemy_factory is not None:
                    create_factory(self.player, agent.position, map_size=self.map_size)

    def get_observation(self) -> dict:
        # Return the current observation for the agent
        # Implement your own observation logic
        obs = {self.player: {}}

        obs[self.player]['agents'] = [{
            'position': x.position,
            'stats': x.stats,
            'health': x.health
        } for x in self.agents]

        obs[self.player]['factories'] = [{
            'position': x.position,
            'stats': x.stats,
            'health': x.health
        } for x in self.factories]

        obs[self.player]['resources'] = [{
            'position': x.position,
            'amount': x.amount,
            'type': x.resource_type
        } for x in self.resources]

        for key, group in itertools.groupby(self.enemy_agents, lambda x: x.player):
            obs[key] = {}
            obs[key]['agents'] = [{
                'position': x.position,
                'stats': x.stats,
                'health': x.health
            } for x in group]

        for key, group in itertools.groupby(self.enemy_factories, lambda x: x.player):
            obs[key] = {}
            obs[key]['agents'] = [{
                'position': x.position,
                'stats': x.stats,
                'health': x.health
            } for x in group]

        return obs

    def choose_action(self, observation) -> np.ndarray:
        # Get the indices of the available actions (where value is 1)
        available_actions = [index for index, value in enumerate([None] * self.num_actions)]
        # Randomly sample an action from the available actions
        random_action = random.choices(available_actions)[0]
        return np.array(random_action)

    def is_game_over(self) -> bool:
        # Check if the game is over for the agent
        # Implement your own game over logic

        num_alive_factories = sum(x.health["alive"] for x in self.factories)
        num_alive_agents = sum(x.health["alive"] for x in self.agents)

        if num_alive_factories == 0 or (num_alive_agents == 0 and sum(
                x.health["amount"] for x in self.factories if x.health["amount"] > 6) == 0):
            return True

        return False
