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
            (item for item in self.enemy_factories if np.array_equal(item.position, agent.position) and item.player != self.player), None)

    def __get_resource(self, agent: Unit):
        return next((item for item in self.resources if np.array_equal(item.position, agent.position)), None)

    def __get_enemy_agent(self, agent: Unit):
        return next((item for item in self.enemy_agents if np.array_equal(item.position, agent.position)), None)

    def execute_actions(self, action_idx, agent_id: str):
        '''
        [center, up, right, down, left, 
        mine spice, 
        transfer,
        produce worker, produce warrior,
        attack unit
        ] -> 10 actions
        '''

        self.agents = [agent for agent in self.agents if agent.health["alive"]]
        self.factories = [factory for factory in self.factories if factory.health["alive"]]

        rewards = 0

        for factory in self.factories:
            factory.step()

        agent = [agent for agent in self.agents if agent.id==agent_id]
        if len(agent) == 0:
            return -1, self.is_game_over()
        else:
            agent = agent[0]
        
        assert action_idx < self.num_actions, f"action out of range expected 0 < action < {self.num_actions} got {action_idx}"

        if action_idx < 5:
            print(f"agent {agent.id} moving {UnitMovement(action_idx).name}")
            previous_position = agent.position

            agent.move(action_idx)

            if np.array_equal(agent.position, previous_position):
                rewards -= 15

            if action_idx == 0:
                rewards -= 1
            
            if agent.health["amount"] <= 20:
                rewards -= 25
        if action_idx == 5:
            resource = self.__get_resource(agent)
            if resource is not None:
                print(f"agent {agent.id} mining resource {resource.id} of type {resource.resource_type.name}")
                agent.mine(resource)
                rewards += 1
            else:
                rewards -= 25
        if action_idx == 6:
            factory = self.__get_factory(agent)
            if factory is not None:
                print(f"agent {agent.id} transferring resource to factory {factory.id}")
                for item in agent.mining_options:
                    if item["type"] == ResourceType.SPICE:
                        rewards += item["amount"]
                        break
                agent.transfer(factory)
            else:
                rewards -= 50
        if action_idx == 7:
            factory = self.__get_factory(agent)
            if factory is not None:
                print(f"factory {factory.id} producing worker")
                new_worker = factory.produce_unit(UnitType.WORKER)
                if new_worker is not None:
                    self.agents.append(new_worker)
                    rewards += 100
            else:
                rewards -= 50
        if action_idx == 8:
            factory = self.__get_factory(agent)
            if factory is not None:
                print(f"factory {factory.id} producing warrior")
                new_warrior = factory.produce_unit(UnitType.WARROR)

                if new_warrior is not None:
                    self.agents.append(new_warrior)
                    rewards += 100
            else:
                rewards -= 50
        if action_idx == 9:
            enemy_agent = self.__get_enemy_agent(agent)

            if enemy_agent is None:
                factory = True
                enemy_agent = self.__get_enemy_factory(agent)
            factory = False

            if enemy_agent is not None:
                print(f"agent {agent.id} attacking enemy agent {enemy_agent.id}")
                enemy_agent.attack(agent)
                rewards += 25

                print("HIT", "FACTORY" if factory else "UNIT")

                if not enemy_agent.health["alive"]:
                    print("MURDER", "FACTORY" if factory else "UNIT")
                    rewards += 100
                if factory:
                    rewards += 50
            else:
                rewards -= 25
                
        if action_idx == 10:
            factory = self.__get_factory(agent)
            enemy_agent = self.__get_enemy_agent(agent)
            enemy_factory = self.__get_enemy_factory(agent)
            if factory is None and enemy_agent is None and enemy_factory is None:
                for item in agent.mining_options:
                    if item['type'] == ResourceType.SPICE and item is not None and item["amount"] >= 20:
                        print(f"agent creating factory @ position {agent.position}")
                        item["amount"] -= 20
                        self.factories.append(
                            create_factory(self.player, agent.position, map_size=self.map_size)
                        )
                        rewards += 100
                    elif item['type'] == ResourceType.SPICE and item is not None:
                        rewards -= 25
            else:
                rewards -= 25
            
        for agent in self.agents:
            was_alive = agent.health["alive"]

            agent.step()

            if not agent.health["alive"] and was_alive:
                rewards -= 100
        
        self.agents = [agent for agent in self.agents if agent.health["alive"]]
        self.factories = [factory for factory in self.factories if factory.health["alive"]]
        
        self.moves += 1

        done = self.is_game_over()
        if done:
            rewards -= 1000
        
        return rewards, done

    def get_observation(self) -> dict:
        # Return the current observation for the agent
        # Implement your own observation logic
        obs = {self.player: {}}

        workers = [agent for agent in self.agents if agent.unit_type == UnitType.WORKER]
        warriors = [agent for agent in self.agents if agent.unit_type == UnitType.WARROR]

        obs[self.player]['worker'] = [{
            'id': x.id,
            'position': x.position,
            'stats': x.stats,
            'health': x.health,
            'mining_options': x.mining_options
        } for x in workers]

        obs[self.player]['warrior'] = [{
            'id': x.id,
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

    def is_game_over(self) -> bool:
        # Check if the game is over for the agent
        # Implement your own game over logic

        num_alive_factories = sum(x.health["alive"] for x in self.factories)
        num_alive_agents = sum(x.health["alive"] for x in self.agents)

        if num_alive_agents == 0: #  sum(x.health["amount"] for x in self.factories if x.health["amount"] > 6) == 0) ????
            return True

        return False
