import numpy as np

from agent.base_agent import Base_Agent
from policy.policy import TabularPolicy
from lib.utility import create_distribution_randomly


"""
A Dynamic Programming agent lives as almgihty God, who colud predict everything next 
to happen no matter what ations to be taken. The transiton table tells him the dynamic 
of the world.    
"""

class DP_Agent(Base_Agent):
    def __init__(self, env, method):
        self.env = env
        self.transitions = env.P
        self.policy = TabularPolicy(self._init_random_policy_table())
        self.method = method

    def _init_random_policy_table(self):
        policy_table = self.env.build_policy_table()
        for state_index, action_probablities in policy_table.items():
            distribution = create_distribution_randomly()(action_probablities)
            policy_table[state_index] = distribution
        return policy_table

    def select_action(self, observation):
        return self.policy.get_action(observation)

    def evaluate(self):
        self.method.evaluate(self)

    def improve(self):
        self.method.improve(self)
