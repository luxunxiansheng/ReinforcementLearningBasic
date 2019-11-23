import numpy as np
import copy
from collections import defaultdict

from base_agent import Base_Agent
from env.grid_world import GridworldEnv
from policy.tabular_policy.random_policy import Random_Policy


class Grid_World_Agent(Base_Agent):
    def __init__(self):
        self.env = GridworldEnv([4, 4])
        self.Q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.policy = Random_Policy(self.Q_table)

    def make_decision(self, observation):
        return self.policy.select_action(observation)

    def evaluate_policy(self):

        shape = self.env.shape
        nS = self.env.nS
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            grid_index = it.iterindex
            transition = self.env.P[grid_index]
            
            self.policy.evaluate(grid_index,transition) 
            
            it.iternext()
