import numpy as np

from base_agent import Base_Agent
from env.grid_world import GridworldEnv
from policy.policy import Tabular_Implicit_Policy,Random_Action_Selector



class Grid_World_Agent(Base_Agent):
    def __init__(self):
        self.env = GridworldEnv([4, 4])
        self.policy = Tabular_Implicit_Policy(self.env.action_space.n,Random_Action_Selector())

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

            self.policy.evaluate(grid_index, transition)

            it.iternext()
