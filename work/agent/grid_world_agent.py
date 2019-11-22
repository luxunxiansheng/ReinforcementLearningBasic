import numpy as np

from base_agent import Base_Agent
from env.grid_world import GridworldEnv
from policy.tabular_policy.random_policy import Random_Policy


class Grid_World_Agent(Base_Agent):
    def __init__(self):
        self.policy = Random_Policy()
        self.env    = GridworldEnv([4,4])

    def make_decision(self,observation):
        return self._policy.select_action(observation)   

    
    def evaluate_policy(self):
       
        shape = self._env.shape
        nS =    self._env.nS

        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        MAX_Y = shape[0]
        MAX_X = shape[1]

        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

        while not it.finished:
            state_index = it.iterindex
            row_index, col_index = it.multi_index
            
            self.policy.Q_table(state_index) = 


           
            it.iternext()

       
