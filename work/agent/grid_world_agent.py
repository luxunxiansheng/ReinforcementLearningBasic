import numpy as np

from algorithm.implicity_policy import dynamic_programming
from agent.base_agent import Base_Agent
from env.grid_world import GridworldEnv
from policy.policy import Random_Action_Selector, Tabular_Implicit_Policy


class Grid_World_Agent(Base_Agent):
    def __init__(self):
        self.env = GridworldEnv([4, 4])
        self.policy = Tabular_Implicit_Policy(
            self.env.action_space.n, Random_Action_Selector())

    def make_decision(self, observation):
        return self.policy.select_action(observation)

    def evaluate_policy(self):
        dynamic_programming.policy_evaluate(self.policy,self.env.P,self.env)
        
    def improve_policy(self):
        pass



