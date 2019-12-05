import sys
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
        dynamic_programming.policy_evaluate(self.policy,self.env)
        
    def value_iteration(self):
        dynamic_programming.value_iteration(self.policy,self.env)

    def improve_policy(self):
        dynamic_programming.policy_improve(self.policy)

    def show_state_value(self):
        outfile = sys.stdout
        it = np.nditer(self.env.grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex

            optimal_value_of_state = dynamic_programming.get_value_of_state(self.policy,s)
            output = "{0:.2f} ******".format(optimal_value_of_state) 
           
            
            _, x = it.multi_index

            if x == 0:
                output = output.lstrip()
            if x == self.env.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.env.shape[1] - 1:
                outfile.write("\n")

            it.iternext()


    def show_optimal_state_values(self):
        outfile = sys.stdout
        it = np.nditer(self.env.grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex

            optimal_value_of_state = dynamic_programming.get_optimal_value_of_state(self.policy,s)
            output = "{0:.2f} ******".format(optimal_value_of_state) 
           
            
            _, x = it.multi_index

            if x == 0:
                output = output.lstrip()
            if x == self.env.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.env.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
