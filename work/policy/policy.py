import sys
from abc import ABC, abstractmethod
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


class Policy(ABC):
    """
    A policy defines the learning agent's way of behaving at a given time. Roughly speaking,
    a policy is a mapping from perceived states of the environment to actions to be taken
    when in those states. It corresponds to what in psychology would be called a set of
    stimulus-response rules or associations. In some cases the policy may be a simple function
    or lookup table, whereas in others it may involve extensive computation such as a search
    process. The policy is the core of a reinforcement learning agent in the sense that it alone
    is suficient to determine behavior. In general, policies may be stochastic.

    """

    @abstractmethod
    def select_action(self, state):
        pass


class Tabular_Policy(Policy):
    def __init__(self,pi,method):
        self.policy_table= pi
        self.method = method

    def select_action(self,state):
        action_probs = list(self.policy_table[state].values())
        action_index = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action_index     

    
    def evaluate(self):
        self.method.evaluate(self)

    
    def improve (self):
        return self.method.improve(self)

    
    def show_policy_on_console(self):
        outfile = sys.stdout
        for state_index, probability_values in self.policy_table.items():
            outfile.write("\n\nstate_index {:2d}\n".format(state_index))
            for action_index,probability_value in probability_values.items():
                outfile.write("        action_index {:2d} : value {}     ".format(action_index,probability_value))
            outfile.write("\n")
        outfile.write('--------------------------------------------------------------------------\n')
    
    def show_policy(self):
        
        x=  []
        y = []
        for state_index, _ in self.policy_table.items():
            x.append(state_index)
            y.append(self.select_action(state_index))  
        
        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.suptitle('Action taken at state')

        # Plot the data
        ax.scatter(x,y)

        # Show the grid lines as dark grey lines
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        plt.show()
