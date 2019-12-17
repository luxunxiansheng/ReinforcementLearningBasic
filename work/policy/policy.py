import sys
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class Action_Selector(ABC):
    def select_action(self, values):
        value_probs = self.get_probability_distribution(values)
        action_index = np.random.choice(np.arange(len(value_probs)), p=value_probs)
        return action_index
    
    def get_probability(self,values,action_index):
        return self.get_probability_distribution(values)[action_index]
    
    @abstractmethod
    def get_probability_distribution(self,action_values):
        pass


class Random_Action_Selector(Action_Selector):
    def get_probability_distribution(self,values):
        num_values = len(values)
        value_probs = np.ones(num_values, dtype=float) / num_values
        return value_probs
        

class Greedy_Action_Selector(Action_Selector):

    def get_probability_distribution(self,values):
        num_values = len(values)
        value_probs = np.zeros(num_values, dtype=float)
        best_action_index = np.argmax(values)
        value_probs[best_action_index] = 1.0
        return value_probs
        


class e_Greedy_Action_Selector(Action_Selector):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def get_probability_distribution(self,values):
        num_values = len(values)
        value_probs = np.ones(num_values, dtype=float) *self.epsilon / num_values
        best_action_index = np.argmax(values)
        value_probs[best_action_index] += (1.0 - self.epsilon)
        return value_probs

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
        self.method.evaluate(self.policy_table)

    
    def improve (self):
        self.method.improve(self.policy_table)

    
    def show_policy(self):
        outfile = sys.stdout
        for state_index, probability_values in self.policy_table .items():
            outfile.write("\n\nstate_index {:2d}\n".format(state_index))
            for action_index,probability_value in probability_values.items():
                outfile.write("        action_index {:2d} : value {}     ".format(action_index,probability_value))
            outfile.write("\n")
        outfile.write('--------------------------------------------------------------------------\n')
    
    
    
