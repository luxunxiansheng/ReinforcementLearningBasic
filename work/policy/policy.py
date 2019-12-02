from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class Action_Selector(ABC):
    def select_action(self, action_values):
        action_probs = self.get_probability_distribution(action_values)
        action_index = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action_index
    
    def get_probability(self,action_values,action_index):
        return self.get_probability_distribution(action_values)[action_index]
    
    @abstractmethod
    def get_probability_distribution(self,action_values):
        pass


class Random_Action_Selector(Action_Selector):
    def get_probability_distribution(self,action_values):
        num_actions = len(action_values)
        action_probs = np.ones(num_actions, dtype=float) / num_actions
        return action_probs
        

class Greedy_Action_Selector(Action_Selector):

    def get_probability_distribution(self,action_values):
        num_actions = len(action_values)
        action_probs = np.zeros(num_actions, dtype=float)
        best_action_index = np.argmax(action_values)
        action_probs[best_action_index] = 1.0
        return action_probs
        


class e_Greed_Action_Selector(Action_Selector):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def get_probability_distribution(self,action_values):
        num_actions = len(action_values)
        action_probs = np.ones(num_actions, dtype=float) *self.epsilon / num_actions
        best_action = np.argmax(action_values)
        action_probs[best_action] += (1.0 - self.epsilon)
        return action_probs

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


class Tabular_Implicit_Policy(Policy):
    """
    The policy is implicit for the action is selected from tabular value funciton.
    """
    
    def __init__(self, num_actions, action_selector):
        self.Q_table = defaultdict(lambda: np.zeros(num_actions))
        self.action_selector=action_selector

    def select_action(self, state):
        action_values=self.Q_table[state]
        return self.action_selector.select_action(action_values)
    
    def get_probability(self,state,action_index):
        action_values = self.Q_table[state]
        return self.action_selector.get_probability(action_values,action_index)
        


    