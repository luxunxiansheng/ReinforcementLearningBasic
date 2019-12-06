import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from agent.base_agent import Base_Agent
from algorithm.implicity_policy import dynamic_programming
from env.gamblers_problem import GamblersProblemEnv
from policy.policy import Random_Action_Selector, Tabular_Implicit_Policy

matplotlib.use('Agg')


class Gamblers_Problem_Agent(Base_Agent):
    def __init__(self):
        self.env = GamblersProblemEnv()
        self.policy = Tabular_Implicit_Policy(self.env.build_Q_table())
        self.policy_history = []
    
    
    def evaluate_policy_once(self):
        dynamic_programming.policy_evaluate(self.policy,self.env)
        
    def value_iteration_once(self):
        dynamic_programming.value_iteration(self.policy,self.env)

    def improve_policy_once(self):
        dynamic_programming.policy_improve(self.policy)
    
    def make_decision(self):
        pass

    
    """ def show_value_of_state(self):
        plt.figure(figsize=(10, 20))

        plt.subplot(2, 1, 1)
        for sweep, state_value in enumerate(self.policy_history):
            plt.plot(state_value, label='sweep {}'.format(sweep))
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.scatter(STATES, policy)
        plt.xlabel('Capital')
        plt.ylabel('Final policy (stake)')

        
        plt.close() """


def value_iteration():
    gamblers_problem_agent_value_it = Gamblers_Problem_Agent()
    
    for _ in range(10):
        gamblers_problem_agent_value_it.value_iteration_once()
        