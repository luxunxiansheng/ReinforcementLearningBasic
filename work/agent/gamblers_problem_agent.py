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
    
      
    def improve_policy_once(self):
        dynamic_programming.policy_improve(self.policy)

    def evaluate_policy_once(self):
        dynamic_programming.policy_evaluate(self.policy,self.env)
        
    def value_iteration(self):
        dynamic_programming.value_iteration(self.policy,self.env)

    
    def make_decision(self):
        pass

         




        
        