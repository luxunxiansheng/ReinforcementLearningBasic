from copy import deepcopy
from common import ActorBase,CriticBase
from lib.utility import (create_distribution_epsilon_greedily,create_distribution_greedily)
from policy.policy import DiscreteStateValueBasedPolicy


class Critic(CriticBase):
    def __init__(self,q_value_function,step_size):
        self.q_value_function = q_value_function
        self.step_size = step_size
    

    def evaluate(self, *args):
        current_state_index = args[0]
        current_action_index = args[1]
        target = args[2]
    
        delta = target - self.q_value_function[current_state_index][current_action_index]
        self.q_value_function[current_state_index][current_action_index] += self.step_size * delta

    def get_value_function(self):
        return self.q_value_function



class LambdaCritic(CriticBase):
    def __init__(self,q_value_function,step_size,discount,lamb):
        self.q_value_function = q_value_function
        self.step_size = step_size
        self.discount = discount
        self.lamb =lamb

        self.eligibility = deepcopy(q_value_function)
        for state_index in q_value_function:
            for action_index in q_value_function[state_index]:
                self.eligibility[state_index][action_index] = 0.0
        

    def evaluate(self, *args):
        current_state_index = args[0]
        current_action_index = args[1]
        target = args[2]
    
        delta = target - self.q_value_function[current_state_index][current_action_index]
        self.eligibility[current_state_index][current_action_index] += 1.0
            
        # backforward view proprogate 
        for state_index in self.q_value_function:
            for action_index in self.q_value_function[state_index]:
                self.q_value_function[state_index][action_index] = self.q_value_function[state_index][action_index]+self.step_size*delta* self.eligibility[state_index][action_index]
                self.eligibility[state_index][action_index] = self.eligibility[state_index][action_index]*self.discount* self.lamb


    def get_value_function(self):
        return self.q_value_function


class Actor(ActorBase):
    def __init__(self, policy, critic,epsilon):
        self.policy = policy
        self.critic = critic
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, *args):
        current_state_index = args[0]
        q_value_function = self.critic.get_value_function()
        q_values = q_value_function[current_state_index]
        soft_greedy_distibution = self.create_distribution_epsilon_greedily(q_values)
        self.policy.policy_table[current_state_index] = soft_greedy_distibution

    def get_current_policy(self):
        return self.policy

    def get_optimal_policy(self):
        policy_table = {}
        q_value_function = self.critic.get_value_function()
        for state_index, _ in q_value_function.items():
            q_values = q_value_function[state_index]
            greedy_distibution = self.create_distribution_greedily(q_values)
            policy_table[state_index] = greedy_distibution
        table_policy = DiscreteStateValueBasedPolicy(policy_table)
        return table_policy

