from copy import deepcopy
import numpy as np 
from numpy import array
from common import ExploitatorBase, ExploratorBase
from lib.utility import (create_distribution_epsilon_greedily,create_distribution_greedily,create_distribution_boltzmann)
from policy.policy import DiscreteStateValueBasedPolicy

class TDExploitator(ExploitatorBase):
    def __init__(self,value_function,step_size=0.1):
        self.value_function = value_function
        self.step_size = step_size
    
    def update(self, *args):
        # V 
        if self._is_v_function(args):
            current_state_index = args[0]
            target = args[1]
            delta = target - self.value_function[current_state_index]
            self.value_function[current_state_index] += self.step_size * delta
        # Q 
        else:
            current_state_index = args[0]
            current_action_index = args[1]
            target = args[2]

            delta = target - self.value_function[current_state_index][current_action_index]
            self.value_function[current_state_index][current_action_index] += self.step_size * delta

    def _is_v_function(self, args):
        return len(args)==2

    def get_value_function(self):
        return self.value_function



class TDNSARSACritic(TDExploitator):
    def __init__(self,value_function,steps,step_size=0.1,discount=1.0):
        super().__init__(value_function, step_size=step_size)
        self.steps = steps
        self.discount = discount
    
    def evaluate(self,*args):
        trajectory = args[0]
        current_timestamp = args[1]
        updated_timestamp = args[2]
        final_timestamp   = args[3]
        
        # V function
        if len(trajectory[0])==2:
            G = 0
            for i in range(updated_timestamp , min(updated_timestamp + self.steps , final_timestamp)):
                G += np.power(self.discount, i - updated_timestamp ) * trajectory[i][1]
                if updated_timestamp + self.steps < final_timestamp:
                    G += np.power(self.discount, self.steps) * self.get_value_function()[trajectory[current_timestamp][0]]

            self.update(trajectory[updated_timestamp][0],G)
        else:
            G = 0
            for i in range(updated_timestamp, min(updated_timestamp + self.steps, final_timestamp)):
                G += np.power(self.discount, i - updated_timestamp) * trajectory[i][2]
            if updated_timestamp + self.steps < final_timestamp:
                G += np.power(self.discount, self.steps) *  self.get_value_function()[trajectory[current_timestamp][0]][trajectory[current_timestamp][1]]
            self.update(trajectory[updated_timestamp][0],trajectory[updated_timestamp][1],G)

class TDNExpectedSARSACritic(TDExploitator):
    def __init__(self,value_function,policy,steps=1,step_size=0.1,discount=1.0):
        super().__init__(value_function, step_size=step_size)
        self.steps = steps
        self.discount = discount
        self.policy = policy 
    
    def evaluate(self,*args):
        trajectory = args[0]
        current_timestamp = args[1]
        updated_timestamp = args[2]
        final_timestamp   = args[3]
        
        G = 0
        for i in range(updated_timestamp, min(updated_timestamp + self.steps, final_timestamp)):
            G += np.power(self.discount, i - updated_timestamp) * trajectory[i][2]
        if updated_timestamp + self.steps < final_timestamp:
            # expected Q value, actullay the v(s)
            expected_next_q = 0
            next_actions = self.policy.policy_table[trajectory[current_timestamp][0]]
            for action, action_prob in next_actions.items():
                expected_next_q += action_prob * self.get_value_function()[trajectory[current_timestamp][0]][action]
            G += np.power(self.discount, self.steps) * expected_next_q
            

        self.update(trajectory[updated_timestamp][0],trajectory[updated_timestamp][1],G)      


class LambdaCritic(ExploitatorBase):
    def __init__(self,value_function,step_size=0.1,discount=1.0,lamb=0.01):
        self.value_function = value_function
        self.step_size = step_size
        self.discount = discount
        self.lamb =lamb

        self.eligibility = deepcopy(value_function)

        if self._is_q_function(value_function):
            for state_index in value_function:
                for action_index in value_function[state_index]:
                    self.eligibility[state_index][action_index] = 0.0
        else:
            for state_index in value_function:
                self.eligibility[state_index] = 0.0

    def _is_q_function(self, value_function):
        return isinstance(value_function,dict)
        

    def update(self, *args):
        if self._is_q_function(self.eligibility):
            current_state_index = args[0]
            current_action_index = args[1]
            target = args[2]
        
            delta = target - self.value_function[current_state_index][current_action_index]
            self.eligibility[current_state_index][current_action_index] += 1.0
                
            # backforward view proprogate 
            for state_index in self.value_function:
                for action_index in self.value_function[state_index]:
                    self.value_function[state_index][action_index] = self.value_function[state_index][action_index]+self.step_size*delta* self.eligibility[state_index][action_index]
                    self.eligibility[state_index][action_index] = self.eligibility[state_index][action_index]*self.discount* self.lamb
        else:
            current_state_index = args[0]
            target = args[1]
        
            delta = target - self.value_function[current_state_index]
            self.eligibility[current_state_index]+= 1.0
                
            # backforward view proprogate 
            for state_index in self.value_function:
                self.value_function[state_index] = self.value_function[state_index]+self.step_size*delta* self.eligibility[state_index]
                self.eligibility[state_index]= self.eligibility[state_index]*self.discount* self.lamb

    def get_value_function(self):
        return self.value_function



class TDLambdaCritic(LambdaCritic):
    def __init__(self,value_function,step_size=0.1,discount=1.0,lamb=0):
        super().__init__(value_function,step_size=step_size,discount=discount,lamb=lamb)
    
    
    def evaluate(self,*args):
        if self._is_q_function(self.value_function):
            current_state_index = args[0]
            current_action_index = args[1]
            reward = args[2]
            next_state_index = args[3]    
            next_action_index = args[4]

            target = reward + self.discount*self.get_value_function()[next_state_index][next_action_index]
            self.update(current_state_index,current_action_index,target)   
        else:
            current_state_index = args[0]
            reward = args[1]
            next_state_index = args[2]    

            target = reward + self.discount*self.get_value_function()[next_state_index]
            self.update(current_state_index,target)   
        
class QLearningLambdaCritic(LambdaCritic):
    def __init__(self,value_function,step_size=0.1,discount=1.0,lamb=0):
        super().__init__(value_function,step_size=step_size,discount=discount,lamb=lamb)
    
    
    def evaluate(self,*args):
        current_state_index = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]    
        

        q_values_next_state = self.get_value_function()[next_state_index]
        best_action_next_state = max(q_values_next_state, key=q_values_next_state.get)
    
        # Q*(s',A*)
        max_value = q_values_next_state[best_action_next_state]
        target = reward + self.discount*max_value
        self.update(current_state_index,current_action_index,target)   
        





class OffPolicyGreedyActor(ExploratorBase):
    def __init__(self,behavior_policy,target_policy,critic):
        self.behavior_policy = behavior_policy
        self.target_policy =   target_policy
        self.critic = critic
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, *args):
        current_state_index = args[0]
        q_value_function = self.critic.get_value_function()
        for state_index, _ in q_value_function.items():
            q_values = q_value_function[state_index]
            greedy_distibution = self.create_distribution_greedily(q_values)
            self.target_policy.policy_table[current_state_index] = greedy_distibution

    def get_behavior_policy(self):
        return self.behavior_policy

    def get_optimal_policy(self):
        self.target_policy    

class GreedyExplorator(ExploratorBase):
    def __init__(self,behavior_policy,exploitator):
        self.behavior_policy = behavior_policy
        self.exploitator = exploitator
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, *args):
        current_state_index = args[0]
        q_value_function = self.exploitator.get_value_function()
        for state_index, _ in q_value_function.items():
            q_values = q_value_function[state_index]
            greedy_distibution = self.create_distribution_greedily(q_values)
            self.behavior_policy.policy_table[current_state_index] = greedy_distibution

    def get_behavior_policy(self):
        return self.behavior_policy


class ESoftExplorator(ExploratorBase):
    def __init__(self, behavior_policy,exploitator,epsilon=0.1):
        self.behavior_policy = behavior_policy
        self.exploitator = exploitator
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)


    def improve(self, *args):
        current_state_index = args[0]
        q_value_function = self.exploitator.get_value_function()
        q_values = q_value_function[current_state_index]
        soft_greedy_distibution = self.create_distribution_epsilon_greedily(q_values)
        self.behavior_policy.policy_table[current_state_index] = soft_greedy_distibution

    def get_behavior_policy(self):
        return self.behavior_policy


class BoltzmannExplorator(ExploratorBase):
    def __init__(self, behavior_policy,exploitator,epsilon=0.1):
        self.behavior_policy = behavior_policy
        self.exploitator = exploitator
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)
    

    def improve(self, *args):
        current_state_index = args[0]
        q_value_function = self.exploitator.get_value_function()
        q_values = q_value_function[current_state_index]
        soft_greedy_distibution = self.create_distribution_epsilon_greedily(q_values)
        self.behavior_policy.policy_table[current_state_index] = soft_greedy_distibution

    def get_behavior_policy(self):
        return self.behavior_policy




