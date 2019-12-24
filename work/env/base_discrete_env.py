from abc import abstractmethod
from gym.envs.toy_text import discrete


class Base_Discrete_Env(discrete.DiscreteEnv):
    
    
    @abstractmethod
    def build_Q_table(self):
        Q_table = {}
        for state_index in range(self.nS):
            Q_table[state_index] = {action_index: 0.0 for action_index in range(self.nA)}
        return Q_table

    @abstractmethod
    def build_V_table(self):
        V_table = {}
        for state_index in range(self.nS):
            V_table[state_index] = 0.0
        return V_table

    @abstractmethod
    def build_policy_table(self):
        policy_table = {}
        for state_index in range(self.nS):
            policy_table[state_index] = {action_index: 1.0/self.nA for action_index in range(self.nA)}
        return policy_table
        


 