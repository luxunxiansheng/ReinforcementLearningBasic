from abc import abstractmethod
from gym.envs.toy_text import discrete


class Base_Discrete_Env(discrete.DiscreteEnv):
    @abstractmethod
    def  _build_transitions(self,nS,nA):
        pass
    
    def build_Q_table(self):
        Q_table = {}
        for state_index in range(self.nS):
            Q_table[state_index] = {action_index: 0.0 for action_index in range(self.nA)}
        return Q_table


        


 