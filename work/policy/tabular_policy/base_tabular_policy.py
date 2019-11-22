from collections import defaultdict
from abc import abstractclassmethod

import numpy as np

from policy.base_policy import Base_Policy


class Base_Tabular_Policy(Base_Policy):
    def __init__(self, env):
        
        #A dictionary that maps from state to  action values
        self.Q_table = defaultdict(lambda: np.zeros(env.action_space.n))


    def select_action(self, observation):
        """
        select the action randomly 

        Arguments:
            observation:  marcov state

        Returns:
            int -- the index of the action selected  

        """

        action_probs = self.get_probability_at_state(observation)
        action_index = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action_index    
    
    @abstractclassmethod
    def get_probability_at_state(self,observation):
        pass