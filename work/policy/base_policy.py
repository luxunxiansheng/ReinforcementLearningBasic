from abc import ABC, abstractmethod

import numpy as np


class Base_Policy(ABC):
    """
    Essentially, a policy is a probability distribution of action at specific state. 
    The distribution can be built either with a Q table or function approximator    
    """
    @abstractmethod
    def select_action(self, observation):
        pass 
        

    @abstractmethod
    def get_probability_at_state(self, observation):
        pass


