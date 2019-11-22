from abc import ABC, abstractmethod

class Base_Policy(ABC):
    """
    Essentially, a policy is a probability distribution of action at specific state. 
    The distribution can be built either with a Q table or function approximator    
    """
        
    @abstractmethod
    def select_action(self, observation):
        """
        Return a action at a specific state by sampling the probability distribution 
        
        Arguments:
            observation {Observation } -- based on the state
        """
        
        pass
 
     


    

        

