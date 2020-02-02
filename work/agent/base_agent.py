from abc import ABC, abstractclassmethod

class Base_Agent(ABC):
    
    @abstractclassmethod
    def select_action(self,observation):
        """
        Given the  observation , the agent 
        decides what action to take in the next step.
        
        Arguments:
            observation {Observation} -- fully or partially observed state
        
        Return: 
            an action 
        
        """ 
        pass
    
    @abstractclassmethod
    def evaluate(self):
        """
        Given a policy, to predict the value function.
        """
        pass  
    

    @abstractclassmethod
    def improve(self):
        """
        Give the value function, how to get an optimal policy 
        """
        pass 
    