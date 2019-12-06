from abc import ABC, abstractclassmethod

class Base_Agent(ABC):
    
    @abstractclassmethod
    def make_decision(self,policy,observation):
        """
        Given a policy and the observation , the agent 
        decides what action to take in the next step.
        
        Arguments:
            policy {Policy} -- Tabular or funciton approximator
            observation {Observation} -- fully or partially observed state
        
        Return: 
            an action 
        
        """ 
        pass

    @abstractclassmethod
    def evaluate_policy_once(self):
        pass

    @abstractclassmethod
    def improve_policy_once(self):
        pass