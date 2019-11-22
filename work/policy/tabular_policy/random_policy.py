from base_tabular_policy  import Base_Tabular_Policy
import numpy as np


class Random_Policy(Base_Tabular_Policy):
    
    def __call__(self,observation):
        """
        select the action randomly 
        
        Arguments:
           
        
        Returns:
            int -- the index of the action selected  
            
        """
                
        action_values = self._Q_table[observation]
        num_actions = len(action_values)
        action_probs = np.ones(num_actions, dtype=float) / num_actions
        action_index = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action_index




   