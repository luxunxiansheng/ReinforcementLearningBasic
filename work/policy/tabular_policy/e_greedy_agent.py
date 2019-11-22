import numpy as np

from base_tabular_policy import Base_Tabular_Policy


class e_Greedy_Policy(Base_Tabular_Policy):
    def __init__(self, epsilon):

        self._epsilon = epsilon

    def get_probability_at_state(self, observation):

        action_values = self.Q_table[observation]
        num_actions = len(action_values)

        action_probs = np.ones(num_actions, dtype=float) * self._epsilon / num_actions
        
        best_action = np.argmax(action_values)
        action_probs[best_action] += (1.0 - self._epsilon)
        
        return action_probs
