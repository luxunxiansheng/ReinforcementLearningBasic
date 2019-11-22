import numpy as np
from base_tabular_policy import Base_Tabular_Policy


class e_Greedy_Policy(Base_Tabular_Policy):
    def __init__(self, epsilon):

        self._epsilon = epsilon

    def __call__(self, observation):

        action_values = self._Q_table[observation]
        num_actions = len(action_values)

        action_probs = np.ones(num_actions, dtype=float) * self._epsilon / num_actions
        best_action = np.argmax(action_values)
        action_probs[best_action] += (1.0 - self._epsilon)
        action_index = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action_index
 
