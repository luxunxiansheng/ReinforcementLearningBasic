from base_tabular_policy import Base_Tabular_Policy
import numpy as np


class Random_Policy(Base_Tabular_Policy):
    def get_probability_at_state(self, observation):
        action_values = self._Q_table[observation]
        num_actions = len(action_values)
        action_probs = np.ones(num_actions, dtype=float) / num_actions
        return action_probs
