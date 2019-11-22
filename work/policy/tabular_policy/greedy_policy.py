import numpy as np

from base_tabular_policy import Base_Tabular_Policy


class Greedy_Policy(Base_Tabular_Policy):

    def get_probability_at_state(self, observation):
        """
        select the action of the maximum value 

        """

        action_values = self._Q_table[observation]
        num_actions = len(action_values)

        action_probs = np.zeros(num_actions, dtype=float)

        best_action_index = np.argmax(action_values)
        action_probs[best_action_index] = 1.0

        return action_probs
