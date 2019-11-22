import numpy as np

from base_tabular_policy import Base_Tabular_Policy


class Greedy_Policy(Base_Tabular_Policy):

    def __call__(self, observation):
        """
        select the action of the maximum value 

        """

        action_values = self._Q_table[observation]
        best_action_index = np.argmax(action_values)
        return best_action_index
