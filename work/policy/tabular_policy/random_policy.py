from collections import defaultdict

import numpy as np

from base_tabular_policy import Base_Tabular_Policy


class Random_Policy(Base_Tabular_Policy):
    def get_probability_at_state(self, state):
        action_values = self.Q_table[state]
        num_actions = len(action_values)
        action_probs = np.ones(num_actions, dtype=float) / num_actions
        return action_probs

    def evaluate(self,state,transitons):
        