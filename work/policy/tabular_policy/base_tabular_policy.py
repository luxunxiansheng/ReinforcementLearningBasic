from collections import defaultdict

import numpy as np

from policy.base_policy import Base_Policy


class Base_Tabular_Policy(Base_Policy):
    def __init__(self, env):
        
        #A dictionary that maps from state to  action values
        self._Q_table = defaultdict(lambda: np.zeros(env.action_space.n))
