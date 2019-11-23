from collections import defaultdict
import numpy as np


class State_Action_Value:
    def __init__(self,num_actions):
        self.Q_table = defaultdict(lambda: np.zeros(num_actions))


    

