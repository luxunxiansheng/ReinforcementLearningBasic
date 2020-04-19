import sys

import numpy as np

from env.base_discrete_env import BaseDiscreteEnv

"""
   A simple grid world which has walls bolck. Refer to https://courses.cs.washington.edu/courses/cse473/13au/slides/15-mdp.pdf
   for more details.

"""

UP = 0
LEFT = 1
RIGHT = 2
DOWN = 3

class GridWorldWithWallsBlockEnv(BaseDiscreteEnv):
    def __init__(self,reward_non_terminals=-0.01):
        self.reward_non_terminals = reward_non_terminals
        self.shape = [3, 4]
        nS = np.prod(self.shape)
        self.grid = np.arange(nS).reshape(self.shape)

        nA = 3  # up, left and right
        self.P = self._build_transitions()
        isd = np.ones(nS)/nS
        super().__init__(nS, nA, self.P, isd)

    def _build_transitions(self):
        P = {
            0: {
                UP:   [(1.0, 0, self.reward_non_terminals, False)],
                LEFT: [(1.0, 0, self.reward_non_terminals, False)],
                RIGHT: [(1.0, 1,self.reward_non_terminals, False)]
            },
            1: {
                UP:   [(1.0, 1, self.reward_non_terminals, False)],
                LEFT: [(1.0, 0, self.reward_non_terminals, False)],
                RIGHT: [(1.0, 2,self.reward_non_terminals, False)]
            },
            2: {
                UP:   [(1.0, 2, self.reward_non_terminals, False)],
                LEFT: [(1.0, 1, self.reward_non_terminals, False)],
                RIGHT: [(1.0, 3, 1.00, True)]
            },
            3: {
                UP:   [(1.0, 3, 1.00, True)],
                LEFT: [(1.0, 3, 1.00, True)],
                RIGHT: [(1.0,3, 1.00, True)]
            },
            4: {
                UP:   [(1.0, 0, self.reward_non_terminals, False)],
                LEFT: [(1.0, 4, self.reward_non_terminals, False)],
                RIGHT: [(1.0,4, self.reward_non_terminals, False)]
            },
            6: {
                UP:   [(1.0, 2, self.reward_non_terminals, False)],
                LEFT: [(1.0, 6, self.reward_non_terminals, False)],
                RIGHT: [(1.0, 7, -1.00, True)]
            },
            7: {
                UP:   [(1.0, 7, -1.00, True)],
                LEFT: [(1.0, 7, -1.00, True)],
                RIGHT: [(1.0,7, -1.00, True)]
            },
            8: {
                UP:   [(1.0, 4, self.reward_non_terminals, False)],
                LEFT: [(1.0, 8, self.reward_non_terminals, False)],
                RIGHT: [(1.0, 9,self.reward_non_terminals, False)]
            },
            9: {
                UP:   [(1.0, 9, self.reward_non_terminals, False)],
                LEFT: [(1.0, 8, self.reward_non_terminals, False)],
                RIGHT: [(1.0, 10,self.reward_non_terminals, False)]
            },
            10: {
                UP:   [(1.0, 6, self.reward_non_terminals, False)],
                LEFT: [(1.0, 9, self.reward_non_terminals, False)],
                RIGHT: [(1.0, 11,self.reward_non_terminals, False)]
            },
            11: {
                UP:   [(1.0, 7, -1, True)],
                LEFT: [(1.0, 10,self.reward_non_terminals, False)],
                RIGHT: [(1.0, 11,self.reward_non_terminals, False)]
            }
            }

        return P

    def build_V_table(self):
        V_table = {
            0:0.0,
            1:0.0,
            2:0.0,
            3:0.0,
            4:0.0,
            6:0.0,
            7:0.0,   
            8:0.0,
            9:0.0,
            10:0.0,
            11:0.0
        }
        return V_table

    def build_Q_table(self):
        Q_table = {
            0: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },
            1: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },
            2: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },
            3: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },
            4: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },
            6: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },
            7: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },   
            8: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },
            9: {UP:    0.0,
                LEFT:  0.0,
                RIGHT: 0.0
                },
            10: {UP:    0.0,
                 LEFT:  0.0,
                 RIGHT: 0.0
                 },
            11: {UP:    0.0,
                 LEFT:  0.0,
                 RIGHT: 0.0
                 }
        }
        return Q_table

    def build_policy_table(self):
        default_prob= 1.0/3
        pi_table = {
            0: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },
            1: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },
            2: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },
            3: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },
            4: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },
            6: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },
            7: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },   
            8: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },
            9: {UP:    default_prob,
                LEFT:  default_prob,
                RIGHT: default_prob
                },
            10: {UP:    default_prob,
                 LEFT:  default_prob,
                 RIGHT: default_prob
                 },
            11: {UP:    default_prob,
                 LEFT:  default_prob,
                 RIGHT: default_prob
                 }
        }
        return pi_table

