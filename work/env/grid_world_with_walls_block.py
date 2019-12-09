import sys

import numpy as np

from env.base_discrete_env import Base_Discrete_Env

"""
   A simple grid world which has walls bolck. Refer to https://courses.cs.washington.edu/courses/cse473/13au/slides/15-mdp.pdf
   for more details.

"""

UP = 0
LEFT = 1
RIGHT = 2
DOWN = 3

REWARD_NON_TERMINALS = 0


class GridWorldWithWallsBlockEnv(Base_Discrete_Env):
    def __init__(self):
        self.shape = [3, 4]
        nS = np.prod(self.shape)
        self.grid = np.arange(nS).reshape(self.shape)

        nA = 3  # up, left and right
        self.P = self._build_transitions(nS, nA)
        isd = np.ones(nS)/nS
        super().__init__(nS, nA, self.P, isd)

    def _build_transitions(self, nS, nA):
        P = {
            0: {
                UP:   [(1.0, 0, REWARD_NON_TERMINALS, False)],
                LEFT: [(1.0, 0, REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0, 1,REWARD_NON_TERMINALS, False)]
            },
            1: {
                UP:   [(1.0, 1, REWARD_NON_TERMINALS, False)],
                LEFT: [(1.0, 0, REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0, 2,REWARD_NON_TERMINALS, False)]
            },
            2: {
                UP:   [(1.0, 2, REWARD_NON_TERMINALS, False)],
                LEFT: [(1.0, 1, REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0, 3, 1.00, True)]
            },
            3: {
                UP:   [(1.0, 3, 1.00, True)],
                LEFT: [(1.0, 3, 1.00, True)],
                RIGHT: [(1.0,3, 1.00, True)]
            },
            4: {
                UP:   [(1.0, 0, REWARD_NON_TERMINALS, False)],
                LEFT: [(1.0, 4, REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0,4, REWARD_NON_TERMINALS, False)]
            },
            6: {
                UP:   [(1.0, 2, REWARD_NON_TERMINALS, False)],
                LEFT: [(1.0, 6, REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0, 7, -1.00, True)]
            },
            7: {
                UP:   [(1.0, 7, -1.00, True)],
                LEFT: [(1.0, 7, -1.00, True)],
                RIGHT: [(1.0,7, -1.00, True)]
            },
            8: {
                UP:   [(1.0, 4, REWARD_NON_TERMINALS, False)],
                LEFT: [(1.0, 8, REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0, 9,REWARD_NON_TERMINALS, False)]
            },
            9: {
                UP:   [(1.0, 9, REWARD_NON_TERMINALS, False)],
                LEFT: [(1.0, 8, REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0, 10,REWARD_NON_TERMINALS, False)]
            },
            10: {
                UP:   [(1.0, 6, REWARD_NON_TERMINALS, False)],
                LEFT: [(1.0, 9, REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0, 11,REWARD_NON_TERMINALS, False)]
            },
            11: {
                UP:   [(1.0, 7, -1, True)],
                LEFT: [(1.0, 10,REWARD_NON_TERMINALS, False)],
                RIGHT: [(1.0, 11,REWARD_NON_TERMINALS, False)]
            }
            }

        return P

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
