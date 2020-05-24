# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /

from env.base_discrete_env import BaseDiscreteEnv
import numpy as np

RIGHT = 1
LEFT = 0


class RandomWalkingEnv(BaseDiscreteEnv):
    """
    Example 6.2

    For convenience, name the state from left to right with number 1 to nS

    """
    def __init__(self,num_states=19):
        nS = num_states
        nA = 2
        self.P = self._build_transitions(nS, nA)

        isd = np.ones(nS) / nS
        super().__init__(nS, nA, self.P, isd)

    def reset(self,randomly=False):
        if randomly:
            super().reset(True)
        else:
            # always start from the middle location
            self.s = int((self.nS+1)/2)-1
            return self.s

    def _build_transitions(self, nS, nA):
        P = {}
        for state_index in range(nS):
            P[state_index] = {action_index: [] for action_index in range(nA)}

        P[0][LEFT] =  [(1.0, 0, 0, True)]
        P[0][RIGHT] = [(1.0, 0, 0, True)]

        P[1][LEFT] =  [(1.0, 0, 0, True)]
        P[1][RIGHT] = [(1.0, 2, 0, False)]


        for state_index in range(2, nS-2):
            P[state_index][LEFT] =  [(1.0, state_index-1, 0, False)]
            P[state_index][RIGHT] = [(1.0, state_index+1, 0, False)]
        
        
        P[nS-2][LEFT]  = [(1.0, nS-3, 0, False)]
        P[nS-2][RIGHT] = [(1.0, nS-1, 1, True)]

        P[nS-1][LEFT]  = [(1.0, nS-1, 0, True)]
        P[nS-1][RIGHT] = [(1.0, nS-1, 0, True)]

        return P
