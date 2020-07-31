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

import sys
from collections import defaultdict

import gym
import numpy as np
from gym import spaces

from env.base_discrete_env import BaseDiscreteEnv


class RoomEnv(BaseDiscreteEnv):
    ROOMS= 6

    def __init__(self):
        nS=  RoomEnv.ROOMS
        nA = RoomEnv.ROOMS

        self.P = self._build_transitions(nS,nA)
        isd = np.ones(nS)/nS
        super().__init__(nS, nA, self.P, isd)

    def _build_transitions(self,nS,nA):
        P = {}

        for state_index in range(nS):
            P[state_index]=defaultdict(lambda:np.zeros(nA))
            for action_index in range(nA):
                P[state_index][action_index]=[(0.0,0,0,False),(0.0,1,0,False),(0.0,2,0,False),(0.0,3,0,False),(0.0,4,0,False),(0.0,5,0,False)]
           
        P[0][4][4]=(1.0,4,0,False)
        P[1][3][3]=(1.0,3,0,False)
        P[1][5][5]=(1.0,5,100,True)
        P[2][3][3]=(1.0,3,0,False)
        P[3][1][1]=(1.0,1,0,False)
        P[3][2][2]=(1.0,2,0,False)
        P[3][4][4]=(1.0,4,0,False)
        P[4][0][0]=(1.0,0,0,False)
        P[4][3][3]=(1.0,3,0,False)
        P[4][5][5]=(1.0,5,100,True)
        P[5][5][5]=(1.0,5,100,True)
        P[5][1][1]= (1.0,5,0,False)
        P[5][4][4]= (1.0,4,0,False)
        
        return P
    
    def build_Q_table(self):
        Q_table = {}
        for state_index in range(self.nS):
            Q_table[state_index] = {action_index: 0.0 for action_index in range(self.nA)}
        return Q_table

    
    def show_optimal_value_of_action(self,policy):
        outfile = sys.stdout
        for state_index in range(self.nS):
            for action_index in range(self.nA):
                optimal_value_of_action = dynamic_programming.get_value_of_action(policy,self,state_index,action_index)
                output = "{0:.2f} ******".format(optimal_value_of_action) 
                outfile.write(output)
            outfile.write("\n")

        outfile.write('----------------------------------------------\n')
