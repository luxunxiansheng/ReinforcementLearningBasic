import sys
from collections import defaultdict

import gym
import numpy as np
import pysnooper
from gym import spaces
from gym.envs.toy_text import discrete

from algorithm.implicity_policy import dynamic_programming


class RoomEnv(discrete.DiscreteEnv):
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