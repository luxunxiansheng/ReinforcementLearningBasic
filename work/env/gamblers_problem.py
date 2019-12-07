import sys

import gym
import numpy as np
import pysnooper
from gym import spaces
from gym.envs.toy_text import discrete

from algorithm.implicity_policy import dynamic_programming


class GamblersProblemEnv(discrete.DiscreteEnv):
    GOAL = 100
    HEAD_PROB = 0.4

    def __init__(self):
        nS = GamblersProblemEnv.GOAL+1
        nA = int(GamblersProblemEnv.GOAL / 2)+1
        self.P = self._build_transitions(nS)

        isd = np.ones(nS) / nS
        super().__init__(nS, nA, self.P, isd)

    def _is_done(self, capital):
        if capital == 100:
            return 1.0, True

        if capital == 0:
            return 0.0, True

        return 0.0, False

    def _get_capital(self, state_index):
        return state_index

    def _get_state_index(self, capital):
        return capital

    def _get_action(self, action_index):
        return action_index

    def _get_action_index(self, action):
        return action

    def _build_transitions(self, nS):

        P = {}
        for state_index in range(nS):
            actions = np.arange(min(self._get_capital(
                state_index), GamblersProblemEnv.GOAL-self._get_capital(state_index))+1)
            P[state_index] = {action_index: []
                              for action_index in range(len(actions))}
            for action_index in range(len(actions)):
                P[state_index][action_index] = [(GamblersProblemEnv.HEAD_PROB,   self._get_state_index(self._get_capital(state_index)+self._get_action(action_index)), self._is_done(self._get_capital(state_index)+self._get_action(action_index))[0], self._is_done(self._get_capital(state_index)+self._get_action(action_index))[1]),
                                                (1-GamblersProblemEnv.HEAD_PROB, self._get_state_index(self._get_capital(state_index)-self._get_action(action_index)), self._is_done(self._get_capital(state_index)-self._get_action(action_index))[0], self._is_done(self._get_capital(state_index)-self._get_action(action_index))[1])]

        return P

    def build_Q_table(self):

        Q_table = {}
        for state_index in range(self.nS-1):
            actions = np.arange(min(self._get_capital(
                state_index), GamblersProblemEnv.GOAL-self._get_capital(state_index))+1)
            Q_table[state_index] = {
                action_index: 0.0 for action_index in range(len(actions))}
        Q_table[100] = {0:1.0}
        return Q_table

    def show_optimal_value_state(self,policy):
        outfile = sys.stdout
        for state_index in range(self.nS):
            optimal_value_of_state = dynamic_programming.get_optimal_value_of_state(policy,state_index)
            output = "{0:.2f} ******".format(optimal_value_of_state) 
            outfile.write(output)

            if state_index % 10== 0 :
                outfile.write("\n")

            
