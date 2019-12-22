import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from env.base_discrete_env import Base_Discrete_Env


class GamblersProblemEnv(Base_Discrete_Env):

    def __init__(self, goal=100, head_prob=0.4):
        self.goal = goal
        self.head_prob = head_prob

        nS = self.goal+1
        nA = int(self.goal/2)+1
        self.P = self._build_transitions(nS)

        isd = np.ones(nS) / nS
        super().__init__(nS, nA, self.P, isd)

    def _is_done(self, capital):
        if capital == self.goal:
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
                state_index), self.goal-self._get_capital(state_index))+1)
            P[state_index] = {action_index: []
                              for action_index in range(len(actions))}
            for action_index in range(len(actions)):
                P[state_index][action_index] = [(self.head_prob, 
                                                 self._get_state_index(self._get_capital(state_index)+self._get_action(action_index)), 
                                                 self._is_done(self._get_capital(state_index)+self._get_action(action_index))[0],
                                                 self._is_done(self._get_capital(state_index)+self._get_action(action_index))[1]),
                                                (1-self.head_prob, 
                                                self._get_state_index(self._get_capital(state_index)-self._get_action(action_index)),
                                                self._is_done(self._get_capital(state_index)-self._get_action(action_index))[0],
                                                self._is_done(self._get_capital(state_index)-self._get_action(action_index))[1])]

        return P

    def build_Q_table(self):
        Q_table = {}
        for state_index in range(self.nS):
            actions = np.arange(min(self._get_capital(
                state_index), self.goal-self._get_capital(state_index))+1)
            Q_table[state_index] = {
                action_index: 0.0 for action_index in range(len(actions))}
        Q_table[self.nS-1] = {0: 1.0}
        return Q_table

    def build_V_table(self):
        V_table = {}
        for state_index in range(self.nS):
            V_table[state_index] = 0.0

        V_table[self.nS-1] = 1.0
        return V_table

    def build_policy_table(self):
        pi_table = {}
        for state_index in range(self.nS):
            actions = np.arange(min(self._get_capital(
                state_index), self.goal-self._get_capital(state_index))+1)
            pi_table[state_index] = {
                action_index: 0.0 for action_index in range(len(actions))}
        return pi_table
