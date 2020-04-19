"""
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.

"""

import io
import sys

import numpy as np

from env.base_discrete_env import BaseDiscreteEnv


class GridworldEnv(BaseDiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[6, 6]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        nS = np.prod(shape)
        self.grid = np.arange(nS).reshape(shape)

        nA = 4
        self.P = self._build_transitions(nS, nA)

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS
        super().__init__(nS, nA, self.P, isd)

    def _build_transitions(self, nS, nA):
         # Transition prob matrix
        P = {}

        it = np.nditer(self.grid, flags=['multi_index'])

        MAX_Y = self.grid.shape[0]
        MAX_X = self.grid.shape[1]

        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            def is_done(s): return s == 0 or s == (nS - 1)

            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        return P
    
    

    def render(self, mode='human'):
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        it = np.nditer(self.grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            _, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
