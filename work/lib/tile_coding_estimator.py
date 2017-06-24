'''
 This estimator is a function approximator of linear method. We take the tile coding as the
 base function.

 One important thing is that tiling is only a map from (state, action) to a series of indices
 It doesn't matter whether the indices have meaning, only if this map satisfy some property
 View the following webpage for more information
 http://incompleteideas.net/sutton/tiles/tiles3.html
'''

import sys
import numpy as np

sys.path.append('/home/ornot/GymRL')
from lib import tile_coding


class Estimator(object):
    # @maxSize: the maximum # of indices
    def __init__(self, env, stepSize=0.3, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings

        # divide step size equally to each tiling
        self.stepSize = stepSize / numOfTilings

        self.hashTable = tile_coding.IHT(maxSize)

        self.env = env

        # weight for each tile
        self.weights = np.zeros(maxSize)

        self.observation_scales = []

        for observationComponent in range(len(env.observation_space.high)):
            self.observation_scales.append(self.numOfTilings / (
                env.observation_space.high[observationComponent] - (env.observation_space.low[observationComponent])))

    # get indices of active tiles for given state and action
    def _get_active_tiles(self, observation, action):
        activeTiles = tile_coding.tiles(self.hashTable, self.numOfTilings, [(self.observation_scales[i] * (
            observation[i] - self.env.observation_space.low[i])) for i in range(len(self.env.observation_space.high))], [action])
        return activeTiles

    # estimate the value of given state and action
    def predict(self, observation, action):
        activeTiles = self._get_active_tiles(observation, action)
        return np.sum(self.weights[activeTiles])

    # learn with given state, action and target
    def update(self, observation, action, target):
        activeTiles = self._get_active_tiles(observation, action)
        estimation = np.sum(self.weights[activeTiles])
        delta = self.stepSize * (target - estimation)
        for activeTile in activeTiles:
            self.weights[activeTile] += delta
