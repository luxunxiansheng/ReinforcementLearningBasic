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

from abc import abstractmethod
import numpy as np
from math import floor


class QFunction:
    @abstractmethod
    def value(self, state, action):
        pass

    @abstractmethod
    def update(self, alpha, delta, state,action):
        pass



class TileCodingBasesValueFunction(QFunction):
    
    ################# Tile coding ##########################################
    # Following are some utilities for tile coding from Rich.
    # To make each file self-contained, I copied them from
    # http://incompleteideas.net/tiles/tiles3.py-remove
    # with some naming convention changes
    # 
    
    class IHT:
        '''
        Indexed Hash Table   
        '''

        "Structure to handle collisions"
        def __init__(self, size_val):
            self.size = size_val
            self.overfull_count = 0
            self.dictionary = {}

        def count(self):
            return len(self.dictionary)

        def full(self):
            return len(self.dictionary) >= self.size

        def get_index(self, obj, read_only=False):
            d = self.dictionary
            if obj in d:
                return d[obj]
            elif read_only:
                return None
            size = self.size
            count = self.count()
            if count >= size:
                if self.overfull_count == 0: print('IHT full, starting to allow collisions')
                self.overfull_count += 1
                return hash(obj) % self.size
            else:
                d[obj] = count
                return count

    @staticmethod
    def hash_coords(coordinates, m, read_only=False):
        if isinstance(m, TileCodingBasesValueFunction.IHT): return m.get_index(tuple(coordinates), read_only)
        if isinstance(m, int): return hash(tuple(coordinates)) % m
        if m is None: return coordinates


    # This implementation is specific to continuous state space and discrete action space
    @staticmethod
    def build_tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
        """returns num-tilings tile indices corresponding to the floats and ints"""
        if ints is None:
            ints = []
        qfloats = [floor(f * num_tilings) for f in floats]
        tiles = []
        for tiling in range(num_tilings):
            tilingX2 = tiling * 2
            coords = [tiling]
            b = tiling
            for q in qfloats:
                coords.append((q + b) // num_tilings)
                b += tilingX2
            coords.extend(ints)
            tiles.append(TileCodingBasesValueFunction.hash_coords(coords, iht_or_size, read_only))
        return tiles


        
    # Tile coding ends
    #######################################################################

    def __init__(self, step_size, x_max, x_min, y_max, y_min, num_of_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings

        # divide step size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.hash_table = TileCodingBasesValueFunction.IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)
        
        self.x_max= x_max
        self.x_min =x_min 
        self.y_max = y_max
        self.y_min = y_min
    
        self.scale_x = self.num_of_tilings / (x_max - x_min)
        self.scale_y = self.num_of_tilings / (y_max - y_min)

    # get indices of active tiles for given state and action
    def get_active_tiles(self, floats, ints):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = TileCodingBasesValueFunction.tiles(self.hash_table, self.num_of_tilings,
                            [self.scale_x * floats[0], self.scale_y * floats[1]],
                            [ints])
        return active_tiles

    # estimate the value of given state and action
    def value(self, state, action):
        if floats[0] == self.x_max:
            return 0.0
        active_tiles = self.get_active_tiles(floats, ints)
        return np.sum(self.weights[active_tiles])

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.step_size * (target - estimation)
        for active_tile in active_tiles:
            self.weights[active_tile] += delta

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)


    def value(self, state, action):
        pass

    
    def update(self, alpha, delta, state, action):
        pass