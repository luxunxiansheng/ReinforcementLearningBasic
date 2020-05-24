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

#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

class ValueFunction:
    @abstractmethod
    def value(self, state):
        pass

    @abstractmethod
    def update(self, alpha, delta, state):
        pass


class LinearApproximationMethod(ValueFunction): 
    def __init__(self, order):
        self.order = order
        self.weights = np.zeros(order + 1)

        # set up bases function
        self.bases = []
        
    # get the value of @state
    def value(self, state):
        # get the feature vector
        feature = np.asarray([func(state) for func in self.bases])
        return np.dot(self.weights, feature)

    def update(self, alpha, delta, state):
        # get derivative value
        derivative_value = np.asarray([func(state) for func in self.bases])
        self.weights += alpha* delta * derivative_value

class PolynomialBasesValueFunction(LinearApproximationMethod):
    def __init__(self, order):
        super().__init__(order)
        for i in range(0, order + 1):
            self.bases.append(lambda s, i=i: pow(s, i))
        
class FourierBasesValueFunction(LinearApproximationMethod):
    def __init__(self, order):
        super().__init__(order)
        for i in range(0, order + 1):
            self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))
