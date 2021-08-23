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

import math

def create_distribution_greedily():
    def fn(dict_values):
        best_action_index = max(dict_values, key=dict_values.get)
        probs = {}
        for index, _ in dict_values.items():
            probs[index] = 0.0
        probs[best_action_index] = 1.0
        return probs
    return fn


def create_distribution_randomly():
    def fn(dict_values):
        probs = {}
        num_values = len(dict_values)
        for index, _ in dict_values.items():
            probs[index] = 1.0/num_values
        return probs
    return fn

def create_distribution_boltzmann():
    def fn(dict_values):
        probs = {}
        z = sum([math.exp(x) for x in dict_values.values()])
        for index, _ in dict_values.items():
            probs[index] = math.exp(dict_values[index])/z 
        return probs
    return fn

def create_distribution_epsilon_greedily(epsilon):
    def fn(dict_values):
        probs = {}
        num_values = len(dict_values)
        for index, _ in dict_values.items():
            probs[index] = epsilon/num_values
        
        best_action_index = max(dict_values, key=dict_values.get)
        probs[best_action_index] += (1.0 - epsilon)
        return probs
    return fn
