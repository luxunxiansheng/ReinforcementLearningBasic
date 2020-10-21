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

from abc import ABC, abstractmethod
from lib.utility import create_distribution_epsilon_greedily

import numpy as np

class Policy(ABC):
    @abstractmethod
    def get_action(self,state):
        pass


class DiscreteActionPolicy(ABC):
    """
    A policy defines the learning agent's way of behaving at a given time. Roughly speaking,
    a policy is a mapping from perceived states of the environment to actions to be taken
    when in those states. It corresponds to what in psychology would be called a set of
    stimulus-response rules or associations. In some cases the policy may be a simple function
    or lookup table, whereas in others it may involve extensive computation such as a search
    process. The policy is the core of a reinforcement learning agent in the sense that it alone
    is suficient to determine behavior. In general, policies may be stochastic.

    """
    @abstractmethod
    def _construct_discrete_distribution(self,state):
        pass   

    def get_action(self, state):
        distribution = self._construct_discrete_distribution(state)
        action = np.random.choice(np.arange(len(distribution)), p=distribution)
        return action


class DiscreteStateValueBasedPolicy(DiscreteActionPolicy):
    def __init__(self,policy_table):
        self.policy_table = policy_table

    def _construct_discrete_distribution(self, state):
        return list(self.policy_table[state].values())


class ContinuousStateValueBasedPolicy(DiscreteActionPolicy):
    def __init__(self,action_space,q_value_estimator,create_distribution_fn):
        self.create_distribution_fn = create_distribution_fn
        self.q_value_estimator = q_value_estimator
        self.action_space = action_space
        

    def _construct_discrete_distribution(self,state):
        q_values ={}
        for action_index in range(self.action_space.n):
            q_values[action_index] = self.q_value_estimator.predict(state,action_index)
        
        return  list(self.create_distribution_fn(q_values).values())
    

class ParameterizedPolicy(DiscreteActionPolicy):
    """
    In tabular policy, the actions' probability distribution is built from the Q values. In ParameterizedPolicy, the distribution 
    is a direct output of a parameterized function. 
    """
    
    def __init__(self,policy_estimator):
        self.policy_estimator = policy_estimator

    def _construct_distribution(self, state):
        distribution = self.policy_estimator.predict(state)
        return distribution
