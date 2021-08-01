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

from abc import  ABCMeta, abstractmethod

class CriticBase(metaclass=ABCMeta):
    """
    The critic calculates the value of state or value of (state,action) pair by following target policy. 
    This is essentially calculating the Bellman optimality equation approximately.
    """
    @abstractmethod
    def evaluate(self,*args):
        pass
    
    @abstractmethod
    def get_value_function(self):
        pass 

    @abstractmethod
    def get_optimal_policy(self):
        pass

class ExplorerBase(metaclass=ABCMeta):
    """
    If the dynamics of the enviroment is hard to know, the explorer will define a behavior policy to sample data from the enviroment. 
    With constrained computing resources, the explorer 's purpose is to learn a policy which can identify those states of high value 
    rather than accuratly calculate value of those states. 
    """
    @abstractmethod
    def explore(self,*args): 
        pass
    
    @abstractmethod
    def get_behavior_policy(self):
        pass

class ActorBase(metaclass=ABCMeta):
    @abstractmethod
    def act(self,*args):
        pass 


class QValueEstimator(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, state, action):
        pass

    @abstractmethod
    def update(self, *args):
        pass

class PolicyEstimator(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def update(self, *args):
        pass

class ValueEstimator(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def update(self, *args):  
        pass

class Agent(metaclass=ABCMeta):
    @abstractmethod
    def learn(self):
        pass 

