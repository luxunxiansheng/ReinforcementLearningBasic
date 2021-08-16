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

import numpy as np 
from policy.policy import Policy

class ContinuousStateValueTablePolicy(Policy):
    def __init__(self,value_esitmator,create_distribution_fn=None):
        self.value_esitmator = value_esitmator
        self.create_distribution_fn = create_distribution_fn
    
    def get_action(self, state):
        distribution = self.get_discrete_distribution(state)
        action = np.random.choice(np.arange(len(distribution)), p=list(distribution.values()))
        return action
    
    def get_discrete_distribution(self,state):
        q_values = self.value_esitmator.predict(state)
        q_values = {index:q_values[index].item() for index in range(0,q_values.size())}
        distribution = self.create_distribution_fn(q_values)
        return distribution 

