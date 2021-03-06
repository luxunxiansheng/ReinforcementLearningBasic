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


from common import ActorBase
from lib.utility import (create_distribution_epsilon_greedily,create_distribution_greedily,create_distribution_boltzmann)


class ESoftActor(ActorBase):
    def __init__(self, policy,critic,epsilon=0.1):
        self.policy = policy
        self.critic = critic
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, *args):
        current_state_index = args[0]
        action_space= args[1]
        estimator = self.critic.get_value_function()
        
        q_values = {}
        for action_index in range(action_space.n):
            q_values[action_index] = estimator.predict(current_state_index,action_index)

        soft_greedy_distibution = self.create_distribution_epsilon_greedily(q_values)
        self.policy.instant_distribution = soft_greedy_distibution

    def get_behavior_policy(self):
        return self.policy

    def get_create_behavior_policy_fn(self):
        return self.create_distribution_epsilon_greedily
