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


from algorithm.value_based.tabular_solution_method.td_method.td_lambda_critic import TDLambdaCritic


class TDLambdaSARSAExploitator(TDLambdaCritic):
    def __init__(self,value_function,step_size=0.1,discount=1.0,lamb=0):
        super().__init__(value_function,step_size=step_size,discount=discount,lamb=lamb)
    
    def evaluate(self,*args):
        if self._is_q_function(self.value_function):
            current_state_index = args[0]
            current_action_index = args[1]
            reward = args[2]
            next_state_index = args[3]    
            next_action_index = args[4]

            target = reward + self.discount*self.get_value_function()[next_state_index][next_action_index]
            self.update(current_state_index,current_action_index,target)   
        else:
            current_state_index = args[0]
            reward = args[1]
            next_state_index = args[2]    

            target = reward + self.discount*self.get_value_function()[next_state_index]
            self.update(current_state_index,target) 