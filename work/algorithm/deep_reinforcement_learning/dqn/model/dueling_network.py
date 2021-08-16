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
 #/  

import torch
import torch.nn as nn
from model.deep_mind_network_base import DeepMindNetworkBase
from utils.utilis import Utilis


class DuelingNetwork(DeepMindNetworkBase):
    def __init__(self, input_channels, output_size):
        super(DuelingNetwork, self).__init__(input_channels)
        self._base = super(DuelingNetwork,self)
        self._output_size = output_size
        self._state_value_header = nn.Sequential(Utilis.layer_init(nn.Linear(512, 1)))
        self._advantage_header=    nn.Sequential(Utilis.layer_init(nn.Linear(512,self._output_size)))
    

    def forward(self, input):
        x = self._base.forward(input)
        state_value = self._state_value_header(x)
        advantage = self._advantage_header(x)
        q_value= state_value+advantage-advantage.mean()

        return  torch.squeeze(q_value)