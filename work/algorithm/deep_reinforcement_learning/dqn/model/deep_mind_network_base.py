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

import torch
import torch.nn as nn

from lib.utility import Utilis
from model.noisy_linear import NoisyLinear

class DeepMindNetworkBase(nn.Module):
    @staticmethod
    def create(network, input_channels, output_size):
        if network == 'DeepMindNetwork':
            from deep_mind_network import DeepMindNetwork
            return DeepMindNetwork(input_channels, output_size)
        if network == 'CategoricalNetwork':
            from categorical_network import CategoricalNetwork
            return CategoricalNetwork(input_channels, output_size)
        if network == 'NoisyNetwork':
            from noisy_network import NoisyNetwork
            return NoisyNetwork(input_channels, output_size)
        if network == 'DuelingNetwork':
            from dueling_network import DuelingNetwork
            return DuelingNetwork(input_channels,output_size)    

        return None    
    '''
    The convolution newtork proposed by Mnih at al(2015) in the paper "Playing Atari with Deep
    Reinforcement Learning. We split the last layer of orignal network in the paper as the header
    and keep the remains into the base part.  Here is the base part."
    '''

    def __init__(self, input_channels,noisy=False):
        super().__init__()

        self.input_channels = input_channels
    
        self._conv1 = nn.Sequential(
            Utilis.layer_init(nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU()
        )

        self._conv2 = nn.Sequential(
            Utilis.layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU()
        )

        self._conv3 = nn.Sequential(
            Utilis.layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU()
        )

        

        if noisy:
            self._fc1 = nn.Sequential(NoisyLinear(7*7*64, 512),
            nn.ReLU()
            )  
        else:
            self._fc1 = nn.Sequential(
                Utilis.layer_init(nn.Linear(7*7*64, 512)),
                nn.ReLU()
            )
        
    def forward(self, input):
        x = torch.transpose(input, 0, 1)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)

        x = x.view(1, -1)

        x = self._fc1(x)

        return x
