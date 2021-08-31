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



import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)


import torch

from lib.utility import config

from env.dino.game_wrapper import GameWrapper
from algorithm.deep_reinforcement_learning.dqn.deep_q_learning import DeepQLearningAgent


# setup the GPU/CPU device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# prepare the Log for recording the RL procedure

app_cfg = config('algorithm/deep_reinforcement_learning/dqn/config.ini')
env_cfg = config("env/dino/config.ini")

def test_dqn(env,config):
        dqn= DeepQLearningAgent(env,config,device)
        dqn.learn()

dino_game = None
try:
    dino_game = GameWrapper(env_cfg)
    test_dqn(dino_game,app_cfg)

finally:
    if dino_game is not None:
        dino_game.end()