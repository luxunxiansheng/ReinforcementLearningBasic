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
from torchvision import transforms

from env.dino.game import Game


class GameWrapper:
    def __init__(self,config):
        self.game = Game(config)
        self.img_rows = config['GAME'].getint("img_rows")
        self.img_columns = config['GAME'].getint("img_columns")
    
    def _preprocess_snapshot(self, screenshot):
        transform = transforms.Compose([transforms.CenterCrop((150, 600)),
                                        transforms.Resize((self.img_rows, self.img_columns)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        return transform(screenshot)

    def reset(self):
        init_screentshot= self._preprocess_snapshot(self.game.reset())
        return torch.stack((init_screentshot,init_screentshot,init_screentshot,init_screentshot))
    
    def step(self,current_state,action):
        screen_shot, reward, terminal, score = self.game.step(action)
        preprocessed_snapshot = self._preprocess_snapshot(screen_shot)
        next_state = current_state.clone()
        next_state[0:-1] = current_state[1:]
        next_state[-1] = preprocessed_snapshot
        return next_state, torch.tensor(reward), torch.tensor(terminal), score

    def end(self):
        self.game.end()
        
    