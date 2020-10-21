from common import ActorBase
import itertools
import sys
from collections import namedtuple

import numpy as np
import numpy

sys.path.append('/home/ornot/GymRL')
from lib import utility

from tqdm import tqdm

class Actor(ActorBase):
    def __init__(self,policy_esitmator,num_episodes, discount_factor=1.0):
        self.policy_esitmator = policy_esitmator
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor

    
    def improve(self,*args):
        for _ in tqdm(range(0,self.num_episodes)):
            trajectory = self._run_one_episode()
            G = 0.0
            for state_index, _, reward in trajectory[::-1]:
                # The return for current state_action pair
                G = reward + self.discount*G
                self.estimator.update(self.step_size,state_index, G)
                if self.distribution is not None:
                    self.distribution[state_index] += 1

    def _run_one_episode(self):
        trajectory = []
        current_state_index = self.env.reset(False)
        while True:
            action_index = self.policy_esitmator.get_action(current_state_index)
            observation = self.env.step(action_index)
            reward = observation[1]
            trajectory.append((current_state_index, action_index, reward))
            done = observation[2]
            if done:
                break
            current_state_index = observation[0]

        return trajectory

    def get_optimal_policy(self):
        return self.policy_esitmator


class REINFORCE:
    def  __init__(self,policy_esitmator,num_episodes, discount_factor=1.0):
        self.actor= Actor(policy_esitmator,num_episodes, discount_factor)
    
    def improve(self):
        self.actor.improve()
        return self.actor.get_optimal_policy()

        