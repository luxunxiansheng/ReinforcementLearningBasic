from common import ActorBase
import sys
from collections import namedtuple

import numpy as np

from lib import utility

from tqdm import tqdm

class Actor(ActorBase):
    def __init__(self,policy_esitmator,step_size=0.01):
        self.policy_esitmator = policy_esitmator
        self.step_size = step_size

    def improve(self,*args):
        state_index=args[0]
        G= args[1]
        self.estimator.update(self.step_size,state_index, G)
        
    def get_optimal_policy(self):
        return self.policy_esitmator


class REINFORCE:
    def  __init__(self,policy_esitmator,num_episodes,step_size=0.01,discount_factor=1.0):
        self.actor= Actor(policy_esitmator,step_size)
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
            
    def improve(self):
        for _ in tqdm(range(0,self.num_episodes)):
            trajectory = self._run_one_episode()
            G = 0.0
            for state_index, _, reward in trajectory[::-1]:
                # The return for current state_action pair
                G = reward + self.discount*G
                self.actor.improve(state_index,G)
                

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

        