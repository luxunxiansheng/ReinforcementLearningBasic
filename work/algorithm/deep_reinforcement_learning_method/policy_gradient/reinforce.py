import itertools
import sys
from collections import namedtuple

import numpy as np
import numpy

sys.path.append('/home/ornot/GymRL')
from lib import utility

from tqdm import tqdm

class REINFORCE:
    def __init__(self,policy_esitmator,num_episodes, discount_factor=1.0) -> None:
        self.policy_esitmator = policy_esitmator
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor

    
    def improve(self):
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




def reinforce(env, statistics, policy_estimator, value_estimator, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        statistics:  An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards. 
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    """

    Transition = namedtuple(
        "Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()

        episode = []

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = policy_estimator.predict(state)
            action = np.random.choice(
                np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(state=state, action=action,
                                      reward=reward, next_state=next_state, done=done))

            # Update statistics
            statistics.episode_rewards[i_episode] += reward
            statistics.episode_lengths[i_episode] = t

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1,
                                                          num_episodes, statistics.episode_rewards[i_episode - 1]), end="")
            # sys.stdout.flush()

            if done:
                break

            state = next_state

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor**i *
                               t.reward for i, t in enumerate(episode[t:]))
            # Update our value estimator
            value_estimator.update(transition.state, total_return)
            # Calculate baseline/advantage
            baseline_value = value_estimator.predict(transition.state)
            advantage = total_return - baseline_value
            # Update our policy estimator
            policy_estimator.update(
                transition.state, advantage, transition.action)
