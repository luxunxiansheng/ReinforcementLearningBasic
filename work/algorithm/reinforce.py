import itertools
import sys
from collections import namedtuple

import numpy as np

sys.path.append('/home/ornot/GymRL')
from lib import utility


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
