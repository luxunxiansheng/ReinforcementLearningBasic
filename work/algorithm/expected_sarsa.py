import itertools
import sys
from collections import defaultdict

import numpy as np

sys.path.append('/home/ornot/GymRL')
from lib import utility


def expected_sarsa(env, num_episodes, statistics, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-expected_sarsa algorithm: on-policy TD control. Finds the optimal epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        statistics: An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    e_greedy_policy = utility.make_epsilon_greedy_policy(q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        observation = env.reset()

        # One step in the environment
        for t in itertools.count():
            # Take a step
            action = utility.make_decision(e_greedy_policy, observation)
            next_observation, reward, done, _ = env.step(action)

            # Update statistics
            statistics.episode_rewards[i_episode] += reward
            statistics.episode_lengths[i_episode] = t

            expected_next_q = 0

            next_actions = e_greedy_policy(next_observation)

            for action, action_prob in enumerate(next_actions):
                expected_next_q += action_prob * q[next_observation][action]

            td_target = reward + discount_factor * expected_next_q

            td_delta = td_target - q[observation][action]
            q[observation][action] += alpha * td_delta

            if done:
                break

            observation = next_observation

    return q
