import itertools
import sys
from collections import defaultdict

import numpy as np

sys.path.append('/home/ornot/GymRL')
from lib import utility


def mc_control_importance_sampling(env, num_episodes, statistics, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.

    Returns:
        Q is a dictionary mapping state -> action values.
    """

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    behavior_policy = utility.create_random_policy(env.nA)
    
    target_policy = utility.make_greedy_policy(Q)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in itertools.count():
            # Sample an action from our policy
            action = utility.make_decision(behavior_policy, state)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            statistics.episode_rewards[i_episode] += reward
            statistics.episode_lengths[i_episode] = t

            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        Weight = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]

            # Update the total reward since step t
            G = discount_factor * G + reward

            # Update weighted importance sampling formula denominator
            C[state][action] += Weight

            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (Weight / C[state][action]) * (G - Q[state][action])

            # If the action taken by the behavior policy is not the action
            # taken by the target policy the probability will be 0 and we can break
            if action != np.argmax(target_policy(state)):
                break

            Weight = Weight * 1. / behavior_policy(state)[action]

    return Q
