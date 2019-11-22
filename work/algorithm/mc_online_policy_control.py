import itertools
import sys
from collections import defaultdict

import numpy as np
from lib import utility




def mc_control_epsilon_greedy(env, num_episodes, statistics, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        statistics: namedTuple of statistics informaiton 

        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = utility.make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

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
            action = utility.make_decision(policy, state)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            statistics.episode_rewards[i_episode] += reward
            statistics.episode_lengths[i_episode] = t

            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        # the tuple is (state,action)
        state_action_in_episode = set([(x[0], x[1]) for x in episode])
        for state, action in state_action_in_episode:
            state_action = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2] * (discount_factor**i)
                     for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state_action] += G
            returns_count[state_action] += 1.0
            Q[state][action] = returns_sum[state_action] / \
                returns_count[state_action]

        # The policy is improved implicitly by changing the Q dictionar

    return Q
