import gym
import itertools
import numpy as np
import sys

sys.path.append('/home/ornot/GymRL')

from lib import utility


def q_learning(env, estimator, num_episodes, statistics, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        statistics:EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    """
    greedy_policy = utility.make_greedy_policy_with_fa(
        estimator, env.action_space.n)

    for i_episode in range(num_episodes):
        # The policy we're following
        epsilon_greedy_policy = utility.make_epsilon_greedy_policy_with_fa(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = statistics.episode_rewards[i_episode - 1]
        sys.stdout.flush()

        # Reset the environment and pick the first action
        observation = env.reset()

        # One step in the environment
        for t in itertools.count():
                    
            # Make the decision
            action = utility.make_decision(epsilon_greedy_policy, observation)

            # Take a step
            next_observation, reward, done, _ = env.step(action)

            # Update statistics
            statistics.episode_rewards[i_episode] += reward
            statistics.episode_lengths[i_episode] = t

            # TD Update
            next_actions = greedy_policy(next_observation)
            q_values_next_max = estimator.predict(
                next_observation, np.argmax(next_actions))

            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * q_values_next_max

            # Update the function approximator using our target

            estimator.update(observation, action, td_target)

            print("\rStep {} @ Episode {}/{} ({})".format(t,
                                                          i_episode + 1, num_episodes, last_reward), end="")

            if done:
               break

            observation = next_observation

