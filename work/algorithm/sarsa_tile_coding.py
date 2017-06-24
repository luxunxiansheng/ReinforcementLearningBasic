import sys
import itertools


sys.path.append('/home/ornot/GymRL')
from lib import utility


def sarsa(env, estimator, num_episodes, statistics, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    sarsa algorithm for on-policy TD control using Function Approximation.
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:

    """

    for i_episode in range(num_episodes):
        # The policy we're following
        e_greedy_policy = utility.make_epsilon_greedy_policy_with_fa(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = statistics.episode_rewards[i_episode - 1]
        sys.stdout.flush()

        # Reset the environment and pick the first action
        obvservation = env.reset()
        action = utility.make_decision(e_greedy_policy, obvservation)
        for t in itertools.count():
            next_observation, reward, done, _ = env.step(action)

            # Update statistics
            statistics.episode_rewards[i_episode] += reward
            statistics.episode_lengths[i_episode] = t

            next_action = utility.make_decision(
                e_greedy_policy, next_observation)
            q_values_next = estimator.predict(next_observation, next_action)

            td_target = reward + discount_factor * q_values_next

            # Update the function approximator using our target
            estimator.update(obvservation, action, td_target)

            print("\rStep {} @ Episode {}/{} ({})".format(t,i_episode + 1, num_episodes, last_reward), end="")

            if done:
                break
            action = next_action
            obvservation = next_observation
