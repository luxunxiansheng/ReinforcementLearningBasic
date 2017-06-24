import numpy as np


def create_random_policy(nA):
    """
    Creates a random policy function.

    Args:
        nA: Number of actions in the environment.

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) / nA
        return A
    return policy_fn


def make_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.

    Args:
        Q: A dictionary that maps from state -> action values

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """

    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def make_decision(policy, observation):
    """
      determine the next action to take in current state based on the policy

    Args:
       policy: A policy functor to generate the action probability distribution
       state"  A input state which will be feeded to the policy functor

    Returns:  A action selected randomly from the action probability distribution 
    """
    action_probs = policy(observation)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    return action


def make_epsilon_greedy_policy_with_fa(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = []
        for action in range(nA):
            state_action_value = estimator.predict(observation, action)
            q_values.append(state_action_value)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def make_greedy_policy_with_fa(estimator, nA):
    def policy_fn(observation):
        A = np.zeros(nA, dtype=float)
        q_values = []
        for action in range(nA):
            state_action_value = estimator.predict(observation, action)
            q_values.append(state_action_value)
        best_action = np.argmax(q_values)
        A[best_action] = 1.0
        return A

    return policy_fn
