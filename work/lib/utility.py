import numpy as np

def create_distribution_greedily():
    def create_fn(values):
        num_values = len(values)
        value_probs = np.zeros(num_values, dtype=float)
        best_action_index = max(values,key=values.get)
        value_probs[best_action_index] = 1.0
        return value_probs
    return create_fn

def create_distribution_randomly():
    def create_fn(values):
        num_values = len(values)
        value_probs = np.ones(num_values, dtype=float) / num_values
        return value_probs
    return create_fn

def create_distribution_epsilon_greedily(epsilon):
    def create_fn(values):
        num_values = len(values)
        value_probs = np.ones(num_values, dtype=float) *epsilon / num_values
        best_action_index = np.argmax(values)
        value_probs[best_action_index] += (1.0 - epsilon)
        return value_probs
    return  create_fn
    



















def create_random_policy(num_actions):
    """
    Creates a random policy function.

    Args:
        num_actions: Number of actions in the environment.

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    def policy_fn(observation):
        actions = np.ones(num_actions, dtype=float) / num_actions
        return actions
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

    def policy_fn(observation):
        actions = np.zeros_like(Q[observation], dtype=float)
        best_action = np.argmax(Q[observation])
        actions[best_action] = 1.0
        return actions
    return policy_fn


def make_epsilon_greedy_policy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        num_actions: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        actions = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[observation])
        actions[best_action] += (1.0 - epsilon)
        return actions
    return policy_fn


def make_decision(policy, observation):
    """
      determine the next action to take in current state based on the policy

    Args:
       policy: A policy functor to generate the action probability distribution
       observation"  A input state which will be feeded to the policy functor

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
        num_actions: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length num_action.

    """
    def policy_fn(observation):
        actions = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = []
        for action in range(num_actions):
            state_action_value = estimator.predict(observation, action)
            q_values.append(state_action_value)
        best_action = np.argmax(q_values)
        actions[best_action] += (1.0 - epsilon)
        return actions
    return policy_fn


def make_greedy_policy_with_fa(estimator, num_actions):
    def policy_fn(observation):
        actions = np.zeros(num_actions, dtype=float)
        q_values = []
        for action in range(num_actions):
            state_action_value = estimator.predict(observation, action)
            q_values.append(state_action_value)
        best_action = np.argmax(q_values)
        actions[best_action] = 1.0
        return actions

    return policy_fn
