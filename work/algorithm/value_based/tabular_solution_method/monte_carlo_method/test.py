
import copy
import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)


from tqdm import tqdm

from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_es_control import MonteCarloESControl
from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_off_policy_control import MonteCarloOffPolicyControl
from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_off_policy_evaluation import MonteCarloOffPolicyEvaluation
from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_on_policy_control import MonteCarloOnPolicyControl
from env.blackjack import BlackjackEnv
from lib.plotting import plot_episode_error
from policy.policy import DiscreteStateValueBasedPolicy
from test_setup import get_env

sys.path.append("/home/ornot/workspace/ReinforcementLearningBasic/work")



real_env = get_env("blackjack")


def test_q_mc_es_control_method(env):
    q_table = env.build_Q_table()
    policy_table = env.build_policy_table()
    behavior_policy = DiscreteStateValueBasedPolicy(policy_table)
    rl_method = MonteCarloESControl(q_table, behavior_policy,env,8000)
    optimal_policy=rl_method.improve()
    env.show_policy(optimal_policy)

test_q_mc_es_control_method(real_env)

def test_mc_onpolicy_control_method(env):
    q_table = env.build_Q_table()
    policy_table = env.build_policy_table()
    target_policy = DiscreteStateValueBasedPolicy(policy_table)
    rl_method = MonteCarloOnPolicyControl(q_table,target_policy,env,0.8,8000)
    optimal_policy=rl_method.improve()
    env.show_policy(optimal_policy)

test_mc_onpolicy_control_method(real_env)


def test_mc_offpolicy_control_method(env):
    q_table = env.build_Q_table()
    policy_table = env.build_policy_table()
    behavior_policy = DiscreteStateValueBasedPolicy(policy_table)
    target_policy = copy.deepcopy(behavior_policy)
    rl_method = MonteCarloOffPolicyControl(q_table, behavior_policy, target_policy, env,8000)
    optimal_policy=rl_method.improve()
    env.show_policy(optimal_policy)

test_mc_offpolicy_control_method(real_env)


def test_mc_offpolicy_evaluation_method_for_blackjack():
    env = BlackjackEnv(False)

    q_table = env.build_Q_table()

    # Random behavior policy
    b_policy_table = env.build_policy_table()
    b_policy = DiscreteStateValueBasedPolicy(b_policy_table)

    # spcific target policy only for blackjack
    t_policy_table = env.build_policy_table()
    for state_index, _ in t_policy_table.items():
        card_sum = state_index[0]
        if card_sum < 20:
            t_policy_table[state_index][BlackjackEnv.HIT] = 1.0
            t_policy_table[state_index][BlackjackEnv.STICK] = 0.0
        else:
            t_policy_table[state_index][BlackjackEnv.HIT] = 0.0
            t_policy_table[state_index][BlackjackEnv.STICK] = 1.0
    t_policy = DiscreteStateValueBasedPolicy(t_policy_table)

    error = []
    init_state = env.reset()
    for episode in tqdm(range(100)):
        error_square = 0.0
        for _ in range(100):
            rl_method = MonteCarloOffPolicyEvaluation(q_table, b_policy, t_policy, env, episode)
            current_q_value= rl_method.evaluate()
            error_square = error_square+(current_q_value[init_state][BlackjackEnv.HIT] + 0.27726)*(current_q_value[init_state][BlackjackEnv.HIT] + 0.27726)
        
        error.append(error_square/100)

    plot_episode_error(error)        
    
test_mc_offpolicy_evaluation_method_for_blackjack()






