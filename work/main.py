from tqdm import tqdm

from agent.grid_world_agent import Grid_World_Agent
from policy.policy import Greedy_Action_Selector,e_Greedy_Action_Selector

def value_iteration():
    grid_world_agent_value_it = Grid_World_Agent()
    
    for _ in range(10):
        grid_world_agent_value_it.value_iteration()
        grid_world_agent_value_it.show_optimal_state_values()
        print("**************************************")

def policy_evaluation():
    grid_world_agent_policy_eva = Grid_World_Agent()
    
    for _ in range(10):
        grid_world_agent_policy_eva.evaluate_policy()
        grid_world_agent_policy_eva.show_state_value()
        print("**************************************")

def policy_iteration():
    grid_world_agent_policy_it = Grid_World_Agent()
    grid_world_agent_policy_it.policy.set_action_selector(Greedy_Action_Selector())
    
    for _ in range(10):
        grid_world_agent_policy_it.evaluate_policy()
        grid_world_agent_policy_it.show_optimal_state_values()
        print("**************************************")
   


def main():
    policy_iteration()
    #policy_evaluation()


if __name__ == "__main__":
    main()


