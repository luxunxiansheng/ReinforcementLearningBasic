from tqdm import tqdm

from agent.grid_world_agent import Grid_World_Agent
from agent.gamblers_problem_agent import Gamblers_Problem_Agent
from policy.policy import Greedy_Action_Selector,e_Greedy_Action_Selector


def main():
    gamblers_problem_agent = Gamblers_Problem_Agent()
    gamblers_problem_agent.value_iteration()
    

if __name__ == "__main__":
    main()


