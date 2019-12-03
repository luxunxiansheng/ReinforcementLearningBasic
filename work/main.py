from tqdm import tqdm

from agent.grid_world_agent import Grid_World_Agent

def policy_iteration():
    grid_world_agent = Grid_World_Agent()

    for _ in tqdm(range(1000)):
        grid_world_agent.evaluate_policy()
        
        grid_world_agent.show_state_values()

        print("**************************************")

    grid_world_agent.improve_policy()
    grid_world_agent.evaluate_policy()

    grid_world_agent.show_state_values()
       

def value_iteration():
    grid_world_agent = Grid_World_Agent()

    for _ in tqdm(range(10)):
        grid_world_agent.value_iteration()
        grid_world_agent.show_state_values()
        print("#######################################") 


def main():
    policy_iteration()


if __name__ == "__main__":
    main()


