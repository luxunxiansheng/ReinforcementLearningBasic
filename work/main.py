from tqdm import tqdm

from agent.grid_world_agent import Grid_World_Agent


def main():
    grid_world_agent = Grid_World_Agent()

    for _ in tqdm(range(10)):
        grid_world_agent.evaluate_policy()
        grid_world_agent.show_state_values()


if __name__ == "__main__":
    main()
