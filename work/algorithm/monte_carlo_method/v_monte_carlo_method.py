import copy

from tqdm import tqdm


class V_Monte_Carlo_Method:
    def __init__(self, v_table, env, episodes=500000, discount=1.0):
        self.v_table = v_table
        self.env = env
        self.episodes = episodes
        self.discount = discount
        self.return_table = self._init_returns()

    def _init_returns(self):
        return_table = {}
        for state_index in self.v_table:
            return_table[state_index] = (0, 0.0)
        return return_table

    def evaluate(self, policy):
        for _ in tqdm(range(0, self.episodes)):
            trajectory=[]
            current_state_index = self.env.reset()
            while True:
                action_index = policy.select_action(current_state_index)
                observation= self.env.step(action_index)
                reward=observation[1]
                trajectory.append((current_state_index,reward)) 
                
                current_state_index = observation[0]
                done = observation[2]
                if done:
                    break
                           
            R = 0.0               
            for state_index,reward in trajectory[::-1]:
                R = reward+self.discount*R
                return_tuple = (self.return_table[state_index][0]+1,self.return_table[state_index][1]+R)
                self.return_table[state_index]=return_tuple
                self.v_table[state_index] = self.return_table[state_index][1]/self.return_table[state_index][0]
                
        
                 

