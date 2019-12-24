from tqdm import tqdm

class V_Monte_Carlo_Method:
    def __init__(self,v_table,env,episodes=1000,discount=1.0):
        self.v_table = v_table
        self.env =env
        self.episodes=episodes
        self.discount = discount
        self.return_table = self._init_returns()

    def _init_returns(self):
        return_table= {}
        for state_index in self.v_table:
            return_table[state_index]= (0,0.0)
        return return_table

    def evaluate(self,policy):
        for _ in tqdm(range(0, self.episodes)):
            while True:
                action_index= policy.select_action(self.env.s)
                obervation = self.env.step(action_index)
                if self.return_table[state_index][0]==0:
                    self.return_table[state_index][0]= self.return_table[state_index][0]
                    self.return_table[state_index][1]= self.discount*self.return_table[state_index]+obervation[1]
                    self.v_table[state_index] = self.return_table[state_index][1]/self.return_table[state_index][0]
          
                state_index= obervation[0]
                done=obervation[2]
                if done:
                    break


        