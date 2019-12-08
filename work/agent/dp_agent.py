from agent.base_agent import Base_Agent
from algorithm.implicity_policy import dynamic_programming
from policy.policy import Tabular_Implicit_Policy


class DP_Agent(Base_Agent):
    def __init__(self,env):
        self.env  = env
        self.policy = Tabular_Implicit_Policy(env.build_Q_table())

    def improve_policy_once(self):
        dynamic_programming.policy_improve(self.policy)

    def evaluate_policy_once(self):
        dynamic_programming.policy_evaluate(self.policy,self.env)
        
    def q_value_iteration(self):
        dynamic_programming.q_value_iteration(self.policy,self.env)
       

    
    def make_decision(self):
        pass
