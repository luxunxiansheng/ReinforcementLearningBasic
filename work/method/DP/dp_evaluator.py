from method.base_critic import Base_Critic


class DP_Critic(Base_Critic):

    def evaluate(self,current_state,current_action,policy,q_table,transition):
            
        q_value = 0
        for prob_to_next_state,next_state,reward,_ in transition[current_state][current_action]:
            
            # calculate the value of the next state
            next_state_value= 0.0
            for action_prob in policy.get_probability_at_state(next_state):
                next_state_value +=     

            
         x