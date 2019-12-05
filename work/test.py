import numpy as np



action_values = [40.0,5.0,6.0,7.0,40.0,3.0,40.0]

for _ in range(30):

    best_action_index = np.random.choice(np.flatnonzero(np.isclose(action_values,max(action_values))))

    print(best_action_index)