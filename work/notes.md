# Bellman is Good

## 1. Bellman optimality equation
> $Q^*(s,a)= E[R_s^a+\gamma{max_{a'}Q^*(s',a')}]=\sum_{s',r}P_{s,a}^{s'}[R_s^a+\gamma{max_{a'}Q^*(s',a')}]$

Bellman optimality equation is the core of the reinforcement learning. No matter what the algorithm is , its ultimate goal is to solve the Bellman optimality equaiton. Today，a common conclusiion in reinforcement learning domain is that policy based algorithms are very different from those value based, but we don't think so. We believe value based methods are the only needed solution to solve the Bellman Optimality Equation and policy based methods can be fit into the same framework as value based do.

## 2. Three components

We define the following three parts to construct the framework to solve a Bellman optimality equation.

### 2.1 Exploitator

Exploitator calculates the optimal V or Q value. Once done,
the optimal policy could be inferenced from the values  immediately.

### 2.2 actor

If we don't know the dynamics of the enviroment, that is, we have no idea on the state transition funciton and/or the rewards, we have to sample data from the enviroment and then give the data to exploitator to calculate the optimal values. Basically, the state and reward space will be very huge. Given limited computing resources, it is impossible for an agent to explore the whole state and reward spaces. Furthmore, it is not wise to take     identical efforts on all of the states when  calculating the values. For the sake of data efficiency, we will give those high value targets more attentions then those of low values. It is actor's responsibility to define the exploration policy and continue to enhance it.

### 2.3 Experiencer

Experiencer will follow the policy defined by the actor to take a tour. States and rewards will be sampled. 

## 3. Off-Policy and On-Policy
The final optimal policy will be inferenced from optimal values. We name this policy as target policy. Greedy policy will be the default policy for  exploritator to solve Bellman optimality equation.

For off policy algorithms, the actor will take so callled behavior policy that is different from target policy to find states and rewards of high values for exploritator.

For on policy algorithms, both the actor and the exploritator will take the same policy.
This policy will converge to the greedy policy in a long run. 

## 4. Policy and Optimal Policy
In reinforcement learning, optimal policy exists only in an ideal way. It is objective. What reinforcment learning does is to approach this policy. We think the immediate policy could also be called optimal policy except not exact.