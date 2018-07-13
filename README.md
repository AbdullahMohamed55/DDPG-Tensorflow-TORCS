# DDPG-Tensorflow-TORCS
This Repo is an implementation of DDPG algorithm using tensorflow on TORCS simluator

DDPG is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn. Policy gradient algorithms utilize a form of policy iteration: they evaluate the policy, and then follow the policy gradient to maximize performance. 
Since DDPG is off-policy and uses a deterministic target policy, this allows for the use of the Deterministic Policy Gradient theorem. DDPG is an actor-critic algorithm as well; it primarily uses two neural networks, one for the actor and one for the critic. 
<p align="center">
  <img width="500" height="300" src="https://yanpanlau.github.io/img/torcs/actor-critic.png">
</p>


These networks compute action predictions for the current state and generate a temporal-difference (TD) error signal each time step. The input of the actor network is the current state, and the output is a single real value representing an action chosen from a continuous action space. The critic’s output is simply the estimated Q-value of the current state and of the action given by the actor. 

The deterministic policy gradient theorem provides the update rule for the weights of the actor network. The critic network is updated from the gradients obtained from the TD error signal. In general, training and evaluating the policy and/or value function with thousands of temporally-correlated simulated trajectories leads to the introduction of enormous amounts of variance in our approximation of the true Q-function (the critic). The TD error signal is excellent at compounding the variance introduced by your bad predictions over time.


We used a replay buffer to store the experiences of the agent during training, and then randomly sample experiences to use for learning in order to break up the temporal correlations within different training episodes. This technique is known as experience replay. 
DDPG uses this. Directly updating the actor and critic neural network weights with the gradients obtained from the TD error signal that was computed from both your replay buffer and the output of the actor and critic networks causes your learning algorithm to diverge (or to not learn at all). It was recently discovered that using a set of target networks to generate the targets for your TD error computation regularizes your learning algorithm and increases stability.


Accordingly, here are the equations for the TD target and the loss function for the critic network.
<p align="center">
  <img width="500" height="100" src="https://ibb.co/dqrZgo">
</p>

Now, as mentioned above, the weights of the critic network can be updated with the
gradients obtained from the loss function. Also, remember that the actor network is

updated with the Deterministic Policy Gradient.
𝐽 ≈1/𝑁 ∑ ∑ 𝑅(𝑠, 𝑎)

We update weights of the critic network
𝑊𝑘+1 = 𝑊𝑘 + α 𝜕𝐽/𝜕θ


### Actor Network
The actor network takes states. And we used for it 2 hidden layers with 300 and 600 hidden units. The output consists of 3 continuous actions, Steering, which is a single unit with tanh activation function. Acceleration, which is a single unit with sigmoid activation function. Brake, another single unit with sigmoid activation function, then we updated the Actor Network from the gradients computed from the Critic Network.


### Critic Network
The critic network takes both the states and action as inputs and outputs the Q-Value. We used 4 hidden layers with 300 then 600 hidden units for the states and 600 hidden units for the actions then the last hidden layer has 600 hidden units. 


