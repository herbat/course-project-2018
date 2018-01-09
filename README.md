# Games neural networks play
My 2018 Neural Networks Course Project at Pazmny Peter Catholic University ITK is a study of game AI-s based on neural networks. 

### History

In 2013, a London-based AI startup called DeepMind published an article which was featured in the Nature magazine. The article can be found [here](https://www.nature.com/articles/nature14236). This learning algorithm is based on two concepts: trial-and-error process of most human and mammal specimens when learning, and the reinforcement mechanism formulated by Bellman - with the purpose to help dynamic programming. These two ideas combined resulted in an algorithm which can mimic the learning mechanism of humans.

First of all, we have to let the network - also called agent in RL - 'try', and give it rewards. This means giving a state input to the network and reading the output as an action, then simulating the next state. Repeating this, the agent will slowly map certain actions to certain states as most rewarding. This can be described as approximating the Q function(hence the name), which describes the optimal action for every state. 

Reinforcement learning is gaining more and more interest among neural network experts, since it can lead to General Artifical Intelligence. Why? Because the agent will do the right thing in most states, without actually knowing what states mean, so an agent working well will be able to learn other tasks as well - or mathematically speaking, approximate different Q functions -, without having to be modified significantly.

### My usage

Based on the techniques described in the previous article, and a tutorial on the [PyTorch page](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), I built a convolutional neural network model using PyTorch framework, which can approximate the Q-function of a Tetris implementation I built myself, and act optimally. 

Currently, my network isn't capable of playing much better than a random agent, but this will change if it gets more training. 

### Preparations

First of all, I tested the network in the tutorial on the openAI gym cartpole, and saw promising imporvements. This wasn't a very accurate test though, since the [cartpole-v0](https://gym.openai.com/envs/CartPole-v0/) game is a much simpler one than Tetris. 

## Model

The model I used is a simplified version of what DeepMind used;

  - one 2D convolutional layer with 1 channel in and 16 channels out
  - one 2D convolutional layer with 16 channels in and 32 channles out 
  - one fully connected layer with 96 inputs and 3 neurons

I used the ReLU activation function across the model.

The original pytorch tutorial contained batchnorm layers, but they significantly lowered speed but didn't increase the actual learning speed, so I discarded them in my own model.

## Training 

To train a convolutional network to play a game, one has to take special measures.

### Method 

Q-learning is based on the concept of reinforcement learning, which can be observed in many species. Reinforcement learning is a process of trial-and-error, in which actions with positive outcomes are rewarded. This makes Q-learning a slow but very intuitive method of learning. For a DQN(Deep Q Network) to learn, one has to have an accurate measurement of reward, since the network will learn only as well as the reward system is constructed. 

Apart from reward, a vital part of Q-learning is memorization. Basically, a DQN learns by replaying its memories randomly, and acting according to the achieved reward of specific actions.

### Data

For the training, I implemented a simpler version of Tetris, which is capable of visualizations as well. I used `matplotlib` for the visualization part. While testing the game, I encountered multiple problems with the visualization, partly thanks to  `matplotlib`'s documentation on `imshow` color mapping being scarce. I found that initiating with a random array made the updating of the color map easier. 

### Experimentations 

At first, training was almost without effect, so I implemented a demo version of tetris, in which I could play and record the transitions of the game. This way, I created memories for the network to remember and learn from. This led to an increase in learning, but still, it was quite slow as it is with all DQN-s.

### Optimization

I used the optimization algorithm of Q learning without any bells or whisthles. This algorithm is basically just using the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation#The_Bellman_equation), and calculating the next action based on the biggest discounted reward. 

## Results

The results weren't very satisfying after a couple hours of training, but this time isn't enough for a DQN to learn a game properly.



