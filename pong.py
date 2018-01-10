import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

path = '/home/itkdeeplearn/herat1/pong_dqn.pth'

# hyperparameters
l_r = 1e-3
gamma = 0.99
decay_rate = 0.99
BATCH_SIZE = 5
D = 80 * 80

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

class PG(nn.Module):

    def __init__(self):
        super(PG, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.lin = nn.Linear(2048, 256)
        self.out = nn.Linear(256, 3)
        self.outputs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.lin(x.view(x.size(0), -1)))
        x = F.softmax(self.out(x), dim=1)
        return x

    def get_weights(self):
        return self.conv2.weight.data.var(1)


def select_action(state):
    probs = model(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    model.outputs.append(m.log_prob(action))
    return action.data[0]


def optimize_model():

    advantage = Tensor(discount_rewards(np.asarray(model.rewards)))
    advantage = (advantage - advantage.mean())/(advantage.std() + np.finfo(np.float32).eps)
    loss = []

    for o, a in zip(model.outputs, advantage):
        loss.append(-o * a)

    optimizer.zero_grad()
    loss = torch.cat(loss).sum()
    loss.backward()
    optimizer.step()

    del model.outputs[:]
    del model.rewards[:]

def convert_state(state):
    return Tensor(np.expand_dims(np.expand_dims(state, axis=0), axis=0))


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float)

#
# def plot_points(ep):
#     plt.figure(2)
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Points')
#     plt.plot(np.array(ep))
#
#     plt.pause(0.001)  # pause a bit so that plots are updated

model = PG()

if use_cuda:
    model.cuda()

# environment and basic shit
env = gym.make("Pong-v0")
observation = env.reset()
prev_state = None #previous screen
reward_sum = 0
num_episodes = 5000
episode_points = []
optimizer = optim.RMSprop(model.parameters(), lr=l_r, weight_decay=decay_rate)
# calculating the policy gradient
rewards = []

for i_episode in range(num_episodes):
    prev_state = np.zeros_like(prepro(observation))
    # noinspection PyRedeclaration
    cur_state = prepro(observation) #current state

    for t in count():

        action = select_action(convert_state(prev_state-cur_state)) #select action probabilistically
        observation, reward, done, info = env.step(action+1) #save info

        model.rewards.append(reward)

        next_state = cur_state-prepro(observation) if not done else None #get next state if not done
        prev_state = cur_state
        cur_state = next_state

        if reward != 0: reward_sum += reward

        if done: # if done, print rewards
            if i_episode % BATCH_SIZE == 0 and i_episode>0: # change policy gradient
                optimize_model()
                rewards = []

            episode_points.append(reward_sum)
            print('Reward sum over 20 games: {}, episode {}'.format(reward_sum, i_episode))
            reward_sum = 0
            # plot_points()
            observation = env.reset()
            break

torch.save(model, path)


