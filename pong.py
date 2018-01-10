import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class Categorical():

    def __init__(self, probs):
        if probs.dim() != 1 and probs.dim() != 2:
            # TODO: treat higher dimensions as part of the batch
            raise ValueError("probs must be 1D or 2D")
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1, True).squeeze(-1)

    def sample_n(self, n):
        if n == 1:
            return self.sample().expand(1, 1)
        else:
            return torch.multinomial(self.probs, n, True).t()

    def log_prob(self, value):
        p = self.probs / self.probs.sum(-1, keepdim=True)
        if value.dim() == 1 and self.probs.dim() == 1:
            # special handling until we have 0-dim tensor support
            return p.gather(-1, value).log()

        return p.gather(-1, value.unsqueeze(-1)).squeeze(-1).log()

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
CONVNETS = False


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

        if CONVNETS:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.lin = nn.Linear(3200, 200)
        else:
            self.lin = nn.Linear(6400, 200)

        self.out = nn.Linear(200, 2)
        self.outputs = []
        self.rewards = []

    def forward(self, x):
        if CONVNETS:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.lin(x.view(x.size(0), -1)))
        else:
            x = F.relu(self.lin(x))

        x = F.softmax(self.out(x), dim=1)
        return x

    def get_weights(self):
        return self.conv2.weight.data.var(1)

def select_action(state):
    state = Tensor(np.expand_dims(np.expand_dims(state, axis=0), axis=0)) if CONVNETS else torch.from_numpy(state).float().unsqueeze(0)
    if use_cuda: state = state.cuda()
    print(type(state))
    pred = model(Variable(state))
    m = Categorical(pred)
    action = m.sample()
    model.outputs.append(m.log_prob(action))
    return 2 if pred.data[0][0] < np.random.uniform() else 3

def optimize_model():

    advantage = Tensor(discount_rewards(np.asarray(model.rewards)))
    advantage = (advantage - advantage.mean())/(advantage.std() + np.finfo(np.float32).eps)
    loss = []

    for o, a in zip(model.outputs, advantage):
        loss.append(-o * a)

    optimizer.zero_grad()
    loss = torch.cat(loss).sum()
    print('loss calculated: {}'.format(loss.data[0]))
    loss.backward()
    print('backprop done')
    optimizer.step()
    print('optimizing done')

    del model.outputs[:]
    del model.rewards[:]

def prepro(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float) if CONVNETS else I.astype(np.float).ravel()

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

        action = select_action(prev_state-cur_state) #select action probabilistically
        observation, reward, done, info = env.step(action) #save info

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


