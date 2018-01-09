import gym
import math
import random
import numpy as np
# import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

path = '/home/itkdeeplearn/herat1/pong_dqn.pth'

# hyperparameters
# H = 200  # number of hidden layer neurons
# batch_size = 10  # every how many episodes to do a param update?
# learning_rate = 1e-4
BATCH_SIZE = 64
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 200
render = False
D = 80 * 80

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = []

        self.capacity = capacity
        self.position = len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # print(type(self.memory[self.position]))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        for i in self.memory:
            if type(i) == type(None): print('Sampling None!!!', type(i))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.lin1  = nn.Linear(2048, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return F.softmax(self.fc(x))

    def get_weights(self):
        return self.conv2.weight.data.var(1)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        pred = model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        return pred
    else:
        return LongTensor([[random.randrange(2)]])


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    state_batch  = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    state_action_values = model(state_batch).gather(1, action_batch)
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def convert_state(state):
    return Tensor(np.expand_dims(np.expand_dims(state, axis=0), axis=0))


class PG: #network for policy gradient algorithm
    def __init__(self):
        self.w1 = np.random.randn(H, D) / np.sqrt(D)  #input weights
        self.w2 = np.random.randn(H) / np.sqrt(H) #output weights

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self, x):
        h = np.dot(self.w1, x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.w2, h)
        p = sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.w2)
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1': dW1, 'W2': dW2}

    def weights(self):
        return {'W1': self.w1, 'W2': self.w2}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float32)

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


# model and rmsmemory init
# model = PG()
# grad_buffer = {k: np.zeros_like(v) for k, v in model.weights().items()}
# rmsprop_cache = {k: np.zeros_like(v) for k, v in model.weights().items()}

model = DQN()
memory = ReplayMemory(10000)
steps_done = 0

if use_cuda:
    model.cuda()

env = gym.make("Pong-v0")
observation = env.reset()
prev_state = None #previous screen
xs, hs, dlogps, drs = [], [], [], []
reward_sum = 0
num_episodes = 10000
episode_points = []
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
record = False

for i_episode in range(num_episodes):
    prev_state = np.zeros_like(prepro(observation))
    # noinspection PyRedeclaration
    cur_state = prepro(observation) #current state
    if i_episode % 1 == 0: record = True
    for t in count():

        action = select_action(convert_state(prev_state-cur_state))
        observation, reward, done, info = env.step(action[0][0])
        reward = Tensor([reward])

        if not done:
            next_state = cur_state-prepro(observation)
        else:
            next_state = None

        memory.push(convert_state(cur_state), action, convert_state(next_state), reward)
        prev_state = cur_state
        cur_state = next_state

        optimize_model()

        if done:
            reward_sum += reward
            if record:
                print('Reward sum over 20 games: {}, episode {}'.format(reward_sum, i_episode))
                record = False
                episode_points.append(reward_sum)
                rewards = 0
                # plot_points()

            observation = env.reset()
            break

torch.save(model, path)

# Policy gradient method:
#
# while True:
#     if render: env.render()
#
#     cur_x = prepro(observation)
#     x = cur_x - prev_x if prev_x is not None else np.zeros(D)
#     prev_x = cur_x
#
#     # forward the policy network and sample an action from the returned probability
#     aprob, h = model.policy_forward(x)
#     action = 2 if np.random.uniform() < aprob else 3  # roll the dice!
#
#     # record various intermediates (needed later for backprop)
#     xs.append(x)  # observation
#     hs.append(h)  # hidden state
#     y = 1 if action == 2 else 0  # a "fake label"
#     dlogps.append(
#         y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
#
#     # step the environment and get new measurements
#     observation, reward, done, info = env.step(action)
#     reward_sum += reward
#
#     drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
#
#     if done:  # an episode finished
#         episode_number += 1
#
#         # stack together all inputs, hidden states, action gradients, and rewards for this episode
#         epx = np.vstack(xs)
#         eph = np.vstack(hs)
#         epdlogp = np.vstack(dlogps)
#         epr = np.vstack(drs)
#         xs, hs, dlogps, drs = [], [], [], []  # reset array memory
#
#         # compute the discounted reward backwards through time
#         discounted_epr = model.discount_rewards(epr)
#         # standardize the rewards to be unit normal (helps control the gradient estimator variance)
#         discounted_epr -= np.mean(discounted_epr)
#         discounted_epr /= np.std(discounted_epr)
#
#         epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
#         grad = model.policy_backward(eph, epdlogp)
#         for k in model.weights(): grad_buffer[k] += grad[k]  # accumulate grad over batch
#
#         # perform rmsprop parameter update every batch_size episodes
#         if episode_number % batch_size == 0:
#             for k, v in model.weights().items():
#                 g = grad_buffer[k]  # gradient
#                 rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
#                 model.weights()[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
#                 grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
#
#         # boring book-keeping
#         running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
#         episode_points.append(reward_sum)
#         plot_points(episode_points)
#         reward_sum = 0
#         observation = env.reset()  # reset env
#         prev_x = None
