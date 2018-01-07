import math
import time
import random
import numpy as np
#import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import count
from functools import reduce
from collections import Counter
from collections import namedtuple
from random import randint as randi

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

WIDTH  = 8
HEIGHT = 16

#HYPERPARAMETERS
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 200

# TETRIS GAME PARTS ---------------------------------------------------
class Direction:

    def __init__(self, v=0, h=0, t=0):
        self.vertical = v
        self.horizontal = h
        self.turn = t


class Position:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Tetris():

    def __init__(self):
        self.width  = WIDTH
        self.height = HEIGHT
        self.speed  = 2
        self.cstep  = 0
        self.state  = np.zeros((self.height, self.width), np.float32)
        self.nomove = False
        self.reward = 0
        self.points = 0
        self.tmppts = 0
        self.stone  = tetris_shapes[randi(0, 6)]
        self.stone_pos = Position(math.ceil(self.width/2) - 2,  0)
        self.stone_dir = Direction(0, 0, 0)
        self.game_over = False

    def step(self, action):
        global points
        self.tmppts = self.points
        actions = {
            0: Direction(0,  1, 0),
            1: Direction(0, -1, 0),
            2: Direction(0,  0, 1),
            3: Direction(1,  0, 0)
        }
        if self.game_over: return self.points, True
        self.remove_lines()
        # print('Steppin da {}th step'.format(self.cstep+1))
        self.stone_dir = actions[action]
        self.cstep += 1
        if self.cstep%self.speed == 0: #move one down every x steps
            self.stone_dir = actions[3]
            self.points += 1
            if self.nomove and self.check_collision(self.stone, self.stone_pos, actions[3]):
                self.insert_stone()
                r = self.reward - self.getreward()
                self.reward = self.getreward()
                return self.points - self.tmppts + r, False


        self.move()
        return self.points-self.tmppts, False

    def move(self):

        if self.check_collision(self.stone, self.stone_pos, self.stone_dir):
            self.stone_dir = Direction(0, 0, 0)
            self.nomove = True
            return

        if self.stone_dir.turn: self.stone = self.rotate_clockwise(self.stone)

        self.stone_pos.x += self.stone_dir.horizontal
        self.stone_pos.y += self.stone_dir.vertical

    def check_collision(self, shape, pos, dir):
        if dir.turn: shape = self.rotate_clockwise(shape)

        o_x = pos.x + dir.horizontal
        o_y = pos.y + dir.vertical

        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                if (not(cy + o_y < self.height and cx + o_x < self.width and
                        cy + o_y >= 0          and cx + o_x >= 0)
                    or (cell and self.state[cy + o_y][cx + o_x])):
                    return True

        return False

    def insert_stone(self):
        if not self.check_collision(self.stone, self.stone_pos, Direction(1, 0, 0)): self.nomove = False; self.move(); return False
        o_x = self.stone_pos.x
        o_y = self.stone_pos.y

        for cy, row in enumerate(self.stone):
            for cx, cell in enumerate(row):
                self.state[cy + o_y][cx + o_x] += cell

        self.nomove = False
        self.new_stone()
        return True

    def rotate_clockwise(self, shape):

        return [[shape[y][x]
                 for y in range(len(shape))]
                for x in range(len(shape[0]) - 1, -1, -1)]

    def show(self):
        if self.game_over: return self.state
        o_x = self.stone_pos.x
        o_y = self.stone_pos.y

        state = deepcopy(self.state)

        for cy, row in enumerate(self.stone):
            for cx, cell in enumerate(row):
                state[cy + o_y][cx + o_x] += cell

        return state

    def new_stone(self):

        shape = tetris_shapes[randi(0, 6)]
        pos   = Position(2,  0)
        dir   = Direction(0, 0, 0)

        if self.check_collision(shape, pos, dir):
            self.game_over = True
            return

        self.stone = shape
        self.stone_pos = pos
        self.stone_dir = dir

    def remove_lines(self):
        global points

        removables = []
        for i, row in enumerate(self.state):
            if row.all() != 0:
                removables.append(i)

        if len(removables) > 0:
            self.state = np.delete(self.state, removables, 0)
            print('{} line(s) removed'.format(len(removables)))
            for _ in removables:
                self.state = np.insert(self.state, 0, 0, axis=0)


        if len(removables) == 4: self.points += points['tetris']
        else: self.points += len(removables) * points['line']

    def roughness(self):
        peaks = self.getpeaks()
        res = 0
        prev = -1
        for i in peaks:
            if prev < 0: prev = i
            else: res += abs(prev-i); prev = i
        return res

    def holes(self):
        result = 0
        peaks = self.getpeaks()
        for i, row in enumerate(self.state):
            for j, cell in enumerate(row):
                if cell == 0 and HEIGHT-i < peaks[j]: result += 1
        return result

    def c_height(self):
        return reduce((lambda x, y: x+y), self.getpeaks())

    def m_height(self):
        return max(self.getpeaks())

    def getpeaks(self):
        peaks = np.zeros(WIDTH)
        for i, row in enumerate(self.state.T):
            try: peaks[i] = HEIGHT - next((i for i, x in enumerate(row) if x), None)
            except Exception: peaks[i] = 0
        return peaks

    def getreward(self):
        return 0 - self.roughness()+self.holes()+self.c_height()+self.m_height()



# ---------------------------------------------------------------------

tetris_shapes_fancy = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]

tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 1, 1],
     [1, 1, 0]],

    [[1, 1, 0],
     [0, 1, 1]],

    [[1, 0, 0],
     [1, 1, 1]],

    [[0, 0, 1],
     [1, 1, 1]],

    [[1, 1, 1, 1]],

    [[1, 1],
     [1, 1]]
]

points = {
    'drop'  : 1,
    'line'  : 100,
    'tetris': 800,
}

# ---------------------------------------------------------------------
# NN PARTS ------------------------------------------------------------
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


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
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.index = 0
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        # self.lin1  = nn.Linear(, 64)
        self.fc = nn.Linear(96, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return F.softmax(self.fc(x.view(x.size(0), -1)))

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
        return LongTensor([[random.randrange(3)]])


def plot_points():
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Points')
    plt.plot(np.array(episode_points))

    plt.pause(0.001)  # pause a bit so that plots are updated


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
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def convert_state(state):
    return Tensor(np.expand_dims(np.expand_dims(state, axis=0), axis=0))


def common(lst):
    data = Counter(lst)
    return data.most_common(1)[0]


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# ---------------------------------------------------------------------

# INIT ----------------------------------------------------------------

game = Tetris()
model = DQN() # torch.load('dqn_chechpoint.pth')

if use_cuda:
    model.cuda()

num_episodes = 2000
episode_points = []
loss = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
memory = ReplayMemory(10000)
steps_done = 0

# ---------------------------------------------------------------------

# ANIMATION -----------------------------------------------------------

# fig = plt.figure()
#
# im = plt.imshow(np.random.rand(HEIGHT, WIDTH), animated=True)
#
# def update(*args):
#     arr = game.step(randi(0, 4))
#     if arr is None: ani.event_source.stop(); return
#     im.set_array(arr/7)
#     return im,
#
# ani = animation.FuncAnimation(fig, update, interval=200, blit=True)
#
# plt.show()

# ---------------------------------------------------------------------

last_sync = 0
starttime = time.time()
c1w_last = model.get_weights()
record = False
rewards = 0
actions = []
for i_episode in range(num_episodes):
    # noinspection PyRedeclaration
    state = convert_state(game.show())
    if i_episode % 10 == 0: record = True
    for t in count():
        action = select_action(state)
        if record: actions.append(action.tolist()[0][0])
        reward, done = game.step(action[0][0])
        reward = Tensor([reward])

        if not done:
            next_state = convert_state(game.show())
        else:
            next_state = None

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        if done:
            rewards += reward
            if record:
                a, c = common(actions)
                print('Most common action: {}, {} percent'.format(a, c/len(actions)*100))
                actions = []
                record = False
                episode_points.append(rewards/10)
                rewards = 0
                # plot_points()

            del game
            game = Tetris()
            break

torch.save(model, 'dqn_checkpoint.pth')

print(time.time() - starttime)
print('Complete')
# plt.ioff()
# plt.show()
