import os
import math
import curses
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from functools import reduce
from collections import namedtuple
from random import randint as randi

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torch
Tensor = torch.FloatTensor

#stdscr = curses.initscr()

WIDTH  = 8
HEIGHT = 16
SPEED  = 4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
        self.speed  = SPEED
        self.cstep  = 0
        self.state  = np.zeros((self.height, self.width), np.uint8)
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
        print('R:{} Ho:{} He:{} MH:{}'.format(self.roughness(), self.holes(), self.c_height(), self.m_height()))
        return 0 - self.roughness()+self.holes()+self.c_height()+self.m_height()


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
    'drop'  : 10,
    'line'  : 100,
    'tetris': 800,
}

game = Tetris()

fig = plt.figure()

im = plt.imshow(np.random.rand(HEIGHT, WIDTH), animated=True)

def convert_state(state):
    return Tensor(np.expand_dims(np.expand_dims(state, axis=0), axis=0))

def save_game(db):
    try:
        fr = open('training_data.npy', 'rb')
        x = np.load(fr)
        fr.close()
        fw = open('training_data.npy', 'wb')
        x = np.concatenate((x, db))
        print('Saving {0} moves...'.format(len(db)))
        np.save('training_data.npy', x)
        print('{0} data points in the training set'.format(len(x)))
    except Exception as e:
        print('no training file exists. Creating one now...')
        fw = open('training_data.npy', 'wb')
        print('Saving {0} moves...'.format(len(db)))
        np.save('training_data.npy', db)

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


database = []

model = torch.load('../dqn_checkpoint.pth')

def update(*args):
    state = game.show()
    # inp = stdscr.getch()
    # action = 0
    # if inp == ord('w'): action = 4
    # elif inp == ord('a'): action = 2
    # elif inp == ord('s'): action = 3
    # elif inp == ord('d'): action = 1

    action = model(Variable(convert_state(state), volatile=True).type(Tensor)).data
    print(action)
    action = action.max(1)[1].view(1, 1).tolist()[0][0]
    reward, done = game.step(action)
    next_state = game.show()
    # database.append(Transition(state, action, next_state, reward))
    if done: ani.event_source.stop(); return
    im.set_array(next_state)
    return im,

ani = animation.FuncAnimation(fig, update, interval=50, blit=True)

plt.show()