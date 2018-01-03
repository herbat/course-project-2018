import os
import math
import curses
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from collections import namedtuple
from random import randint as randi


import torch
Tensor = torch.FloatTensor


stdscr = curses.initscr()

WIDTH  = 8
HEIGHT = 12


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
        self.speed  = 10
        self.cstep  = 0
        self.state  = np.zeros((self.height, self.width), np.uint8)
        self.nomove = False
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
            0: Direction(0,  0, 0),
            1: Direction(0,  1, 0),
            2: Direction(0, -1, 0),
            3: Direction(1,  0, 0),
            4: Direction(0,  0, 1)
        }
        if self.game_over: return self.points, True
        self.remove_lines()
        # print('Steppin da {}th step'.format(self.cstep+1))
        self.stone_dir = actions[action]
        self.cstep += 1
        if action == 3: self.points += points['drop']
        if self.cstep%self.speed == 0: #move one down every x steps
            self.stone_dir.vertical = 1
            if self.nomove and self.check_collision(self.stone, self.stone_pos, actions[3]):
                self.insert_stone()
                return self.points-self.tmppts, False


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
                    print('collision')
                    return True

        return False

    def insert_stone(self):
        print('Insertin da stone')

        o_x = self.stone_pos.x
        o_y = self.stone_pos.y

        for cy, row in enumerate(self.stone):
            for cx, cell in enumerate(row):
                self.state[cy + o_y][cx + o_x] += cell

        self.nomove = False
        self.new_stone()

    def rotate_clockwise(self, shape):

        return [[shape[y][x]
                 for y in range(len(shape))]
                for x in range(len(shape[0]) - 1, -1, -1)]

    def show(self):
        if self.game_over: return self.state
        # print('Showing image')
        o_x = self.stone_pos.x
        o_y = self.stone_pos.y

        state = deepcopy(self.state)

        for cy, row in enumerate(self.stone):
            for cx, cell in enumerate(row):
                state[cy + o_y][cx + o_x] += cell

        # for i in state:
        #     for j in i:
        #         print(j, end=' ') if j > 0 else print('.', end=' ')
        #     print()

        return state

    def new_stone(self):
        print('Puttin a new stone')

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
            for _ in removables:
                self.state = np.insert(self.state, 0, 0, axis=0)


        if len(removables) == 4: self.points += points['tetris']
        else: self.points += len(removables) * points['line']




tetris_shapes = [
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

database = []

def update(*args):
    state = game.show()
    inp = stdscr.getch()
    action = 0
    if inp == ord('w'): action = 4
    elif inp == ord('a'): action = 2
    elif inp == ord('s'): action = 3
    elif inp == ord('d'): action = 1

    reward, done = game.step(action)
    print(reward)
    next_state = game.show()
    database.append(Transition(state, action, next_state, reward))
    if done: ani.event_source.stop(); save_game(database); return
    im.set_array(next_state/7)
    return im,

ani = animation.FuncAnimation(fig, update, interval=50, blit=True)

plt.show()