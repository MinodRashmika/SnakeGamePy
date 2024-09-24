import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

from docutils.utils.math.latex2mathml import mover
from numba.np.arrayobj import np_array

from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0 #discount factor
        self.memory = deque(maxlen=MAX_MEMORY) #popleft() when memory is exceeded
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x-20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Danger Straight
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #Danger Right
             (dir_d and game.is_collision(point_l)) or
             (dir_u and game.is_collision(point_r)) or
             (dir_l and game.is_collision(point_u)) or
             (dir_d and game.is_collision(point_d)),

            #Danger Left
             (dir_u and game.is_collision(point_l)) or
             (dir_l and game.is_collision(point_d)) or
             (dir_d and game.is_collision(point_r)) or
             (dir_r and game.is_collision(point_u)),

            #Move Direction
             dir_l,
             dir_r,
             dir_u,
             dir_d,

            #Food_Location
             game.food.x < game.head.x, #left
             game.food.y < game.head.y, #down
             game.food.x > game.head.x, #right
             game.food.y > game.head.y, #up
        ]

        print(np.array(state, dtype=int))
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if MAXMEM is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #return list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_steps, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_steps, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get Old State
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train the long memory (experience replay), plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)

if __name__ == '__main__':
    train()