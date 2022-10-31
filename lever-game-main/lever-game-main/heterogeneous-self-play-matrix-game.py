import matplotlib.pyplot as plt
import numpy as np

from env.Battle_Sex import BattleofSex
from env.StagHunt import StagHunt
from util.logger import Logger
from util.render import Renderer
from agents.qlearning import QLearning
from agents.exploration import LinearDecay

# Defining some hyper-parameters which we will use later


total_episodes = 100     # number of episodes we're going to train for
population = 30          # number of agents we're going to train simultaneously
AgentTypes = 2           # number of heterogeneous types for agents
# e-greedy hyperparameters
epsilon_start = 1
epsilon_end = 0.01
decayed_by = total_episodes / 2

# And defining a utility function that initializes a population of players to train
def init_population(num_actions, population_size):
    players = []
    for _ in range(population_size):
        p = QLearning(1, num_actions)#1个状态 n个动作
        players.append(p)
    return players

def make_action(p1, p2, state, episode, strategy):
    action1 = p1.select_action(state, epsilon= strategy.get_epsilon(episode))
    action2 = p2.select_action(state, epsilon=strategy.get_epsilon(episode))
    return (action1, action2)

env = StagHunt()
log = Logger(population, total_episodes, AgentTypes)
P = init_population(env.num_choices, population)
Q = init_population(env.num_choices, population)
strategy = LinearDecay(epsilon_start, epsilon_end, decayed_by)
# training
for ep in range(total_episodes):
    iteration = 0
    for p,q in zip(P, Q):
        state = 0
        action = make_action(p, q, state, ep, strategy)
        _, r, done, info = env.step(action)

        # our environment has only one state but action [0,0],[0,1],[1,0],[1,1].
        new_state = state
        p.update_q_values(state, action, new_state, r[0], done)
        q.update_q_values(state, action, new_state, r[1], done)
        # log training process
        _, r, _, _ = env.step((p.get_optimal_action(state), q.get_optimal_action(state)))
        log.update_training_log(iteration, ep, r)
        iteration += 1
log.show_results(smoothness=20)

# ZSC testing

def cross_play_testing(players1, players2):
    population_size = len(players1)
    reward_matrix = np.zeros((population_size, population_size, 2))


    for i1 in range(population_size):
        for i2 in range(i1):
            p1 = players1[i1]
            p2 = players2[i2]

            state = 0

            action = (p1.get_optimal_action(state), p2.get_optimal_action(state))
            _, r, _, _ = env.step(action)

            reward_matrix[i1, i2, :] = r

    # the reward matrix is symmetric, the full matrix can be calculated as M + M.T
    reward_matrix[:, :, 0] = reward_matrix[:, :, 0] + np.transpose(reward_matrix[:, :, 0])
    reward_matrix[:, :, 1] = reward_matrix[:, :, 1] + np.transpose(reward_matrix[:, :, 1])

    return reward_matrix

env = StagHunt()
log = Logger(population, total_episodes, AgentTypes)
P = init_population(env.num_choices, population)
Q = init_population(env.num_choices, population)
strategy = LinearDecay(epsilon_start, epsilon_end, decayed_by)
for ep in range(total_episodes):
    iteration = 0
    for p,q in zip(P, Q):
        state = 0
        action = make_action(p, q, state, ep, strategy)
        _, r, done, info = env.step(action)

        # our environment has only one state but action [0,0],[0,1],[1,0],[1,1].
        new_state = state
        p.update_q_values(state, action, new_state, r[0], done)
        q.update_q_values(state, action, new_state, r[1], done)
        # log training process
        _, r, _, _ = env.step((p.get_optimal_action(state), q.get_optimal_action(state)))
        log.update_training_log(iteration, ep, r)
        iteration += 1
    # Cross-play Testing
    Reward = cross_play_testing(P, Q)
    log.update_testing_log(ep, Reward)
log.show_results(smoothness=20)
