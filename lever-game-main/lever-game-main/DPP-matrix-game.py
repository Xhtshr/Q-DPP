import matplotlib.pyplot as plt
import numpy as np

from env.StagHunt import StagHunt
from env.Battle_Sex import BattleofSex
from util.logger import Logger
from util.render import Renderer
from agents.qlearning import QLearning
from agents.exploration import LinearDecay

'''
我在这里随便写都能编译
Author Haotian Xu
Date 2022/9/24
This is a prototype program for heterogeneous cross-play(XP) with DPP regularization
'''

population = 30
AgentTypes = 2
total_episodes = 100
# e-greedy hyperparameters
epsilon_start = 1
epsilon_end = 0.01
decayed_by = total_episodes / 2

env = StagHunt()

# 生成一个大种群，许多个体
def init_population(num_actions, population_size):
    players = []
    for _ in range(population_size):
        p = QLearning(1, num_actions)#1个状态 n个动作
        players.append(p)
    return players

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

def make_action(p1, p2, state, episode, strategy):
    action1 = p1.select_action(state, epsilon= strategy.get_epsilon(episode))
    action2 = p2.select_action(state, epsilon=strategy.get_epsilon(episode))
    return (action1, action2)

# 自博弈与评估
log = Logger(population, total_episodes, AgentTypes)
P = init_population(env.num_choices, population)
Q = init_population(env.num_choices, population)
strategy = LinearDecay(epsilon_start, epsilon_end, decayed_by)
lambda_weight = 0.85


l_cards_1, l_cards_2 = [], []
# 计算EXPECTED CARDINALITY
def calculate_EC(l_cards_1, l_cards_2, M):

    L_1 = M[:,:,0] @ M[:,:,0].T
    l_card_1 = np.trace(np.eye(L_1.shape[0]) - np.linalg.inv(L_1 + np.eye(L_1.shape[0])))
    l_cards_1.append(l_card_1)
    # 计算EXPECTED CARDINALITY
    L_2 = M[:,:,1] @ M[:,:,1].T
    l_card_2 = np.trace(np.eye(L_2.shape[0]) - np.linalg.inv(L_2 + np.eye(L_2.shape[0])))
    l_cards_2.append(l_card_2)
    return l_card_1, l_card_2


DPP = False
# population training
for ep in range(total_episodes):
    M = cross_play_testing(P, Q)  # N人策略的元收益
    l_card_1, l_card_2 = calculate_EC(l_cards_1, l_cards_2, M)
    iteration = 0
    k = len(P)
    for p, q in zip(P, Q):
        ind = 1
        population_r = np.array([0, 0])
        state = 0
        action = make_action(p, q, state, ep, strategy)
        _, r, done, info = env.step(action)
        # our environment has only one state but action [0,0],[0,1],[1,0],[1,1].
        new_state = state
        if DPP:
            p.update_q_values(state, action, new_state, lambda_weight * r[0] + (1-lambda_weight) * l_card_1, done)
            q.update_q_values(state, action, new_state, lambda_weight * r[1] + (1-lambda_weight) * l_card_2, done)
            # 重新计算EXPECTED CARDINALITY
            M = cross_play_testing(P[:k-ind], Q[:k-ind])
            l_card_1, l_card_2 = calculate_EC(l_cards_1, l_cards_2, M)
            ind += 1
        else:
            p.update_q_values(state, action, new_state, r[0], done)
            q.update_q_values(state, action, new_state, r[1], done)
        # log training process
        _, r, _, _ = env.step((p.get_optimal_action(state), q.get_optimal_action(state)))
        population_r += np.array(r)
        log.update_training_log(iteration, ep, list(population_r))
        iteration += 1
    M = cross_play_testing(P, Q)  # N人策略的元收益
    log.update_testing_log(ep, M)
    # 计算EXPECTED CARDINALITY
    L_1 = M[:, :, 0] @ M[:, :, 0].T
    l_card_1 = np.trace(np.eye(L_1.shape[0]) - np.linalg.inv(L_1 + np.eye(L_1.shape[0])))
    l_cards_1.append(l_card_1)
    # 计算EXPECTED CARDINALITY
    L_2 = M[:, :, 1] @ M[:, :, 1].T
    l_card_2 = np.trace(np.eye(L_2.shape[0]) - np.linalg.inv(L_2 + np.eye(L_2.shape[0])))
    l_cards_2.append(l_card_2)

# 构造meta game payoff M
log.show_results(smoothness=20)

