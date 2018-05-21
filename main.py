#!/usr/local/env python
import numpy as np
import pandas as pd

from robby_the_robot.board import Board
from robby_the_robot.q_matrix import QMatrix
from robby_the_robot.utils import print_progress_bar


def execute_training_step(board: Board, q_matrix: QMatrix, action_tax: int):
    current_state = board.get_state()
    action = q_matrix.choose_action(current_state)
    reward = board.perform_action_and_get_reward(action, action_tax)
    new_state = board.get_state()
    q_matrix.update_q_matrix(current_state, action, new_state, reward)
    return reward


def execute_training_episode(q_matrix: QMatrix, grid_width: int, grid_height: int, number_of_steps: int,
                             action_tax: int) -> float:
    board = Board(grid_width, grid_height)
    total_reward = 0
    for step in range(number_of_steps):
        total_reward += execute_training_step(board, q_matrix, action_tax)
    return total_reward


def execute_training_epoch(number_of_episodes, number_of_steps, grid_width, grid_height, epsilon, epsilon_delta,
                           learning_rate, discount_factor, action_tax):
    q_matrix = QMatrix(epsilon, learning_rate, discount_factor)
    best_q_matrix = q_matrix
    rewards = np.zeros((number_of_episodes,))
    print('Training')
    for episode in range(1, number_of_episodes + 1):
        print_progress_bar(episode, number_of_episodes)
        if episode % 50 == 0 and epsilon > 0.1:
            q_matrix.reduce_epsilon(epsilon_delta)
        reward = execute_training_episode(q_matrix, grid_width, grid_height, number_of_steps, action_tax)
        if reward > np.max(rewards):
            best_q_matrix = q_matrix
        rewards[episode - 1] = reward
    print('DONE')
    return best_q_matrix, rewards


def execute_test_step(board: Board, q_matrix: QMatrix, action_tax: int = 0):
    current_state = board.get_state()
    action = q_matrix.choose_action(current_state)
    reward = board.perform_action_and_get_reward(action, action_tax)
    return reward


def execute_test_episode(q_matrix: QMatrix, grid_width: int, grid_height: int, number_of_steps: int,
                         action_tax: int) -> float:
    board = Board(grid_width, grid_height)
    total_reward = 0
    for step in range(number_of_steps):
        total_reward += execute_test_step(board, q_matrix, action_tax)
    return total_reward


def execute_test_epoch(number_of_episodes, number_of_steps, q_matrix, grid_width, grid_height, action_tax):
    rewards = np.zeros((number_of_episodes,))
    print('Testing')
    for episode in range(1, number_of_episodes + 1):
        print_progress_bar(episode, number_of_episodes)
        rewards[episode - 1] = execute_test_episode(q_matrix, grid_width, grid_height, number_of_steps, action_tax)
    print('DONE')
    return rewards


def execute_experiment(number_of_episodes=5000, number_of_steps=200, grid_width=10, grid_height=10, epsilon=1.0,
                       epsilon_delta=0.01, learning_rate=0.2, discount_factor=0.9, action_tax=0.0):
    q_matrix, training_rewards = execute_training_epoch(number_of_episodes, number_of_steps, grid_width, grid_height,
                                                        epsilon, epsilon_delta, learning_rate, discount_factor,
                                                        action_tax)
    test_rewards = execute_test_epoch(number_of_episodes, number_of_steps, q_matrix, grid_width, grid_height,
                                      action_tax)
    return training_rewards, test_rewards, q_matrix


experiments = {
    'Experiment 1': {
        'number of episodes': 5000,
        'number of steps': 200,
        'learning rate': 0.2,
        'discount factor': 0.9,
        'epsilon': 1.0,
        'epsilon delta': 0.01,
        'action tax': 0.0
    },
    'Experiment 2': {
        'number of episodes': 5000,
        'number of steps': 200,
        'learning rate': 0.25,
        'discount factor': 0.9,
        'epsilon': 1.0,
        'epsilon delta': 0.01,
        'action tax': 0.0
    },
    'Experiment 3': {
        'number of episodes': 5000,
        'number of steps': 200,
        'learning rate': 0.5,
        'discount factor': 0.9,
        'epsilon': 1.0,
        'epsilon delta': 0.01,
        'action tax': 0.0
    },
    'Experiment 4': {
        'number of episodes': 5000,
        'number of steps': 200,
        'learning rate': 0.75,
        'discount factor': 0.9,
        'epsilon': 1.0,
        'epsilon delta': 0.01,
        'action tax': 0.0
    },
    'Experiment 5': {
        'number of episodes': 5000,
        'number of steps': 200,
        'learning rate': 1.0,
        'discount factor': 0.9,
        'epsilon': 1.0,
        'epsilon delta': 0.01,
        'action tax': 0.0
    },
    'Experiment 6': {
        'number of episodes': 5000,
        'number of steps': 200,
        'learning rate': 0.2,
        'discount factor': 0.9,
        'epsilon': 0.25,
        'epsilon delta': 0.0,
        'action tax': 0.0
    },
    'Experiment 7': {
        'number of episodes': 5000,
        'number of steps': 200,
        'learning rate': 0.2,
        'discount factor': 0.9,
        'epsilon': 1.0,
        'epsilon delta': 0.01,
        'action tax': -0.5
    }
}

results = {}
for name, experiment in experiments.items():
    training_rewards, test_rewards, q_matrix = execute_experiment(experiment['number of episodes'],
                                                                  experiment['number of steps'], 10, 10,
                                                                  experiment['epsilon'],
                                                                  experiment['epsilon delta'],
                                                                  experiment['learning rate'],
                                                                  experiment['discount factor'],
                                                                  experiment['action tax'])
    pd.DataFrame(training_rewards).to_csv(path_or_buf=f'results/{name}_training_rewards.csv')
    test_rewards = pd.DataFrame(test_rewards).to_csv(path_or_buf=f'results/{name}_test_rewards.csv')
    q_matrix = q_matrix.q_matrix
    q_matrix.to_csv(path_or_buf=f'results/{name}_q_matrix.csv')
