import itertools

import numpy as np
import pandas as pd

from robby_the_robot.utils import Actions, SensorValues, Sensors


class QMatrix:
    def __init__(self, epsilon=1, learning_rate=0.1, discount_factor=0.9):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_matrix = pd.DataFrame(
            {key: [0.0] * len(Actions) for key in itertools.product(SensorValues, repeat=len(Sensors))},
            index=Actions).T

    def reduce_epsilon(self, amount=0.01):
        self.epsilon = max(0.1, self.epsilon - amount)

    def choose_action(self, state):
        if np.random.sample() > self.epsilon:
            return Actions(np.argmax(self.q_matrix.loc[state].values))
        else:
            return Actions(np.random.randint(len(Actions)))

    def get_max_q_value(self, state):
        return np.max(self.q_matrix.loc[state].values)

    def get_q_value(self, state, action):
        return self.q_matrix.loc[state, action]

    def set_q_value(self, state, action, value):
        self.q_matrix.loc[state, action] = value

    def update_q_matrix(self, old_state, action, new_state, reward):
        old_q_value = self.get_q_value(old_state, action)
        value = old_q_value + self.learning_rate * (
                reward + self.discount_factor * self.get_max_q_value(new_state) - old_q_value)
        self.set_q_value(old_state, action, value)
