import numpy as np

from robby_the_robot.utils import SensorValues, Rewards, sensor_kernels, Actions, CellContents


class Board:
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.width = grid_width + 2
        self.height = grid_height + 2
        self.board = self.get_initial_board()
        self.robby_position = self.get_robby_position()

    def get_initial_board(self):
        board = np.random.randint(2, size=(self.grid_width, self.grid_height), dtype='int8')
        robby_position = np.random.randint(self.grid_height), np.random.randint(self.grid_width)
        board[robby_position] += 3
        return np.pad(array=board, pad_width=1, mode='constant', constant_values=2)

    def get_robby_position(self):
        return np.argwhere(self.board >= 3).reshape(2, )

    def get_cell_content(self, position):
        return self.board[tuple(position)]

    def pick_up_can(self, position):
        self.board[tuple(position)] -= 1

    def move_robby(self, target_position):
        self.board[tuple(self.robby_position)] -= 3
        self.board[tuple(target_position)] += 3
        self.robby_position = target_position

    def get_state(self):
        row, column = self.get_robby_position()
        surroundings = self.board[row - 1:row + 2, column - 1:column + 2]
        return tuple(map(SensorValues, (
            surroundings[1, 1] - 3, surroundings[0, 1], surroundings[2, 1], surroundings[1, 2], surroundings[1, 0])))

    def get_target_cell(self, action):
        kernel = sensor_kernels[action]
        return self.robby_position + kernel

    def perform_action_and_get_reward(self, action, action_tax=0):
        reward = action_tax
        robby_position = self.get_robby_position()
        target_cell = self.get_target_cell(action)
        target_content = self.get_cell_content(target_cell)

        if action == Actions.PICK_UP_CAN:
            if target_content == CellContents.ROBBY_AND_CAN:
                reward += Rewards.PICK_UP_CAN
                self.pick_up_can(robby_position)
            else:  # Cell_Contents.ROBBY
                reward += Rewards.TRY_TO_PICKUP_CAN_IN_EMPTY_CELL
        else:  # action is a move
            if target_content == CellContents.WALL:
                reward += Rewards.CRASH_INTO_WALL
            else:  # target cell is empty or has a can
                self.move_robby(target_cell)

        return reward
