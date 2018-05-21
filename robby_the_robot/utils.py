from enum import IntEnum


class Sensors(IntEnum):
    HERE = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


sensor_kernels = {
    Sensors.HERE: (0, 0),
    Sensors.NORTH: (-1, 0),
    Sensors.SOUTH: (1, 0),
    Sensors.EAST: (0, 1),
    Sensors.WEST: (0, -1)
}


class SensorValues(IntEnum):
    EMPTY = 0
    CAN = 1
    WALL = 2


class Actions(IntEnum):
    PICK_UP_CAN = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_EAST = 3
    MOVE_WEST = 4


class CellContents(IntEnum):
    EMPTY = 0
    CAN = 1
    WALL = 2
    ROBBY = 3
    ROBBY_AND_CAN = 4


class Rewards(IntEnum):
    PICK_UP_CAN = 10
    CRASH_INTO_WALL = -5
    TRY_TO_PICKUP_CAN_IN_EMPTY_CELL = -1
    ACTION_TAX = 0.0


def print_progress_bar(episode, number_of_episodes, num_blocks=50):
    progress = float(episode) / number_of_episodes * 100
    arrow_block = (episode * num_blocks) // number_of_episodes
    left_pad = arrow_block - 1
    right_pad = 50 - arrow_block
    print(f'\rProgress: [{"=" * left_pad}>{"•" * right_pad}] {progress:.2f}% {episode}/{number_of_episodes} ', end='')
