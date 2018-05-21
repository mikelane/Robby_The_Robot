from unittest import TestCase
from robby_the_robot.q_matrix import QMatrix
from robby_the_robot.utils import SensorValues


class TestQMatrix(TestCase):
    def test_choose_random_action(self):
        qm = QMatrix(epsilon=1)
        state = (SensorValues.EMPTY, SensorValues.EMPTY, SensorValues.EMPTY, SensorValues.EMPTY, SensorValues.EMPTY)
        action = qm.choose_action(state=state)
        self.assertIn(action, range(5), msg=f'Action was {action} which is outside of {list(range(5))}')

    def test_choose_best_action(self):
        qm = QMatrix(epsilon=0)
        state = tuple(map(SensorValues, (1, 1, 1, 2, 1)))
        action = qm.choose_action(state=state)
        self.assertEqual(action, 0)
