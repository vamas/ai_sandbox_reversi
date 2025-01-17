import unittest

from v1.dqn_train_new import position_one_hot_encode, position_one_hot_decode, action_encode
from v1.moveinfo import MoveInfo
from v1.player import Player
from v1.position import Position


class TestOthello(unittest.TestCase):

    def setUp(self):
        pass

    def test_position_one_hot_encode(self):
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(0,3)), 4), 8)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(0, 2)), 4), 4)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(0, 1)), 4), 2)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(0, 0)), 4), 1)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(1, 3)), 4), 128)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(1, 2)), 4), 64)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(1, 1)), 4), 32)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(1, 0)), 4), 16)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(2, 3)), 4), 2048)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(2, 2)), 4), 1024)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(2, 1)), 4), 512)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(2, 0)), 4), 256)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(3, 3)), 4), 32768)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(3, 2)), 4), 16384)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(3, 1)), 4), 8192)
        self.assertEqual(position_one_hot_encode(MoveInfo(Player.BLACK, Position(3, 0)), 4), 4096)

    def test_position_one_hot_decode(self):
        self.assertEqual(position_one_hot_decode(1, 4), Position(0,0))
        self.assertEqual(position_one_hot_decode(2, 4), Position(0, 1))
        self.assertEqual(position_one_hot_decode(4, 4), Position(0, 2))
        self.assertEqual(position_one_hot_decode(8, 4), Position(0, 3))
        self.assertEqual(position_one_hot_decode(16, 4), Position(1, 0))
        self.assertEqual(position_one_hot_decode(32, 4), Position(1, 1))
        self.assertEqual(position_one_hot_decode(64, 4), Position(1, 2))
        self.assertEqual(position_one_hot_decode(128, 4), Position(1, 3))
        self.assertEqual(position_one_hot_decode(256, 4), Position(2, 0))
        self.assertEqual(position_one_hot_decode(512, 4), Position(2, 1))
        self.assertEqual(position_one_hot_decode(1024, 4), Position(2, 2))
        self.assertEqual(position_one_hot_decode(2048, 4), Position(2, 3))
        self.assertEqual(position_one_hot_decode(4096, 4), Position(3, 0))
        self.assertEqual(position_one_hot_decode(8192, 4), Position(3, 1))
        self.assertEqual(position_one_hot_decode(16384, 4), Position(3, 2))
        self.assertEqual(position_one_hot_decode(32768, 4), Position(3, 3))

    def action_encode(self):
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(0,0))), 0)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(0, 1))), 1)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(0, 2))), 2)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(0, 3))), 3)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(1, 0))), 4)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(1, 1))), 5)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(1, 2))), 6)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(1, 3))), 7)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(2, 0))), 8)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(2, 1))), 9)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(2, 2))), 10)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(2, 3))), 11)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(3, 0))), 12)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(3, 1))), 13)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(3, 2))), 14)
        self.assertEqual(action_encode(MoveInfo(Player.BLACK, Position(3, 3))), 15)


