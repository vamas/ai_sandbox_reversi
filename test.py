import unittest

from v1.dqn_helpers import legal_actions_to_binary_mask, action_decode
from v1.dqn_train_new import position_one_hot_encode, action_encode, \
    one_hot_encoding_to_idx
from v1.moveinfo import MoveInfo
from v1.player import Player
from v1.position import Position, SkipPosition


class TestOthello(unittest.TestCase):

    def setUp(self):
        pass

    def test_legal_moves_mask(self):
        legal_moves = [Position(0,0), Position(1, 0), SkipPosition()]
        mask = legal_actions_to_binary_mask(legal_moves)
        print(mask)

    # [1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1]
    # [1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0]
    # [1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1]

    # def test_position_one_hot_encode(self):
    #     self.assertEqual(position_one_hot_encode(Position(0,3), 4), 8)
    #     self.assertEqual(position_one_hot_encode(Position(0, 2), 4), 4)
    #     self.assertEqual(position_one_hot_encode(Position(0, 1), 4), 2)
    #     self.assertEqual(position_one_hot_encode(Position(0, 0), 4), 1)
    #     self.assertEqual(position_one_hot_encode(Position(1, 3), 4), 128)
    #     self.assertEqual(position_one_hot_encode(Position(1, 2), 4), 64)
    #     self.assertEqual(position_one_hot_encode(Position(1, 1), 4), 32)
    #     self.assertEqual(position_one_hot_encode(Position(1, 0), 4), 16)
    #     self.assertEqual(position_one_hot_encode(Position(2, 3), 4), 2048)
    #     self.assertEqual(position_one_hot_encode(Position(2, 2), 4), 1024)
    #     self.assertEqual(position_one_hot_encode(Position(2, 1), 4), 512)
    #     self.assertEqual(position_one_hot_encode(Position(2, 0), 4), 256)
    #     self.assertEqual(position_one_hot_encode(Position(3, 3), 4), 32768)
    #     self.assertEqual(position_one_hot_encode(Position(3, 2), 4), 16384)
    #     self.assertEqual(position_one_hot_encode(Position(3, 1), 4), 8192)
    #     self.assertEqual(position_one_hot_encode(Position(3, 0), 4), 4096)
    #     self.assertEqual(position_one_hot_encode(Position(-1, -1), 4), 65536)

    # def test_position_one_hot_decode(self):
    #     self.assertEqual(position_one_hot_decode(65536, 4), SkipPosition())
    #     self.assertEqual(position_one_hot_decode(1, 4), Position(0,0))
    #     self.assertEqual(position_one_hot_decode(2, 4), Position(0, 1))
    #     self.assertEqual(position_one_hot_decode(4, 4), Position(0, 2))
    #     self.assertEqual(position_one_hot_decode(8, 4), Position(0, 3))
    #     self.assertEqual(position_one_hot_decode(16, 4), Position(1, 0))
    #     self.assertEqual(position_one_hot_decode(32, 4), Position(1, 1))
    #     self.assertEqual(position_one_hot_decode(64, 4), Position(1, 2))
    #     self.assertEqual(position_one_hot_decode(128, 4), Position(1, 3))
    #     self.assertEqual(position_one_hot_decode(256, 4), Position(2, 0))
    #     self.assertEqual(position_one_hot_decode(512, 4), Position(2, 1))
    #     self.assertEqual(position_one_hot_decode(1024, 4), Position(2, 2))
    #     self.assertEqual(position_one_hot_decode(2048, 4), Position(2, 3))
    #     self.assertEqual(position_one_hot_decode(4096, 4), Position(3, 0))
    #     self.assertEqual(position_one_hot_decode(8192, 4), Position(3, 1))
    #     self.assertEqual(position_one_hot_decode(16384, 4), Position(3, 2))
    #     self.assertEqual(position_one_hot_decode(32768, 4), Position(3, 3))


    def test_action_encode(self):
        self.assertEqual(action_encode(SkipPosition()), 16)
        self.assertEqual(action_encode(Position(0,0)), 0)
        self.assertEqual(action_encode(Position(0, 1)), 1)
        self.assertEqual(action_encode(Position(0, 2)), 2)
        self.assertEqual(action_encode(Position(0, 3)), 3)
        self.assertEqual(action_encode(Position(1, 0)), 4)
        self.assertEqual(action_encode(Position(1, 1)), 5)
        self.assertEqual(action_encode(Position(1, 2)), 6)
        self.assertEqual(action_encode(Position(1, 3)), 7)
        self.assertEqual(action_encode(Position(2, 0)), 8)
        self.assertEqual(action_encode(Position(2, 1)), 9)
        self.assertEqual(action_encode(Position(2, 2)), 10)
        self.assertEqual(action_encode(Position(2, 3)), 11)
        self.assertEqual(action_encode(Position(3, 0)), 12)
        self.assertEqual(action_encode(Position(3, 1)), 13)
        self.assertEqual(action_encode(Position(3, 2)), 14)
        self.assertEqual(action_encode(Position(3, 3)), 15)

    def test_action_decode(self):
        self.assertEqual(action_decode(16), SkipPosition())
        self.assertEqual(action_decode(0), Position(0,0))
        self.assertEqual(action_decode(1), Position(0, 1))
        self.assertEqual(action_decode(2), Position(0, 2))
        self.assertEqual(action_decode(3), Position(0, 3))
        self.assertEqual(action_decode(4), Position(1, 0))
        self.assertEqual(action_decode(5), Position(1, 1))
        self.assertEqual(action_decode(6), Position(1, 2))
        self.assertEqual(action_decode(7), Position(1, 3))
        self.assertEqual(action_decode(8), Position(2, 0))
        self.assertEqual(action_decode(9), Position(2, 1))
        self.assertEqual(action_decode(10), Position(2, 2))
        self.assertEqual(action_decode(11), Position(2, 3))
        self.assertEqual(action_decode(12), Position(3, 0))
        self.assertEqual(action_decode(13), Position(3, 1))
        self.assertEqual(action_decode(14), Position(3, 2))
        self.assertEqual(action_decode(15), Position(3, 3))


    # def test_action_one_hot_encode(self):
    #     self.assertEqual(action_one_hot_encode(Position(0, 0)), 1)
    #     self.assertEqual(action_one_hot_encode(Position(0, 1)), 2)
    #     self.assertEqual(action_one_hot_encode(Position(0, 2)), 4)
    #     self.assertEqual(action_one_hot_encode(Position(0, 3)), 8)
    #     self.assertEqual(action_one_hot_encode(Position(1, 0)), 1)
    #     self.assertEqual(action_one_hot_encode(Position(1, 1)), 2)
    #     self.assertEqual(action_one_hot_encode(Position(1, 2)), 4)
    #     self.assertEqual(action_one_hot_encode(Position(1, 3)), 8)

    def test_position_one_hot_to_idx(self):
        self.assertEqual(one_hot_encoding_to_idx(65536), 16)
        self.assertEqual(one_hot_encoding_to_idx(1), 0)
        self.assertEqual(one_hot_encoding_to_idx(2), 1)
        self.assertEqual(one_hot_encoding_to_idx(4), 2)
        self.assertEqual(one_hot_encoding_to_idx(8), 3)
        self.assertEqual(one_hot_encoding_to_idx(16), 4)
        self.assertEqual(one_hot_encoding_to_idx(32), 5)
        self.assertEqual(one_hot_encoding_to_idx(64), 6)
        self.assertEqual(one_hot_encoding_to_idx(128), 7)
        self.assertEqual(one_hot_encoding_to_idx(264), 8)
        self.assertEqual(one_hot_encoding_to_idx(512), 9)
        self.assertEqual(one_hot_encoding_to_idx(1024), 10)
        self.assertEqual(one_hot_encoding_to_idx(2048), 11)
        self.assertEqual(one_hot_encoding_to_idx(4096), 12)
        self.assertEqual(one_hot_encoding_to_idx(8192), 13)
        self.assertEqual(one_hot_encoding_to_idx(16384), 14)
        self.assertEqual(one_hot_encoding_to_idx(32768), 15)
