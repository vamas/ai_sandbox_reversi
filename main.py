import sys
from tqdm import tqdm

from v1.manual_agent import ManualAgent
from v1.player import Player
from v1.random_agent import RandomAgent
from v1.minimax_agent import MinimaxAgent
# from v1.gamemanager_console import GameManager
from v1.manual_agent import ManualAgent
from v1.gamemanager_gui import GameManager

games = 10

if __name__ == "__main__":

    # black = MinimaxAgent(Player.BLACK, 3)
    black = ManualAgent(Player.BLACK)
    white = RandomAgent(Player.WHITE)


    # wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}
    #
    # for i in tqdm(range(games), desc="Playing games"):
    #     game_manager = GameManager(black, white)
    #     winner = game_manager.run()
    #     wins[winner] += 1
    #
    # print(f"BLACK wins: ", wins[Player.BLACK])
    # print(f"WHITE wins: ", wins[Player.WHITE])

    game_manager = GameManager(black, white)
    game_manager.run()

    sys.exit()
