import sys

from console import Console
from game import Game
from player import Player
from agent_random import AgentRandom
from agent_minimax import AgentMinimax
from gui import GUI

if __name__ == "__main__":
    game = Game(AgentMinimax(1, "Player1"), Player(-1, "Player2"))
    # game = Game(Player(1, "Player1"), AgentMinimax(-1, "Player2"))
    # game = Game(Player(1, "Player1"), Player(-1, "Player2"))
    # game = Game(AgentMinimax(1, "Player1"), AgentRandom(-1, "Player2"))
    gui = GUI(game)
    gui.run()


    # result = {}
    # for i in range(10):
    #     game = Game(AgentMinimax(1, "Player1"), AgentRandom(-1, "Player2"))
    #     console = Console(game)
    #     winner = console.run()
    #     result[winner.color_name] = result.get(winner.color_name, 0) + 1
    # print(result)

    sys.exit()