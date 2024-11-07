import sys

from agent_minimax_depth_10 import AgentMinimaxDepth10
from agent_minimax_depth_3_custom_heuristic import AgentMinimaxDepth3CustomHeuristic
from agent_minimax_depth_5 import AgentMinimaxDepth5
from console import Console
from game import Game
from player import Player
from agent_random import AgentRandom
from agent_minimax_depth_3 import AgentMinimaxDepth3
from gui import GUI

if __name__ == "__main__":
    # game = Game(Player("Player1", 1), AgentRandom("Player2", -1))
    # game = Game(Player("Player1", 1), Player("Player2", -1))
    # game = Game(Player("Player1", 1), AgentMinimaxRandom("Player2", -1))
    game = Game(AgentMinimaxDepth3CustomHeuristic("Player1", 1), AgentMinimaxDepth5("Player2", -1))
    gui = GUI(game)
    gui.run()


    # result = {"Player1":0, "Player2":0}
    # time_taken = {"Player1":0, "Player2":0}
    # for i in range(10):
    #     # game =  Game(AgentMinimaxRandom("Player1", 1), AgentRandom("Player2", -1))
    #     # game = Game(AgentRandom("Player1", 1), AgentRandom("Player2", -1))
    #     game = Game(AgentMinimaxDepth3("Player1", 1), AgentMinimaxDepth5("Player2", -1))
    #     console = Console(game)
    #     winner, speed = console.run()
    #     result[winner] = result.get(winner, 0) + 1
    #     time_taken["Player1"] = time_taken.get("Player1", 0) + speed["Player1"]
    #     time_taken["Player2"] = time_taken.get("Player2", 0) + speed["Player2"]
    #     print(f"Game {i} winner: {winner}")
    # print(result)
    # win_ratio = result["Player1"] / (result["Player2"] + result["Player1"])
    # print(f"Win ratio: {win_ratio}")
    # print(f"Speed: {time_taken}")
    sys.exit()