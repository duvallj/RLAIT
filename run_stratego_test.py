from rlait.task.stratego.Stratego import Stratego
from rlait.approach.random.Random import Random
from rlait.approach.alphazero.AlphaZero import AlphaZero
from rlait.util import dotdict, BadMoveException

import numpy as np

from time import sleep

N = 10
task = Stratego(N)

def main():
    board = task.empty_state(phase=0)
    ai1 = Random().init_to_task(task)
    ai2 = AlphaZero(dotdict()).init_to_task(task)
    print(task.state_string_respresentation(board))

    while not task.is_terminal_state(board):
        print("To move: {}".format(board.next_player))
        board = task.apply_move(ai1.get_move(board), board)
        print(task.state_string_respresentation(board))
        print(task.state_string_respresentation(task.get_canonical_form(board)))
        """
        player_move = None
        new_board = None
        while player_move is None:
            player_move = task.string_to_move(input("Enter move: "), board)
            if player_move is not None:
                try:
                    new_board = task.apply_move(player_move, board)
                except BadMoveException:
                    print("Error: move is illegal due to apply_move rules. Try again.")
                    player_move = None
        board = new_board
        """
        board = task.apply_move(ai2.get_move(board), board)
        #"""
        print(task.state_string_respresentation(board))
        print(task.state_string_respresentation(task.get_canonical_form(board)))

    print('The winner is player {}'.format(task.get_winners(board)))

if __name__ == "__main__":
    main()
