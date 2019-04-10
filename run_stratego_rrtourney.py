from rlait.task.Stratego import Stratego
from rlait.approach.Random import Random
from rlait.approach.AlphaZero import AlphaZero
from rlait.approach.InteractivePlayer import InteractivePlayer
from rlait.util import dotdict, BadMoveException

import numpy as np
from time import sleep
from itertools import combinations

N = 10
task = Stratego(N)

ai_list = [
    #Random().init_to_task(task),
    AlphaZero(dict()).init_to_task(task),
    InteractivePlayer().init_to_task(task),
]

def run_game(ai1, ai2):
    board = task.empty_state(phase=0)
    print(task.state_string_respresentation(board))

    while not task.is_terminal_state(board):
        # player 1 move sequence
        print("To move: {}".format(board.next_player))
        move = ai1.get_move(board)
        print(task.move_string_representation(move, board))
        board = task.apply_move(move, board)
        to_print = task.state_string_respresentation(board)
        print(to_print)

        if task.is_terminal_state(board): break

        # player 2 move sequence
        print("To move: {}".format(board.next_player))
        move = ai2.get_move(board)
        print(task.move_string_representation(move, board))
        board = task.apply_move(move, board)
        to_print = task.state_string_respresentation(board)
        print(to_print)

    print('The winner is player {}'.format(task.get_winners(board)))

if __name__ == "__main__":
    for ai1, ai2 in combinations(ai_list, 2):
        run_game(ai1, ai2)
