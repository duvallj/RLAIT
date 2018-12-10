from .Stratego import Stratego
from ...approach.random.Random import Random
from ...util import BadMoveException

import numpy as np

from time import sleep

N = 10
task = Stratego(N)

def parse_move(movestr, state):
    outmove = task.empty_move(state.phase) * 0

    if True: #try:
        if state.phase == 0:
            x = int(movestr[0])
            y = int(movestr[1])
            assert movestr[2] == ';'
            p = movestr[3]
            outmove[y, x, task.str_to_piece[p]] = 1
        elif state.phase == 1:
            y1 = int(movestr[0])
            x1 = int(movestr[1])
            assert movestr[2] == ','
            y2 = int(movestr[3])
            x2 = int(movestr[4])
            outmove[y1, x1] = 1
            outmove[y2, x2] = 1
        else:
            print("Error: state malformed?")
            return None
    else: #except:
        print("Error: move formatted incorrectly. Must be in format 01;2 for first phase, 01,23 for second phase")
        return None

    if not (outmove * task.get_legal_moves(state)).any():
        print("Error: move is illegal due to legal_moves mask. Try again.")
        return None
    else:
        return outmove


def main():
    board = task.empty_state(phase=0)
    ai = Random().init_to_task(task)
    print(task.string_respresentation(board))

    while not task.is_terminal_state(board):
        print("To move: {}".format(board.next_player))
        board = task.apply_move(ai.get_move(board), board)
        print(task.string_respresentation(board))
        print(task.string_respresentation(task.get_canonical_form(board)))
        player_move = None
        new_board = None
        while player_move is None:
            player_move = parse_move(input("Enter move: "), board)
            if player_move is not None:
                try:
                    new_board = task.apply_move(player_move, board)
                except BadMoveException:
                    print("Error: move is illegal due to apply_move rules. Try again.")
                    player_move = None
        board = new_board
        print(task.string_respresentation(board))
        print(task.string_respresentation(task.get_canonical_form(board)))

    print('The winner is player {}'.format(task.get_winners(board)))
