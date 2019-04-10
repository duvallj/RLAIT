#from rlait.task.Stratego import Stratego
from rlait.task.Othello import Othello
from rlait.approach.Random import Random
from rlait.approach.InteractivePlayer import InteractivePlayer
#from rlait.approach.AlphaZero import AlphaZero
from rlait.util import dotdict, BadMoveException

import numpy as np

from time import sleep

def main(task, ai1, ai2):
    board = task.empty_state(phase=0)
    print(task.state_string_representation(board))

    first_player_number = board.next_player

    while not task.is_terminal_state(board):
        move = None
        if board.next_player == first_player_number:
            print("To move: {} ({})".format(board.next_player, ai1.approach_name))
            move = ai1.get_move(board)
        else:
            print("To move: {} ({})".format(board.next_player, ai2.approach_name))
            move = ai2.get_move(board)
            
        print(task.move_string_representation(move, board))
        board = task.apply_move(move, board)
        to_print = task.state_string_representation(board)
        print(to_print)

    print('The winner is player {}'.format(task.get_winners(board)))

def play_noprint(task, ai1, ai2):
    board = task.empty_state(phase=0)

    first_player_number = board.next_player

    while not task.is_terminal_state(board):
        move = None
        if board.next_player == first_player_number:
            move = ai1.get_move(board)
        else:
            move = ai2.get_move(board)
        board = task.apply_move(move, board)

    winners = task.get_winners(board)
    if first_player_number in winners: return 1
    elif len(winners) != 0: return -1
    return 0

if __name__ == "__main__":
    N = 4 #10
    task = Othello(N) #Stratego(N)
    ai1 = Random()
    ai2 = InteractivePlayer()
    main(task, ai1, ai2)
