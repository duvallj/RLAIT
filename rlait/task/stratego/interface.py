from .Stratego import Stratego
from ...approach.random.Random import Random

import numpy as np

from time import sleep

N = 10
task = Stratego(N)
str_to_piece = {
    'M': 0,
    '9': 1,
    '8': 2,
    '7': 3,
    '6': 4,
    '5': 5,
    '4': 6,
    '3': 7,
    '2': 8,
    'S': 9,
    'B': 10,
    'F': 11,
    'H': 12,
}
piece_to_str = dict(((str_to_piece[key], key) for key in str_to_piece))

def visualize_board(state):
    backup_next_player = state.next_player
    # show it from player 1's POV all the time
    state.next_player = 1
    strout = "   " + "  ".join(map(str, range(N))) + "\n"
    for y in range(N):
        strout += str(y) + " "
        for x in range(N):
            spot = state[y, x]
            if spot[:13, state.next_player].any():
                v = np.argmax(spot[:13, state.next_player])
                strout += " " + piece_to_str[v]
            elif spot[:13, task._other_player(state.next_player)].any():
                if True: #spot[-1, task._other_player(state.next_player)]:
                    v = np.argmax(spot[:13, task._other_player(state.next_player)])
                    strout += "-" + piece_to_str[v].lower()
                else:
                    strout += " ?"
            else:
                strout += " ."
            strout += " "
        strout += "\n"
    state.next_player = backup_next_player
    return strout

def parse_move(movestr, state):
    outmove = task.empty_move(state.phase) * 0

    try:
        if state.phase == 0:
            x = int(movestr[0])
            y = int(movestr[1])
            assert movestr[2] == ':'
            p = movestr[3]
            outmove[y, x, str_to_piece[p]] = 1
        elif state.phase == 1:
            x1 = int(movestr[0])
            y1 = int(movestr[1])
            assert movestr[2] == ','
            x2 = int(movestr[3])
            y2 = int(movestr[4])
            outmove[y1, x1] = 1
            outmove[y2, x2] = 1
        else:
            print("Error: state malformed?")
            return None
    except:
        print("Error: move formatted incorrectly. Must be in format 01:2 for first phase, 01,23 for second phase")
        return None

    if not (task.get_legal_moves(state) & outmove).any():
        print("Error: move is illegal. Try again.")
        return None
    else:
        return outmove


def main():
    board = task.empty_state(phase=0)
    ai = Random().init_to_task(task)
    print(visualize_board(board))

    while not task.is_terminal_state(board):
        print("To move: {}".format(board.next_player))
        board = task.apply_move(ai.get_move(board), board)
        print(visualize_board(board))

    print('The winner is player {}'.format(task.get_winners(board)))