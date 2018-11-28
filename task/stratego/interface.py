from .Stratego import Stratego

import numpy as np

task = Stratego()
N = 10
piece_to_str = {
    0: 'M',
    1: '9',
    2: '8',
    3: '7',
    4: '6',
    5: '5',
    6: '4',
    7: '3',
    8: '2',
    9: 'S',
   10: 'B',
   11: 'F',
   12: 'H',
}

def visualize_board(state):
    # only shows it from state.next_player's POV
    strout = ""
    for y in range(N):
        for x in range(N):
            spot = state[y, x]
            if spot[:13, state.next_player].any():
                v = np.argmax(spot[:13, state.next_player])
                strout += " " + piece_to_str[v]
            elif spot[:13, task._other_player(state.next_player)].any():
                
                if spot[-1, task._other_player(state.next_player)]:
                v = np.argmax(spot[:13, task._other_player(state.next_player)])
                    strout += "-" + piece_to_str[v].lower()
                else:
                    strout += " ?"
            else:
                strout += " ."
            strout += " "
        strout += "\n"
    return strout
            
def main():
    pass
    
if __name__ == "__main__":
    main()