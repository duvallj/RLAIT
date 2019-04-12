from .Task import Task
from ..util import Move, State, STATE_TYPE_OPTION_TO_INDEX, BadMoveException

import numpy as np

class SimplestTask(Task):
    def __init__(self):
        """
        This is probably the simplest possible task you can get.

        First player: plays 1 or -1
        Second player: has to match what the first player played to win, otherwise they lose.
        Going to cut down on docstrings for this just to save a bit of space.
        """
        super().__init__(task_name="simplest_task", num_phases=1)


    def empty_move(self, phase=0):
        move = np.ones((2,1), dtype=float).view(Move)
        move.state_type = STATE_TYPE_OPTION_TO_INDEX['flat']
        return move

    def empty_state(self, phase=0):
        state = np.zeros((2,1), dtype=int).view(State)
        state.task_name = self.task_name
        state.state_type = STATE_TYPE_OPTION_TO_INDEX['rect']
        state.next_player = 0
        return state

    def iterate_all_moves(self, phase=0):
        out = self.empty_move()

        for x in range(2):
            out[x] = 1
            yield out.copy()
            out[x] = 0


    def iterate_legal_moves(self, state):
        for i in self.iterate_all_moves():
            yield i

    def get_legal_mask(self, state):
        return self.empty_move()

    def get_canonical_form(self, state):
        return state[::([1, -1, 1][state.next_player])]

    def _move_to_number(self, move):
        nmove = -1
        if move[1] > move[0]:
            nmove = 1

        return nmove

    def apply_move(self, move, state):
        nstate = state.copy()

        nmove = self._move_to_number(move)

        if state.next_player == 0:
            nstate[0] = nmove
        elif state.next_player == 1:
            nstate[1] = nmove

        nstate.next_player = state.next_player + 1
        return nstate

    def is_terminal_state(self, state):
        return state.next_player > 1

    def get_winners(self, state):
        if state[0] == state[1]:
            return {1}
        else:
            return {0}

    def state_string_representation(self, state):
        return "({}) a:{} b:{}".format(['first', 'second', 'gameover'][state.next_player], state[0], state[1])

    def move_string_representation(self, move, state):
        return str(self._move_to_number(move))

    def string_to_move(self, move_str, phase=0):
        nmove = int(move_str)
        move = self.empty_move() * 0
        if nmove == 1:
            move[1] = 1
        else:
            move[0] = 1

        return move
