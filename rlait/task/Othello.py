from .Task import Task
from ..util import Move, State, STATE_TYPE_OPTION_TO_INDEX, BadMoveException

import numpy as np

DIRECTIONS = (
    (1,0),
    (1,1),
    (0,1),
    (-1,1),
    (-1,0),
    (-1,-1),
    (0,-1),
    (1,-1),
)

class Othello(Task):
    def __init__(self, board_size=8):
        """
        Initializes a task

        Parmeters
        ---------
        board_size : int (default 8)
            Size of board to play on
        """
        super().__init__(task_name="othello", num_phases=1)

        self.N = board_size

        self._empty_move = np.ones((self.N*self.N), dtype=bool).view(Move)
        self._empty_move.state_type = STATE_TYPE_OPTION_TO_INDEX['flat']

        self._empty_state = np.zeros((self.N, self.N), dtype=int).view(State)
        self._empty_state.task_name = self.task_name
        self._empty_state.state_type = STATE_TYPE_OPTION_TO_INDEX['rect']
        self._empty_state.next_player = 1

        c1 = self.N//2-1
        c2 = self.N//2
        self._empty_state[c1,c1] = 1
        self._empty_state[c2,c2] = 1
        self._empty_state[c1,c2] = -1
        self._empty_state[c2,c1] = -1


    def empty_move(self, phase=0):
        """
        Gets an empty move vector for sizing purposes.

        Parameters
        ----------
        phase : int, optional
            The phase of task to generate for

        Returns
        -------
        Move
            A move vector with all fields set to 1
        """

        return self._empty_move

    def empty_state(self, phase=0):
        """
        Gets an empty state vector for sizing purposes.

        Parameters
        ----------
        phase : int, optional
            The phase of task to generate for

        Returns
        -------
        State
            A state vector with no players present
        """

        return self._empty_state

    def iterate_all_moves(self, phase=0):
        """
        Iterates over all possible moves, even illegal ones, for a given phase.

        Parameters
        ----------
        phase : int, optional
            The phase to generate all the moves for

        Yields
        ------
        Move
            An expected move for that state
        """

        for i in range(self.N*self.N):
            yield i


    def iterate_legal_moves(self, state):
        """
        Iterates over all the legal moves for a given state

        Parameters
        ----------
        state : State

        Yields
        ------
        Move
        """

        mask = self.get_legal_mask(state)

        out = self.empty_move() * 0

        for i in range(self.N*self.N):
            if mask[i] == 1:
                out[i] = 1
                yield out.copy()
                out[i] = 0

    def _out_of_bounds(self, py, px):
        return 0>py or 0>px or self.N<=py or self.N<=px

    def _opponent(self, player):
        return -1*player

    def _find_bracket(self, board, square, player, direction):
        """
        Find a square that forms a bracket with `square` for `player` in the given
        `direction`.  Returns None if no such square exists.
        """
        dy, dx = direction
        sy, sx = square
        bry, brx = sy+dy, sx+dx

        if self._out_of_bounds(bry, brx):
            return None
        if board[bry, brx] == player:
            return None

        opp = self._opponent(player)
        while board[bry, brx] == opp:
            bry += dy
            brx += dx
            if self._out_of_bounds(bry, brx):
                return None

        return None if board[bry, brx] == 0 else (bry, brx)

    def _is_legal(self, board, move, player):
        r = not self._out_of_bounds(*move) and board[move] == 0 and \
            any(self._find_bracket(board, move, player, direction) for direction in DIRECTIONS)
        return r

    def get_legal_mask(self, state):
        """
        Gets a move vector mask for all the legal moves for a state

        Parameters
        ----------
        state : State

        Returns
        -------
        Move
            A move vector with 1s in the place where the state's `next_player` can go
        """

        out = self.empty_move() * 0

        for y in range(self.N):
            for x in range(self.N):
                if self._is_legal(state, (y,x), state.next_player):
                    out[x+y*self.N] = 1

        return out


    def get_canonical_form(self, state):
        """
        Gets the canonical form of a state, normalized for how the current
        player would "see" to board if they were the first player.

        Parameters
        ----------
        state : State

        Returns
        -------
        State
            The original state, assumed to be from player 0's perspective, transformed to
            be from state's `next_player`'s perspective.
        """

        # players are stored as 1 and -1, so this works
        nstate = state * state.next_player
        nstate.next_player = 1
        return nstate


    def apply_move(self, move, state):
        """
        Applies a move to a state, returning an updated state

        Parameters
        ----------
        move : Move
            Move to make
        state : State
            State to update

        Returns
        -------
        State
            Updated state

        Raises
        ------
        BadMoveException
            If the move is not legal for the state
        TypeError
            If the phase of the move and state mismatch
        """

        if move.shape != self.empty_move(move.phase).shape:
            raise TypeError("The shape of the move vector {} does not match the empty move vector for phase {}".format(move.shape, state.phase))
            return None
        if  state.shape != self.empty_state(state.phase).shape:
            raise TypeError("The shape of the state vector {} does not match the empty state vector for phase {}".format(state.shape, state.phase))
            return None

        legal_moves = self.get_legal_mask(state)
        legal = move * legal_moves

        if np.sum(legal) == 0:
            raise BadMoveException("No legal moves found")

        nstate = state.copy()

        spot = np.unravel_index(np.argmax(legal), nstate.shape)
        nstate[spot] = state.next_player

        for d in DIRECTIONS:
            bracket = self._find_bracket(state, spot, state.next_player, d)
            if bracket is None: continue
            dy, dx = d
            bry, brx = bracket
            py, px = spot

            while py != bry or px != brx:
                py += dy
                px += dx
                nstate[py, px] = state.next_player

        nstate.next_player = self._opponent(state.next_player)

        # forcing a pass
        if np.sum(self.get_legal_mask(nstate)) == 0:
            nstate.next_player = state.next_player

        return nstate


    def is_terminal_state(self, state):
        """
        Checks if a state is terminal, ie the game is over

        Parameters
        ----------
        state : State
            State to check

        Returns
        -------
        bool
            True if terminal, False if not
        """

        player_moves = np.sum(self.get_legal_mask(state))
        if player_moves > 0: return False

        state.next_player = self._opponent(state.next_player)
        opponent_moves = np.sum(self.get_legal_mask(state))
        state.next_player = self._opponent(state.next_player)
        if opponent_moves > 0: return False

        return True


    def get_winners(self, state):
        """
        Gets all the winners of a (supposedly terminal) state. Supports ties.

        Parameters
        ----------
        state : State
            State to check for winners

        Returns
        -------
        set
            A set containing all the winners. Empty if no winners, also in case of tie
        """

        player1_score = np.sum(state == 1)
        player2_score = np.sum(state == -1)

        winners = set()

        if player1_score > player2_score:
            winners.add(1)
        if player2_score > player1_score:
            winners.add(-1)

        return winners

    def state_string_representation(self, state):
        """
        Returns a string representation of a board, fit for printing and/or caching

        Parameters
        ----------
        state : State

        Returns
        -------
        str
        """
        mapping = {-1:'o', 1:'@', 0:'.'}

        st = str(state.next_player)+'\n'
        for y in range(self.N):
            for x in range(self.N):
                st += mapping[state[y, x]]
            st += '\n'

        return st

    def move_string_representation(self, move, state):
        """
        Returns a string representation of a move, fit for printing and/or caching

        Parameters
        ----------
        move : Move

        Returns
        -------
        str
        """

        return str(np.argmax(move))

    def string_to_move(self, move_str, phase=0):
        """
        Returns a move given a string from `move_string_representation`

        Parameters
        ----------
        move_str : str
        phase : int, optional

        Returns
        -------
        Move
        """

        nmove = self.empty_move() * 0
        nmove[int(move_str)] = 1

        return nmove
