from ..util import State, Move
from .Task import Task
from .go import Board, Array, Location


class GState(State):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None,
                move_num=0, previous=None, score=0):
        obj = super(GState, subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)

        obj.move_num = move_num
        obj.previous = previous
        obj.score = score

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        super().__array_finalize__(obj)
        self.move_num = getattr(obj, 'move_num', 0)
        self.previous = geattr(obj, 'previous', None)
        self.score = getattr(obj, 'score', 0)

    # copied from https://stackoverflow.com/a/26599346
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(GState, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.move_num, self.previous, self.score)
        # Return a tuple that replaces the parent's __reduce__
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # Set the info attributes accordingly
        self.move_num = state[-3]
        self.previous = state[-2]
        self.score = state[-1]
        # Call the parent's __setstate__ with the original tuple
        super(GState, self).__setstate__(state[0:-3])


class Go(Task):
    BLACK = 1
    WHITE = -1
    EMPTY = 0

    def __init__(self, board_size=9, pass_thresh=0.15):
        """
        Initializes the Go task.

        Parmeters
        ---------
        board_size : int (9)
            Size of board to play on
        pass_thresh : float (0.15)
            If every value in a move vector is below this,
            assume the AI wants to pass.
        """
        super().__init__(task_name="go", num_phases=1)

        self.N = board_size

        self._empty_move = np.zeros((self.N*self.N), dtype=bool).view(Move)
        self._empty_move.state_type = STATE_TYPE_OPTION_TO_INDEX['flat']

        self._empty_state = np.zeros((self.N, self.N), dtype=int).view(GState)
        self._empty_state.task_name = self.task_name
        self._empty_state.state_type = STATE_TYPE_OPTION_TO_INDEX['rect']
        self._empty_state.next_player = self.BLACK


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
            A move vector with all fields set to 0
        """

        return self._empty_move.copy()

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

        return self._empty_state.copy()

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

        out = self.empty_move(phase=phase)
        # include passing move
        yield out.copy()
        for i in range(self.N * self.N):
            out[i] = 1
            yield out.copy()
            out[i] = 0


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

        out = self.empty_move(phase=phase)
        # include passing move
        yield out.copy()

        for i in range(self.N*self.N):
            if mask[i] == 1:
                out[i] = 1
                yield out.copy()
                out[i] = 0

    def _check_for_suicide(self, state, x, y):
        """
        Checks if move is suicidal.
        """
        if self.count_liberties(state, x, y) == 0:
            return True
        return False

    def _check_for_ko(self, state):
        """
        Checks if board state is redundant.
        """
        if not (state.previous is None) and \
           not (state.previous.previous is None) and \
           (state.previous.previous == state).all():
            return True
        return False

    def _take_pieces(self, state, x, y):
        """
        Checks if any pieces were taken by the last move at the specified
        coordinates.  If so, removes them from play and tallies resulting
        points.
        """
        scores = []
        for p, (x1, y1) in self._get_surrounding(state, x, y):
            # If location is opponent's color and has no liberties, tally it up
            if p is state.next_player and self.count_liberties(state, x1, y1) == 0:
                score = self._kill_group(state, x1, y1)
                scores.append(score)
        return sum(scores)

    def _get_none(self, state, x, y):
        """
        Returns None if coordinates are not within array dimensions.
        """
        if 0<y<state.shape[0] and 0<x<state.shape[1]:
            return state[y, x]
        else:
            return None

    def _get_surrounding(self, state, x, y):
        """
        Gets information about the surrounding locations for a specified
        coordinate.  Returns a tuple of the locations clockwise starting from
        the top.
        """
        coords = (
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        )
        return filter(lambda i: bool(i[0]), [
            (self._get_none(state, a, b), (a, b))
            for a, b in coords
        ])

    def _get_group(self, state, x, y, traversed):
        """
        Recursively traverses adjacent locations of the same color to find all
        locations which are members of the same group.
        """
        loc = state[y, x]

        # Get surrounding locations which have the same color and whose
        # coordinates have not already been traversed
        locations = [
            (p, (a, b))
            for p, (a, b) in self._get_surrounding(state, x, y)
            if p is loc and (a, b) not in traversed
        ]

        # Add current coordinates to traversed coordinates
        traversed.add((x, y))

        # Find coordinates of similar neighbors
        if locations:
            return traversed.union(*[
                self._get_group(a, b, traversed)
                for _, (a, b) in locations
            ])
        else:
            return traversed

    def get_group(self, state, x, y):
        """
        Gets the coordinates for all locations which are members of the same
        group as the location at the given coordinates.
        """

        return self._get_group(state, x, y, set())

    def _kill_group(self, state, x, y):
        """
        Kills a group of black or white pieces and returns its size for
        scoring.
        """

        group = self.get_group(state, x, y)
        score = len(group)

        for x1, y1 in group:
            state[y1, x1] = self.EMPTY

        return score

    def _get_liberties_recur(self, state, x, y, traversed):
        """
        Recursively traverses adjacent locations of the same color to find all
        surrounding liberties for the group at the given coordinates.
        """
        loc = state[y, x]

        if loc is self.EMPTY:
            # Return coords of empty location (this counts as a liberty)
            return set([(x, y)])
        else:
            # Get surrounding locations which are empty or have the same color
            # and whose coordinates have not already been traversed
            locations = [
                (p, (a, b))
                for p, (a, b) in self._get_surrounding(state, x, y)
                if (p is loc or p is self.EMPTY) and (a, b) not in traversed
            ]

            # Mark current coordinates as having been traversed
            traversed.add((x, y))

            # Collect unique coordinates of surrounding liberties
            if locations:
                return set.union(*[
                    self._get_liberties(state, a, b, traversed)
                    for _, (a, b) in locations
                ])
            else:
                return set()

    def _get_liberties(self, state, x, y):
        """
        Gets the coordinates for liberties surrounding the group at the given
        coordinates.
        """
        return self._get_liberties_recur(state, x, y, set())

    def _count_liberties(self, state, x, y):
        """
        Gets the number of liberties surrounding the group at the given
        coordinates.
        """
        return len(self._get_liberties(state, x, y))

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

        mask = self.empty_move(phase=phase)

        for y in range(self.N):
            for x in range(self.N):
                if state[y, x] is self.EMPTY:
                    continue

                # Check if any pieces have been taken
                taken = self._take_pieces(state, x, y)

                # Check if move is suicidal.  A suicidal move is a move that takes no
                # pieces and is played on a coordinate which has no liberties.
                if taken == 0:
                    legal = self._check_for_suicide(state, x, y)
                    if not legal: continue

                # Check if move is redundant.  A redundant move is one that would
                # return the board to the state at the time of a player's last move.
                legal = self._check_for_ko(state)
                if not legal: continue

                mask[x + y*self.N] = 1

        return mask


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
        nstate = state.copy()

        if state.next_player == -1:
            # update all the history as well
            nstate = state * -1
            nstate.next_player = 1
            ustate = nstate.previous
            pstate = nstate
            while not (ustate is None):
                ustate = ustate * -1
                pstate.previous = ustate
                ustate, pstate = ustate.previous, ustate
        else:
            pass

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

        # TODO: How do I check for when they want to pass?
        return None

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

        return None

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
            A set containing all the winners. Empty if no winners. Ties depend on game implementation.
        """

        return None

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

        return None

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

        return None

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

        return None
