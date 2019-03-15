from ..util import State, Move
from .Task import Task
from .go import Board, Array, Location


class GState(State):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None,
                move_num=0, previous=None):
        obj = super(GState, subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)

        obj.move_num = move_num
        obj.previous = previous

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        super().__array_finalize__(obj)
        self.move_num = getattr(obj, 'move_num', 0)
        self.previous = geattr(obj, 'previous', None)

    # copied from https://stackoverflow.com/a/26599346
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(GState, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.move_num, self.previous)
        # Return a tuple that replaces the parent's __reduce__
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # Set the info attributes accordingly
        self.move_num = state[-2]
        self.previous = state[-1]
        # Call the parent's __setstate__ with the original tuple
        super(GState, self).__setstate__(state[0:-2])


class Go(Task):
    def __init__(self, board_size=11):
        """
        Initializes the Go task.

        Parmeters
        ---------
        task_name : str
            Name of task, used for printing
        num_phases : int
            Total number of different phases this task can have
        """
        super().__init__(task_name="othello", num_phases=1)

        self.N = board_size

        self._empty_move = np.ones((self.N*self.N), dtype=bool).view(Move)
        self._empty_move.state_type = STATE_TYPE_OPTION_TO_INDEX['flat']

        self._empty_state = np.zeros((self.N, self.N), dtype=int).view(GState)
        self._empty_state.task_name = self.task_name
        self._empty_state.state_type = STATE_TYPE_OPTION_TO_INDEX['rect']
        self._empty_state.next_player = 1


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

        out = self.empty_move(phase=phase) * 0
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

        out = self.empty_move(phase=phase) * 0

        for i in range(self.N*self.N):
            if mask[i] == 1:
                out[i] = 1
                yield out.copy()
                out[i] = 0

    def move(self, x, y):
        """
        Makes a move at the given location for the current turn's color.
        """
        # Check if coordinates are occupied
        if self[x, y] is not self.EMPTY:
            raise BoardError('Cannot move on top of another piece!')

        # Store history and make move
        self._push_history()
        self[x, y] = self._turn

        # Check if any pieces have been taken
        taken = self._take_pieces(x, y)

        # Check if move is suicidal.  A suicidal move is a move that takes no
        # pieces and is played on a coordinate which has no liberties.
        if taken == 0:
            self._check_for_suicide(x, y)

        # Check if move is redundant.  A redundant move is one that would
        # return the board to the state at the time of a player's last move.
        self._check_for_ko()

        self._flip_turn()
        self._redo = []

    def _check_for_suicide(self, x, y):
        """
        Checks if move is suicidal.
        """
        if self.count_liberties(x, y) == 0:
            self._pop_history()
            raise BoardError('Cannot play on location with no liberties!')

    def _check_for_ko(self):
        """
        Checks if board state is redundant.
        """
        try:
            if self._array == self._history[-2][0]:
                self._pop_history()
                raise BoardError('Cannot make a move that is redundant!')
        except IndexError:
            # Insufficient history...let this one slide
            pass

    def _take_pieces(self, x, y):
        """
        Checks if any pieces were taken by the last move at the specified
        coordinates.  If so, removes them from play and tallies resulting
        points.
        """
        scores = []
        for p, (x1, y1) in self._get_surrounding(x, y):
            # If location is opponent's color and has no liberties, tally it up
            if p is self._next_turn and self.count_liberties(x1, y1) == 0:
                score = self._kill_group(x1, y1)
                scores.append(score)
                self._tally(score)
        return sum(scores)

    def _flip_turn(self):
        """
        Iterates the turn counter.
        """
        self._turn = self._next_turn
        return self._turn

    @property
    def _state(self):
        """
        Returns the game state as a named tuple.
        """
        return self.State(self.copy._array, self._turn, copy(self._score))

    def _load_state(self, state):
        """
        Loads the specified game state.
        """
        self._array, self._turn, self._score = state

    def _push_history(self):
        """
        Pushes game state onto history.
        """
        self._history.append(self._state)

    def _pop_history(self):
        """
        Pops and loads game state from history.
        """
        current_state = self._state
        try:
            self._load_state(self._history.pop())
            return current_state
        except IndexError:
            return None

    def undo(self):
        """
        Undoes one move.
        """
        state = self._pop_history()
        if state:
            self._redo.append(state)
            return state
        else:
            raise BoardError('No moves to undo!')

    def redo(self):
        """
        Re-applies one move that was undone.
        """
        try:
            self._push_history()
            self._load_state(self._redo.pop())
        except IndexError:
            self._pop_history()
            raise BoardError('No undone moves to redo!')

    def _tally(self, score):
        """
        Adds points to the current turn's score.
        """
        self._score[self._turn] += score

    def _get_none(self, x, y):
        """
        Same thing as Array.__getitem__, but returns None if coordinates are
        not within array dimensions.
        """
        try:
            return self[x, y]
        except ArrayError:
            return None

    def _get_surrounding(self, x, y):
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
            (self._get_none(a, b), (a, b))
            for a, b in coords
        ])

    def _get_group(self, x, y, traversed):
        """
        Recursively traverses adjacent locations of the same color to find all
        locations which are members of the same group.
        """
        loc = self[x, y]

        # Get surrounding locations which have the same color and whose
        # coordinates have not already been traversed
        locations = [
            (p, (a, b))
            for p, (a, b) in self._get_surrounding(x, y)
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

    def get_group(self, x, y):
        """
        Gets the coordinates for all locations which are members of the same
        group as the location at the given coordinates.
        """
        if self[x, y] not in self.TURNS:
            raise BoardError('Can only get group for black or white location')

        return self._get_group(x, y, set())

    def _kill_group(self, x, y):
        """
        Kills a group of black or white pieces and returns its size for
        scoring.
        """
        if self[x, y] not in self.TURNS:
            raise BoardError('Can only kill black or white group')

        group = self.get_group(x, y)
        score = len(group)

        for x1, y1 in group:
            self[x1, y1] = self.EMPTY

        return score

    def _get_liberties(self, x, y, traversed):
        """
        Recursively traverses adjacent locations of the same color to find all
        surrounding liberties for the group at the given coordinates.
        """
        loc = self[x, y]

        if loc is self.EMPTY:
            # Return coords of empty location (this counts as a liberty)
            return set([(x, y)])
        else:
            # Get surrounding locations which are empty or have the same color
            # and whose coordinates have not already been traversed
            locations = [
                (p, (a, b))
                for p, (a, b) in self._get_surrounding(x, y)
                if (p is loc or p is self.EMPTY) and (a, b) not in traversed
            ]

            # Mark current coordinates as having been traversed
            traversed.add((x, y))

            # Collect unique coordinates of surrounding liberties
            if locations:
                return set.union(*[
                    self._get_liberties(a, b, traversed)
                    for _, (a, b) in locations
                ])
            else:
                return set()

    def get_liberties(self, x, y):
        """
        Gets the coordinates for liberties surrounding the group at the given
        coordinates.
        """
        return self._get_liberties(x, y, set())

    def count_liberties(self, x, y):
        """
        Gets the number of liberties surrounding the group at the given
        coordinates.
        """
        return len(self.get_liberties(x, y))

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
                i = y*self.N + x


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
