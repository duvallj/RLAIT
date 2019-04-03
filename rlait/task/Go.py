from ..util import State, Move, BadMoveException, STATE_TYPE_OPTION_TO_INDEX
from .Task import Task

import numpy as np

PLAYER_TO_STRING = ['.','@','o']

class GState(State):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None,
                move_num=0, previous=None):
        obj = super(GState, subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)

        obj.move_num = move_num
        obj.previous = previous
        obj.mask = None

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        super().__array_finalize__(obj)
        self.move_num = getattr(obj, 'move_num', 0)
        self.previous = getattr(obj, 'previous', None)
        self.mask = None

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
        self.mask = None
        # Call the parent's __setstate__ with the original tuple
        super(GState, self).__setstate__(state[0:-2])


class Go(Task):
    BLACK = 1
    WHITE = -1
    EMPTY = 0

    def __init__(self, board_size=9, komi=0.5):
        """
        Initializes the Go task.

        Parmeters
        ---------
        board_size : int (9)
            Size of board to play on
        komi : float (0.5)
            Number of points to give to White due to their disadvantage
        """
        super().__init__(task_name="go", num_phases=1)

        self.N = board_size

        # one extra spot to store passing move
        self._empty_move = np.zeros((self.N*self.N + 1), dtype=bool).view(Move)
        self._empty_move.state_type = STATE_TYPE_OPTION_TO_INDEX['flat']

        # one extra row to store prisoner differential
        self._empty_state = np.zeros((self.N+1, self.N), dtype=int).view(GState)
        self._empty_state.task_name = self.task_name
        self._empty_state.state_type = STATE_TYPE_OPTION_TO_INDEX['rect']
        self._empty_state.next_player = self.BLACK
        # include komi as part of initial score
        self._empty_state[self.N, 0] = -int(komi)


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
        for i in range(self.N * self.N + 1):
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

        out = self.empty_move(phase=state.phase)

        for i in range(self.N*self.N + 1):
            if mask[i] == 1:
                out[i] = 1
                yield out.copy()
                out[i] = 0

    def _check_for_suicide(self, state, x, y):
        """
        Checks if move is suicidal.
        """
        if self._count_liberties(state, x, y) == 0:
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
            if p == -1*state.next_player and self._count_liberties(state, x1, y1) == 0:
                score = self._kill_group(state, x1, y1)
                scores.append(score)
        return sum(scores)

    def _get_none(self, state, x, y):
        """
        Returns None if coordinates are not within array dimensions.
        """
        if 0<=y<self.N and 0<=x<self.N:
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
        return filter(lambda i: not (i[0] is None), [
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
            if p == loc and (a, b) not in traversed
        ]

        # Add current coordinates to traversed coordinates
        traversed.add((x, y))

        # Find coordinates of similar neighbors
        if locations:
            return traversed.union(*[
                self._get_group(state, a, b, traversed)
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

        if loc == self.EMPTY:
            # Return coords of empty location (this counts as a liberty)
            return set([(x, y)])
        else:
            # Get surrounding locations which are empty or have the same color
            # and whose coordinates have not already been traversed
            locations = []
            for p, (a, b) in self._get_surrounding(state, x, y):
                if (p == loc or p == self.EMPTY) and (a, b) not in traversed:
                    locations.append((p, (a, b)))

            # Mark current coordinates as having been traversed
            traversed.add((x, y))

            # Collect unique coordinates of surrounding liberties
            if locations:
                return set.union(*[
                    self._get_liberties_recur(state, a, b, traversed)
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
        liberties = self._get_liberties(state, x, y)
        return len(liberties)

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

        mask = self.empty_move(phase=state.phase)
        # passing always valid
        mask[-1] = 1

        for y in range(self.N):
            for x in range(self.N):
                if not (state[y, x] == self.EMPTY): continue

                state[y, x] = state.next_player

                # Check if any pieces have been taken
                taken = self._take_pieces(state.copy(), x, y)

                # Check if move is suicidal.  A suicidal move is a move that takes no
                # pieces and is played on a coordinate which has no liberties.
                if taken == 0:
                    illegal = self._check_for_suicide(state, x, y)
                    if illegal:
                        state[y, x] = self.EMPTY
                        continue

                # Check if move is redundant.  A redundant move is one that would
                # return the board to the state at the time of a player's last move.
                illegal = self._check_for_ko(state)
                if illegal:
                    state[y, x] = self.EMPTY
                    continue

                state[y, x] = self.EMPTY
                mask[x + y*self.N] = 1

        state.mask = mask

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

        Notes
        -----
        If the player passes (last value in the move vector is highest), then
        they give a prisoner to the opponent and the turn flips.
        """

        if state.mask is None:
            state.mask = self.get_legal_mask(state)
        masked_move = move * state.mask
        if np.sum(masked_move) == 0:
            raise BadMoveException("Error: move {} is illegal for board {}".format(move, state))
            return None

        move_loc = np.unravel_index(np.argmax(masked_move), move.shape)[0]
        nstate = state.copy()

        if move_loc == self.N * self.N:
            # player passes, add 1 to opponent score
            nstate[self.N, 0] -= state.next_player
        else:
            # player makes a move
            y, x = move_loc // self.N, move_loc % self.N
            # We already know it is legal (due to earlier mask calculation)
            nstate[y, x] = state.next_player
            nstate[self.N, 0] += state.next_player * self._take_pieces(nstate, x, y)

        nstate.next_player = -1 * state.next_player
        nstate.previous = state

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

        Notes
        -----
        A game of Go only ends when both players willingly or are forced to pass.
        This is kept track of in the state by checking if the current board
        is equal to one two boards previous (impossible with Ko rule, only
        happens with double pass).
        """

        return not (state.previous is None) and \
            not (state.previous.previous is None) and \
            (state == state.previous.previous).all()

    def _get_territory_recur(self, state, x, y, traversed):
        """
        Returns color, traversed

        color value key:
        self.BLACK - empty group is surrounded by black
        self.WHITE - empty group is surrounded by white
        self.EMPTY - empty group has no boundaries, ie still exploring or blank board
        None - mixed white and black borders


        Should always be called starting from an empty piece.
        """
        color = self.EMPTY
        locations = []
        traversed.add((x, y))

        for p, (a, b) in self._get_surrounding(state, x, y):
            if (a, b) not in traversed:
                if p == self.EMPTY:
                    locations.push_back((a, b))
                else:
                    if color == self.EMPTY:
                        color = p
                    else:
                        # there is already a color conflict
                        color = None

        for (a, b) in locations:
            new_color, new_traversed = self._get_territory_recur(state, a, b, traversed)
            if color == self.EMPTY:
                # always update with new if we don't have a color yet
                color = new_color
            elif not (new_color == self.EMPTY or color == new_color):
                # breaking this down:
                # if the old color is something to update with, and there is
                # a conflict, update to have a conflict. Catches previous conflicts too.
                color = None

            traversed = traversed.union(new_traversed)

        return color, traversed

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

        Notes
        -----
        The winners in Go depend on the the differential in territory captured
        combined with the differential in prisoners captured by the other player.
        Basically,
        `black_score = black_territory - black_prisoners`
        `white_score = white_territory - white_prisoners`
        The prisoner differential is kept within the state at localtion (N, 0)
        off the board, `white_prisoners - black_prisoners`. This includes komi
        to begin with. So, we can just add up the differentials and get a score.
        """
        territory_diff = 0

        traversed = set()
        for y in range(self.N):
            for x in range(self.N):
                if (x, y) not in traversed and state[y, x] == self.EMPTY:
                    color, new_traversed = self._get_territory_recur(state, x, y, set())
                    # adds to diff if black, subtracts from diff if white
                    if not (color is None):
                        territory_diff += color * len(new_traversed)

                    traversed = traversed.union(new_traversed)

        if territory_diff + state[self.N, 0] > 0:
            return {self.BLACK}
        elif territory_diff + state[self.N, 0] < 0:
            return {self.WHITE}
        else:
            # are tied, white wins automatically
            return {self.WHITE}

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

        out = str(state.next_player) + "\n"
        for y in range(self.N):
            for x in range(self.N):
                out += PLAYER_TO_STRING[state[y, x]]
            out += "\n"
        out += "pdiff: {}\n".format(state[self.N, 0]-0.5)

        return out


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

        loc = np.unravel_index(np.argmax(move), move.shape)[0]

        if loc == self.N * self.N:
            return 'PASS'
        else:
            return "{},{}".format(loc // self.N, loc % self.N)

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

        move = self.empty_move(phase=phase)

        if move_str == 'PASS':
            move[-1] = 1
        else:
            y, x = move_str.split(',')
            y, x = int(y), int(x)
            move[y*self.N + x] = 1

        return move
