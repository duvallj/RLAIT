from .Task import Task
from ..util import Move, State, STATE_TYPE_OPTION_TO_INDEX, BadMoveException

from itertools import combinations

import numpy as np

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

class SState(State):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None,
                move_num=0):
        obj = super(SState, subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)

        obj.move_num = move_num

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        super().__array_finalize__(obj)
        self.move_num = getattr(obj, 'move_num', 0)

    # copied from https://stackoverflow.com/a/26599346
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(SState, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.move_num,)
        # Return a tuple that replaces the parent's __reduce__
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # Set the info attributes accordingly
        self.move_num = state[-1]
        # Call the parent's __setstate__ with the original tuple
        super(SState, self).__setstate__(state[0:-1])

class Stratego(Task):

    def __init__(self, size=10, max_moves=1000, **kwargs):
        """
        Initializes Stratego

        Parmeters
        ---------
        size : int
            Size of board to play on

        Notes
        -----
        See https://www.hasbro.com/common/instruct/Stratego.pdf for complete rules.
        """
        kwargs['task_name'] = kwargs.get('task_name', 'stratego')
        kwargs['num_phases'] = 2
        super().__init__(**kwargs)

        self.N = size
        self.max_moves = max_moves

        # initialize empty move/state vectors
        # explanations for sizes found in docstrings
        self._empty_moves = [
            np.ones((self.N, self.N, 14), dtype=bool).view(Move),
            np.ones((self.N, self.N), dtype=bool).view(Move)
        ]

        for i in range(len(self._empty_moves)):
            self._empty_moves[i].task_name = self.task_name
            self._empty_moves[i].phase = i

        self._empty_moves[0].state_type = STATE_TYPE_OPTION_TO_INDEX['deeprect']
        self._empty_moves[1].state_type = STATE_TYPE_OPTION_TO_INDEX['rect']

        self._empty_state = np.zeros((self.N, self.N, 14, 2), dtype=bool).view(SState)
        self._empty_state.task_name = self.task_name
        self._empty_state.state_type = STATE_TYPE_OPTION_TO_INDEX['deeprect']

        self._moveable_mask = np.array([
            1, #(M)arshall
            1, # 9
            1, # 8
            1, # 7
            1, # 6
            1, # 5
            1, # 4
            1, # 3
            1, # 2
            1, #(S)py
            0, #(B)omb
            0, #(F)lag
            0, #(H)azard
            0, # Seen?
        ], dtype=np.uint8)

        self._masked_move = np.ones((14,))
        self._masked_move[-1] = 0

        # place hazards
        for y in range(self.N//2-1, self.N//2+1):
            for x in range(2, self.N, 4):
                self._empty_state[y, x:x+2, 12:, :] = 1

        self._total_pieces_allowed = np.array([
            1, #(M)arshall
            1, # 9
            2, # 8
            3, # 7
            4, # 6
            4, # 5
            4, # 4
            5, # 3
            8, # 2
            1, #(S)py
            6, #(B)omb
            1, #(F)lag
            0, #(H)azard
            0, # Seen?
        ], dtype=np.uint8)

    def empty_move(self, phase=1):
        """
        Gets an empty move vector for sizing purposes.

        Parameters
        ----------
        phase : int, optional
            The phase of task to generate for
            Default: 1, the main play phase

        Returns
        -------
        Move
            A move vector with all fields set to 1

        Notes
        -----
        In the first phase of Stratego, both players place all their pieces.
        So, a move vector will look like:
        ```
        [               # Rows of board
            [           # Columns of row
                [       # Piece on space
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                    Spy, # below are unmovable, included to pad out shape
                    Bomb,
                    Flag,
                    Hazard,
                    Seen?
                ],
                ...
                * N total
            ],
            ...
            * N total
        ]
        ```

        Legally, players may only place pieces on their side of the board, up
        to `(N//2)-1` rows away from their baseboard.

        -----

        In the second phase of Stratego, players move pieces one at a time,
        in one of 4 directions (up, down, left, right). Scouts (2) can move
        any number of spaces, and can attack on the same turn.

        Because a piece can only move into a space not occupied by its
        teammates, a legal move mask will always produce one pair of values that
        represent a move.

        For example, take this 6x6 board (enemy pieces are negative)
        ```
        [
            [-F,-B, 0, 0, 0, 0]
            [-B, 0, 0,-4, 0, 0]
            [ 3, H, 0, 0,-2, 0]
            [ 0, 0, 0, 0, H, 0]
            [ 0, 0, 0, 0, 0, B]
            [ 0, 0, 0, 2, B, F]
        ]
        ```

        Valid moves for the positive person look like:

        ```
        Miner attacks bomb:
        [
            [ 0, 0, 0, 0, 0, 0]
            [ 1, 0, 0, 0, 0, 0]
            [ 1, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
        ]

        Scout attacks 4:
        [
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 1, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 1, 0, 0]
        ]

        Scout attacks 2 (moves next to 4 then attacks):
        [
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 1, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 1, 0, 0]
        ]

        Scout moves sideways:
        [
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 0, 0, 0, 0]
            [ 0, 0, 1, 1, 0, 0]
        ]
        ```

        Flags and bombs cannot move. If a Spy initiates an attack against 10, the Spy wins.
        In all other cases, the spy loses. If any piece except a Miner (3) initiates an attack
        against a bomb, the bomb wins.
        """

        if phase == 0:
            return self._empty_moves[0]
        elif phase == 1:
            return self._empty_moves[1]
        else:
            return None

    def empty_state(self, phase=1):
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

        Notes
        -----
        See `empty_move` for order of pieces. Piece dimension comes before
        player dimension for easier calculations later. Hazards belong to neither
        player, but exist on both players' slots to signify there's something
        there.

        ```
        [                       # Rows on board
            [                   # Columns in row
                [               # Pieces on space
                    [
                        player 0?,
                        player 1?,
                    ],
                    ...
                    * 13
                ],
                ...
                * N total
            ],
            ...
            * N total
        ]
        ```

        That comes out to an (N, N, 13, 2) ndarray for both phases
        """
        cpy = self._empty_state[:]
        cpy.phase = phase
        return cpy

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

        empty = self.empty_move(phase) * 0

        if phase == 0:
            for y in range(self.N):
                for x in range(self.N):
                    for piece in str_to_piece:
                        empty[y, x, str_to_piece[piece]] = 1
                        yield empty.copy()
                        empty[y, x, str_to_piece[piece]] = 0
        elif phase == 1:
            for y1 in range(self.N):
                for x1 in range(self.N):
                    empty[y1, x1] = 1
                    # Instead of looping through all the spots again,
                    # only check what would possibly happen (even with scounts)
                    for y2 in range(self.N):
                        for x2 in range(x1-1, x1+2):
                            if x2 < 0 or x2 >= self.N: continue
                            if x1 == x2 and y1 == y2: continue
                            empty[y2, x2] = 1
                            yield empty.copy()
                            empty[y2, x2] = 0
                    for x2 in range(self.N):
                        for y2 in range(y1-1, y1+2):
                            if y2 < 0 or y2 >= self.N: continue
                            if x1 == x2 and y1 == y2: continue
                            empty[y2, x2] = 1
                            yield empty.copy()
                            empty[y2, x2] = 0
                    empty[y1, x1] = 0
        else:
            yield None

    def iterate_legal_moves(self, state, move=None):
        """
        Iterates over all the legal moves for a given state

        Parameters
        ----------
        state : State

        Yields
        ------
        Move
        """

        outmove = self.empty_move(state.phase) * 0

        legal = self.get_legal_mask(state)
        if move is not None:
            legal = move * legal
        if np.sum(legal) == 0:
            raise BadMoveException("No legal moves found")

        if state.phase == 0:
            # This just gets a list of all indexes where legal is 1, ok
            # for iterating with just one place
            rl, cl, pl = np.unravel_index(np.argsort(legal, axis=None)[::-1], legal.shape)
            for r, c, p in zip(rl, cl, pl):
                if legal[r, c, p]:
                    outmove[r, c, p] = 1
                    yield outmove.copy()
                    outmove[r, c, p] = 0
        elif state.phase == 1:
            # we need to have more complicated stuff for legal moves
            # where it takes two places as a move
            indexes = np.unravel_index(np.argsort(legal, axis=None), legal.shape)
            pair_value = np.vectorize(self._move_value(legal), signature="(i,i)->()")
            pairs = np.asarray(list(combinations(zip(*indexes), 2)))
            values = pair_value(pairs)
            check_mask = values != 0
            pairs = pairs[check_mask][np.argsort(values[check_mask])]
            pairs = list(pairs)[::-1]

            for pair in pairs:
                start = tuple(pair[0])
                end = tuple(pair[1])
                startw, endw = state[start], state[end]
                if not startw[:, state.next_player].any():
                    startw, endw = endw, startw
                    start, end = end, start
                if not startw[:, state.next_player].any() \
                   or endw[:, state.next_player].any():
                    # illegal moves, check the next pair
                    continue

                my = abs(start[0]-end[0])
                mx = abs(start[1]-end[1])
                piece = np.argmax(startw[:, state.next_player])

                if piece == 8: # scouts are special
                    if mx > 1 and my > 1:
                        # illegal motion
                        continue
                    if mx > my:
                        if not endw[:, self._other_player(state.next_player)].any() \
                           and my > 0:
                            continue

                        not_good = False
                        direction = -1 if start[1] > end[1] else 1
                        for x in range(
                           start[1]+direction,
                           end[1] + (my > 0), # go to the end if attacking sideways
                           direction
                           ):
                            if state[start[0], x, :, :].any(): # there is a piece in the way of motion
                                not_good = True
                        if not_good: continue
                    elif my > mx:
                        if not endw[:, self._other_player(state.next_player)].any() \
                           and mx > 0:
                            continue

                        not_good = False
                        direction = -1 if start[0] > end[0] else 1
                        for y in range(
                           start[0]+direction,
                           end[0] + (mx > 0), # go to the end if attacking sideways
                           direction
                           ):
                            if state[y, start[1], :, :].any(): # there is a piece in the way of motion
                                not_good = True
                        if not_good: continue
                    elif mx==1 and my==1 and \
                        not endw[:, self._other_player(state.next_player)].any():
                         # can't move diagonally w/o attacking
                         continue
                else:
                    if mx > 1 or my > 1 \
                       or (mx==1 and my==1):
                        # regular pieces can't move more than once space
                        continue

                # if we've made it this far, that means the move is legal
                outmove[start] = 1
                outmove[end] = 1
#                print("phase 1 yielded", self.move_string_representation(outmove, state),
#                        mx, my, piece_to_str[piece])
#                print("actual move:\n", outmove)
#                print("actual state:\n", self.state_string_respresentation(state))
                yield outmove.copy()
                outmove[start] = 0
                outmove[end] = 0
        else:
            raise TypeError("State phase {} is out of bounds for {}".format(state.phase, self.task_name))

    def _contains_friendly_piece(self, state, r, c):
        return state[r, c, :, state.next_player].any()

    def _other_player(self, player):
        return 1-player

    def _contains_attackable_piece(self, state, r, c):
        return state[r, c, :12, self._other_player(state.next_player)].any()

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

        Notes
        -----
        Like chess, stratego doesn't really have a way to show which piece
        should move into a specified spot, other than to specify the original
        spot to move from as well. This is explained in more detail in `apply_move`
        """

        if state.phase == 0:
            pieces = np.apply_along_axis(
                lambda s: np.argmax(s) if s.any() else -1,
                2,
                state[:, :, :, state.next_player]
            )
            current_counts = dict(zip(*np.unique(pieces, return_counts=True)))
            legal = self.empty_move(0) * 0
            for r in range(self.N):
                for c in range(self.N):
                    for p in range(12):
                        legal[r, c, p] = \
                            current_counts.get(p, 0) < self._total_pieces_allowed[p] and \
                            (
                                state.next_player == 0 and r < self.N//2-1 or \
                                state.next_player == 1 and r > self.N//2 \
                            ) and not self._contains_friendly_piece(state, r, c)

            legal.phase = 0
            legal.task_name = state.task_name
            legal.state_type = self.empty_move(0).state_type

            return legal

        elif state.phase == 1:
            legal = self.empty_move(1)*0
            legal.phase = 1
            legal.next_player = state.next_player
            legal.task_name = state.task_name
            legal.state_type = self.empty_move(1).state_type

            for r in range(self.N):
                for c in range(self.N):
                    if (state[r, c, :, state.next_player] * self._moveable_mask).any():
                        if state[r, c, 8, state.next_player]:
                            # Scout piece, handle accordingly
                            placed = False
                            for i, j in ((0,1),(0,-1),(1,0),(-1,0)):
                                for dist in range(1, self.N):
                                    rc, cc = r+i*dist, c+j*dist
                                    if not (0<=rc<self.N and 0<=cc<self.N): break
                                    if not state[rc, cc].any():
                                        # Nothing there, can move to that spot
                                        legal[rc, cc] = 1
                                        placed = True
                                        for ii, jj in ((0,1),(0,-1),(1,0),(-1,0)):
                                            rcc, ccc = rc+ii, cc+jj
                                            # check if spot is in bounds, can be attacked
                                            if (0<=rcc<self.N and 0<=ccc<self.N) \
                                                and self._contains_attackable_piece(state, rcc, ccc):
                                                # Scouts can move and attack on the same turn
                                                legal[rcc, ccc] = 1
                                    else:
                                        # Hits a blockage, cannot move farther in that direction
                                        break
                            if placed:
                                legal[r, c] = 1
                                pass
                        else:
                            placed = False
                            for i, j in ((0,1),(0,-1),(1,0),(-1,0)):
                                if not (0<=r+i<self.N and 0<=c+j<self.N): continue
                                if not self._contains_friendly_piece(state, r+i, c+j):    # Due to the way hazards are handled, this works
                                    legal[r+i, c+j] = 1
                                    placed = True
                            if placed:
                                legal[r, c] = 1
                                pass
            return legal
        else:
            return None

    def _mask_hidden_pieces(self, piece_vec):
        if piece_vec.any() and not piece_vec[-1]:
            return self._masked_move
        else:
            return piece_vec

    def get_canonical_form(self, state):
        """
        Gets the canonical form of a state, eg how a player sees the board.
        For example, if player 0 and player 1 both "see" the exact same thing
        (their opponents pieces are in the same configuration theirs are),
        this method will return the same output for each player.

        Parameters
        ----------
        state : State

        Returns
        -------
        State
            The original state, assumed to be from player 0's perspective, transformed to
            be from state's `next_player`'s perspective.

        Notes
        -----
        Keeping track of which pieces players remember about the other is done through a
        special "Seen?" flag kept in each piece location. It would be slightly easier to
        have this be stateful to a Stratego instance, but because Tasks are supposed to
        be stateless and only operate on the state data passed in we need to do this.
        """

        cpy = state.copy()

        cpy[:, :, :, self._other_player(state.next_player)] = np.apply_along_axis(
            self._mask_hidden_pieces, 2,
            cpy[:, :, :, self._other_player(state.next_player)]
        )

        # don't reverse order of columns and rows to make move
        # parsing better

        return cpy


    def _move_value(self, legal_moves):
        def _internal_move_value(pair):
            a = legal_moves[tuple(pair[0])]
            b = legal_moves[tuple(pair[1])]
            return \
                (
                    a != 0 and b != 0 \
                ) and ( \
                    a + b \
                )
        return _internal_move_value

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
        For phase 0, this will choose the highest valued number in the passed in
        ove matrix (after being masked by `get_legal_mask`) and place that piece
        there. `get_legal_mask` is assumed to have taken care of all the piece
        counting.

        For phase 1, this expects an array of floats as a move, by which the highest
        pairs (after being masked by `get_legal_mask`) are chosen as
        move candidates before the first legal one is made. The reasoning
        for this is explained below

        Unfortunately, the mask returned by `get_legal_mask` might not
        cover everything. Imagine a board like this:

        ```
        0 0 0 0
        0 0 5 0
        0 3 4 0
        ```

        `get_legal_mask` would return:

        ```
        0 0 1 0
        0 1 1 1
        1 1 1 1
        ```

        Note that this would potentially allow friendly pieces to move into each other,
        which is clearly not allowed. Excluding which piece is making the move, however,
        has the potential to create ambiguity as to which piece should move.

        If a move like such was passed in for the board above:
        ```
        0 0 3 0
        0 8 3 3
        7 9 9 6
        ```

        The 9-9 pair would be classified as illegal, so the 9-8 pair would actually be taken.
        """

        if move.shape != self.empty_move(move.phase).shape:
            raise TypeError("The shape of the move vector {} does not match the empty move vector for phase {}".format(move.shape, state.phase))
            return None
        if  state.shape != self.empty_state(state.phase).shape:
            raise TypeError("The shape of the state vector {} does not match the empty state vector for phase {}".format(state.shape, state.phase))
            return None

        if move.phase != state.phase:
            raise TypeError("The phases of the move ({}) and the state ({}) do not match".format(move.phase, state.phase))
            return None

        nstate = state.copy()
        nstate.move_num += 1

        legal_moves = self.get_legal_mask(state)
        legal = move * legal_moves
        if np.sum(legal) == 0:
            raise BadMoveException("No legal moves found")

        if state.phase == 0:
            spot = np.unravel_index(np.argmax(legal), legal.shape)
            #print(spot)
            nstate[spot][state.next_player] = 1
            nstate.next_player = self._other_player(state.next_player)
            if np.sum(self.get_legal_mask(nstate)) == 0:
                nstate.phase = 1
            return nstate

        elif state.phase == 1:
            # use iterate_legal_moves to get the first pair to check
            found_move = False
            for test_move in self.iterate_legal_moves(state, legal):
                indexes = np.unravel_index(np.argsort(test_move, axis=None)[-2:], test_move.shape)
                # again, whyyyyyy?
                start, end = (indexes[0][0], indexes[1][0]), (indexes[0][1], indexes[1][1])

                startw, endw = state[start], state[end]
                if not startw[:, state.next_player].any():
                    startw, endw = endw, startw
                    start, end = end, start

                piece = np.argmax(startw[:, state.next_player])

                if endw[:, self._other_player(state.next_player)].any():
                    # pieces attack each other
                    attacker = piece
                    defender = np.argmax(endw[:, self._other_player(state.next_player)])

                    # in order:
                    # bomb checking
                    # marshall-spy checking
                    # flag checking
                    # heirarchy checking, except bombs
                    if defender == 10 and attacker == 7 or \
                       defender == 0  and attacker == 9 or \
                       defender == 11 or \
                       attacker < defender and defender != 10:
                        # attacker wins
                        # therefore, attacker moves
                        nstate[start][attacker, state.next_player] = 0
                        nstate[end][attacker, state.next_player] = 1
                        # set Seen? flag
                        nstate[end][-1, state.next_player] = 1
                        # and defender gets erased
                        nstate[end][defender, self._other_player(state.next_player)] = 0
                        # clear Seen? flag
                        nstate[end][-1, self._other_player(state.next_player)] = 0
                        nstate[start][-1, state.next_player] = 0
                    elif attacker == defender:
                        # they both die
                        # attacker first
                        nstate[start][attacker, state.next_player] = 0
                        # clear Seen? flag
                        nstate[start][-1, state.next_player] = 0
                        # also defender
                        nstate[end][defender, self._other_player(state.next_player)] = 0
                        # clear Seen? flag
                        nstate[end][-1, self._other_player(state.next_player)] = 0
                    else:
                        # defender wins
                        # therefore, attacker dies
                        # attacker gets erased
                        nstate[start][attacker, state.next_player] = 0
                        # clear Seen? flag
                        nstate[start][-1, state.next_player] = 0
                        # and defender is Seen
                        nstate[end][-1, self._other_player(state.next_player)] = 1
                else:
                    # piece just moves
                    nstate[start][piece, state.next_player] = 0
                    nstate[end][piece, state.next_player] = 1
                    # copy Seen? flag
                    nstate[end][-1, state.next_player] = nstate[start][-1, state.next_player]
                    nstate[start][-1, state.next_player] = 0
                    nstate.next_player = self._other_player(state.next_player)

                #print(start, "->", end)
                found_move = True
                break

            if not found_move:
                raise BadMoveException("Out of all the pairs analyzed, no legal moves found.")
                return None
            # always return
            nstate.next_player = self._other_player(state.next_player)
            return nstate

        else:
            raise BadMoveException("{} phase {} doesn't exist".format(self.task_name, state.phase))
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

        if state.phase == 0:
            # no one can win while people are still placing pieces
            return False

        if state.move_num >= self.max_moves:
            # don't play past a certain point
            return True

        if np.sum(state[:, :, 11, :]) < 2:  # Someone's flag has been captured
            return True

        if not state[:, :, :10, state.next_player].any() \
           or not state[:, :, :10, self._other_player(state.next_player)].any(): # one player cannot move
            return True

        return False

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
            A set containing the winner. Empty if the game is still going. There are no ties in Stratego.
        """

        if not state[:, :, 11, state.next_player].any() \
           or not state[:, :, :10, state.next_player].any():
            return set((self._other_player(state.next_player),))
        if not state[:, :, 11, self._other_player(state.next_player)].any() \
           or not state[:, :, :10, self._other_player(state.next_player)].any():
            return set((state.next_player,))

        return set()

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
        strout = "   " + "  ".join(map(str, range(self.N))) + "\n"
        for y in range(self.N):
            strout += str(y) + " "
            for x in range(self.N):
                spot = state[y, x]
                if spot[:13, state.next_player].any():
                    v = np.argmax(spot[:13, state.next_player])
                    strout += " " + piece_to_str[v]
                elif spot[:13, self._other_player(state.next_player)].any():
                    if spot[-1, self._other_player(state.next_player)]:
                        strout += "-"
                    else:
                        strout += "?"
                    v = np.argmax(spot[:13, self._other_player(state.next_player)])
                    strout += piece_to_str[v].lower()
                else:
                    if spot[-1, self._other_player(state.next_player)]:
                        strout += " !"
                    else:
                        strout += " ."
                strout += " "
            strout += "\n"
        return strout

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

        if state.phase != move.phase:
            raise TypeError("Move phase {} cannot be applied to State phase {}".format(move.phase, state.phase))

        if state.phase == 0:
            spot = np.unravel_index(np.argmax(move), move.shape)
            return "{0}{1};{2}".format(spot[0], spot[1], piece_to_str[spot[2]])
        elif state.phase == 1:
            pair = np.unravel_index(np.argsort(move, axis=None)[-2:], move.shape)
            # please don't ask why this is a thing I have no idea but I had to do it
            start, end = (pair[0][0], pair[1][0]), (pair[0][1], pair[1][1])
            if not self._contains_friendly_piece(state, start[0], start[1]):
                start, end = end, start
            return "{0}{1},{2}{3}".format(start[0], start[1], end[0], end[1])
        else:
            return None

    def string_to_move(self, move_str, state):
        """
        Returns a move given a string from `move_string_representation`

        Parameters
        ----------
        move_str : str

        Returns
        -------
        Move
        """

        outmove = self.empty_move(state.phase) * 0

        try:
            if state.phase == 0:
                y = int(move_str[0])
                x = int(move_str[1])
                assert move_str[2] == ';'
                p = move_str[3]
                outmove[y, x, str_to_piece[p]] = 1
            elif state.phase == 1:
                y1 = int(move_str[0])
                x1 = int(move_str[1])
                assert move_str[2] == ','
                y2 = int(move_str[3])
                x2 = int(move_str[4])
                outmove[y1, x1] = 1
                outmove[y2, x2] = 1
            else:
                raise TypeError("Phase is out of bounds")
                return None
            return outmove
        except AssertionError:
            raise BadMoveException("Move string \"{}\" is malformed for {}".format(move_str, self.task_name))
            return None
