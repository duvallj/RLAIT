from ..task import Task
from ..util import Move, State

from itertools import combinations

import numpy as np

class Stratego(Task):
    task_name = "stratego"
    def __init__(self, size=10, **kwargs):
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
        super().__init__(self, **kwargs)
        
        self.N = size
        
        # initialize empty move/state vectors
        # explanations for sizes found in docstrings
        self._empty_moves = [
            np.ones((self.N, self.N, 14), dtype=bool).view(Move),
            np.ones((self.N, self.N), dtype=bool).view(Move)
        ]
        
        for i in range(len(self._empty_moves)):
            self._empty_moves[i].task_name = self.task_name
            self._empty_moves[i].phase = i
        
        self._empty_moves[0].state_type = STATE_TYPE_OPTIONS_TO_INDEX['deeprect']
        self._empty_moves[1].state_type = STATE_TYPE_OPTIONS_TO_INDEX['rect']
        
        self._empty_state = np.zeros((self.N, self.N, 13, 2), dtype=bool).view(State)
        self._empty_state.task_name = self.task_name
        self._empty_state.state_type = STATE_TYPE_OPTIONS_TO_INDEX['deeprect']
        
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
        
        self._masked_move = np.ones((13,))
        self._masked_move[-1] = 0
        
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
       
    def _contains_friendly_piece(self, state, r, c):
        return state[r, c, :, state.next_player].any()
     
    def _other_player(self, player):
        return 1-player
     
    def _contains_attackable_piece(self, state, r, c):
        return state[r, c, :12, self._other_player(state.next_player)].any()
        
    def get_legal_moves(self, state):
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
        Like chess, stratego doesn't really have 
        """
        
        if state.phase == 0:
            # TODO: count pieces, make sure you can't
            # have an army of Marshals
            current_counts = dict(zip(*np.unique(state[:, :, state.next_player])))
            legal = np.fromfunction(
                lambda r, c, p:
                    current_counts[p] < self._total_pieces_allowed[p] and \
                    (
                        state.next_player == 0 and r < self.N//2-1 or \
                        state.next_player == 1 and r > self.N//2 \
                    ) and not self._contains_friendly_piece(state, r, c) \
                ,
                self.empty_move(0).shape,
                dtype=bool
            ).view(Move)
            
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
                    if (state[r, c, state.next_player] | self._moveable_mask).any():
                        if state[r, c, 8, state.next_player]:
                            # Scout piece, handle accordingly
                            placed = False
                            for i, j in ((0,1),(0,-1),(1,0),(-1,0)):
                                for dist in range(self.N):
                                    rc, cc = r+i*dist, c+j*dist
                                    if not (0<=rc<self.N and 0<=cc<self.N): break
                                    if not state[rc, cc].any():
                                        # Nothing there, can move to that spot
                                        legal[rc, cc] = 1
                                        placed = True
                                        for ii, jj in ((0,1),(0,-1),(1,0),(-1,0)):
                                            rcc, ccc = rc+ii, cc+jj
                                            if (0<=rcc<self.N and 0<=ccc<self.N) \ # in bounds
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
        if not piece_vec[-1]:
            return self._masked_move
    
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
        
        cpy = state[:]
        
        cpy[:, :, :, self._other_player(state.next_player)] = np.apply_along_axis(
            self._mask_hidden_pieces, 2, 
            cpy[:, :, :, self._other_player(state.next_player)]
        )
        
        if cpy.next_player == 1:
            # reverse order of columns, rows, and player vectors
            cpy = cpy[::-1, ::-1, ::-1]
        
        return cpy
        
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
        move matrix (after being masked by `get_legal_moves`) and place that piece
        there. `get_legal_moves` is assumed to have taken care of all the piece
        counting.
        
        For phase 1, this expects an array of floats as a move, by which the highest
        pairs (after being masked by `get_legal_moves`) are chosen as
        move candidates before the first legal one is made. The reasoning
        for this is explained below
        
        Unfortunately, the mask returned by `get_legal_moves` might not
        cover everything. Imagine a board like this:
        
        ```
        0 0 0 0
        0 0 5 0
        0 3 4 0
        ```
        
        `get_legal_moves` would return:
        
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
        
        if move.shape != self.empty_move(state.phase).shape:
            raise TypeError("The shape of the move vector {} does not match the empty move vector for phase {}".format(move, state.phase))
            return None
        if  state.shape != self.empty_state().shape:
            raise TypeError("The shape of the state vector {} does not match the empty state vector for phase {}".format(state, state.phase))
            return None
        
        nstate = state.copy()
        legal = move & self.get_legal_moves(state)
        if np.sum(legal) == 0:
            raise BadMoveException("No legal moves found")
            return None
        
        if state.phase == 0:
            spot = np.unravel_index(np.argmax(legal), legal.shape)
            nstate[spot][state.next_player] = 1
            nstate.next_player = self._other_player(state.next_player)
            return nstate
        elif state.phase == 1:
            indexes = np.unravel_index(np.argsort(legal), legal.shape)
            pair_value = lambda a, b: legal[a][state.next_player] + legal[b][state.next_player])
            pairs = sorted(filter(pair_value, combinations(indexes, 2)), cmp=pair_value)
            for pair in pairs:
                start, end = pair
                startw, endw = state[start], state[end]
                if not startw[:, state.next_player].any():
                    startw, endw = endw, startw
                    start, end = end, start
                if not startw[:, state.next_player].any() \
                   or endw[:, state.next_player].any():
                    # illegal moves, check the next pair
                    continue
                
                if endw[:, self._other_player(state.next_player)].any():
                    # pieces attack each other
                    attacker = np.argmax(startw[:, state.next_player])
                    defender = np.argmax(endw[:, self._other_player(state.next_player)])
                    
                    if defender == 10 and attacker == 7 or \
                       defender == 1  and attacker == 9 or \
                       defender == 11 or \
                       attacker < defender:
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
                    break
                else:
                    # piece just moves
                    piece = np.argmax(startw[:, state.next_player])
                    nstate[start][piece, state.next_player] = 0
                    nstate[end][piece, state.next_player] = 1
                    # copy Seen? flag
                    nstate[end][-1, state.next_player] = nstate[start][-1, state.next_player]
                    nstate[start][-1, state.next_player] = 0
                    nstate.next_player = self._other_player(state.next_player)
                    break
            
            # always return
            nstate.next_player = self._other_player(state.next_player)
            return nstate
            
        else:
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
           or not state[:, :, :10, self._other_player(state.next_player)]:
            return set((state.next_player,))
        
        return set()
