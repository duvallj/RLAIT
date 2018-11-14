from ..task import Task
from ..util import Move, State

import numpy as np

class Stratego(Task):
    def __init__(self, size=10, **kwargs):
        """
        Initializes Stratego 
        
        Parmeters
        ---------
        task_name : str
            Name of task, used for printing
        num_phases : int
            Total number of different phases this task can have
        """
        kwargs['task_name'] = kwargs.get('task_name', 'stratego')
        kwargs['num_phases'] = 2
        super().__init__(self, **kwargs)
        
        self.N = size
        
        # initialize empty move/state vectors
        # explanations for sizes found in docstrings
        self._empty_moves = [
            Move(np.ones((self.N, self.N, 13), dtype=bool)),
            Move(np.ones((self.N, self.N), dtype=bool))
        ]
        
        self._empty_state = State(np.ones((self.N, self.N, 2, 13), dtype=bool))
        
        self._moveable_mask = np.ndarray([
            1, # 10
            1, # 9
            1, # 8
            1, # 7
            1, # 6
            1, # 5
            1, # 4
            1, # 3 
            1, # 2
            1, # Spy
            0, # Bomb
            0, # Flag
            0, # Hazard
        ])
        
    
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
        
        See https://www.hasbro.com/common/instruct/Stratego.pdf for complete rules.
        
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
                    Spy,
                    Bomb,
                    Flag,
                    Hazard
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
        See `empty_move` for order of pieces in the last dimension. Player
        dimension comes before piece dimension. Hazards belong to neither
        player, but exist on both players' slots to signify there's something
        there.
        
        ```
        [                       # Rows on board
            [                   # Columns in row
                [               # Players on space
                    [           # Pieces from player
                        ... (13)
                    ],
                    [...]       # the other player
                ],
                ...
                * N total
            ],
            ...
            * N total
        ]
        ```
        
        That comes out to an (N, N, 2, 13) ndarray for both phases
        """
        
        return self._empty_state
       
    def _contains_friendly_piece(self, state, r, c):
        return state[r, c, state.next_player].any()
        
    def get_legal_moves(self, state, player):
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
        
        if state.phase == 0:
            return Move(np.fromfunction(
                lambda r, c, p: (state.next_player == 0 and r < self.N//2-1 or state.next_player == 1 and r > self.N//2) and not self._contains_friendly_piece(state, r, c), 
                self.empty_move(0).shape
                dtype=bool
            ))
        elif state.phase == 1:
            legal = self.empty_move(1)*0
            
            for r in range(self.N):
                for c in range(self.N):
                    if (state[r, c, state.next_player] | self._moveable_mask).any():
                        if state[r, c, state.next_player, 8]:
                            # Scout piece, handle accordingly
                        else:
                            for i, j in ((0,1),(0,-1),(1,0),(-1,0)):
                                
                        
        else:
            return None
        
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
        """
        
        return None
        
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
