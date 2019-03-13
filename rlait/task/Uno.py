from ..util import State, Move
from .Task import Task

class Uno(Task):
    def __init__(self, num_players=4, num_colors=4, num_cards=8):
        """
        Initializes the Uno task

        Parmeters
        ---------
        num_players : int (4)
            The number of players allowed in a game
        num_colors : int (4)
            How many different colors the cards can be
        num_cards : int (8)
            How many different numbers there can be on a card
        """
        self.task_name = task_name
        self.num_phases = num_phases


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

        return None

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

        return None

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

        yield None


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

        yield None

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

        return None

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
