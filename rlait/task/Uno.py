from ..util import State, Move
from .Task import Task

class Uno(Task):
    def __init__(self, num_players=4, num_colors=4, num_cards=8, init_hand_size=7):
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
        init_hand_size : int (7)
            How many cards players should be dealt initially
        """
        super().__init__(task_name="uno", num_phases=1)

        self.num_players = num_players
        self.num_colors = num_colors
        self.num_cards = num_cards
        self.init_hand_size = init_hand_size

        self._empty_move = np.zeros(self.num_colors * self.num_cards + 1).view(Move)
        self._empty_move.state_type = STATE_TYPE_OPTION_TO_INDEX['flat']

        self._empty_state = np.zeros((self.num_players+1, self.num_colors, self.num_cards)).view(State)
        self._empty_state.task_name = self.task_name
        self._empty_state.state_type = STATE_TYPE_OPTION_TO_INDEX['deeprect']
        self._empty_state.next_player = 0


    def get_random_card(self):
        return random.randint(0, self.num_colors-1), random.randint(0, self.num_cards-1)

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

        # a lot of times, AIs expect an "empty state" to actually be the initial state
        state = self._empty_state.copy()

        # initial card in the pile
        state[-1, self.get_random_card()] = 1

        for p in range(num_players):
            for c in range(self.init_hand_size):
                state[p, self.get_random_card()] += 1

        return state

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

        for i in range(self.num_colors * self.num_cards + 1):
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

        out = self.empty_move(phase=state.phase)

        legal_mask = self.get_legal_mask(state)

        for i in range(self.num_colors * self.num_cards + 1):
            if legal_mask[i] == 1:
                out[i] = 1
                yield out.copy()
                out[i] = 0

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

        out = self.empty_move(phase=state.phase)

        for y in range(self.num_colors):
            for x in range(self.num_cards):
                if state[state.next_player, y, x] and \
                  (state[-1, :, x].any() or state[-1, y, :].any()): # any of same color or number
                    out[y*self.num_cards+x] = 1

        # drawing a card always valid
        out[-1] = 1

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

        # TODO: we do want some way to tell how many cards the other players have
        # would a scalar be best or should there be some other way?
        # think about it.

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

        legal_mask = self.get_legal_mask(state)
        legal_moves = move * legal_mask

        if not legal_moves.any():
            raise BadMoveException("Error: no legal moves provided in {} for the state {}".format(move, state))



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

        for p in range(self.num_players):
            if not state[p, :, :].any():
                # player has played all their cards
                return True

        # no on has played all their cards
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
