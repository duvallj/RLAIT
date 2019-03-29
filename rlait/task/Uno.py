from ..util import State, Move, BadMoveException, STATE_TYPE_OPTION_TO_INDEX
from .Task import Task

import numpy as np
import random

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

        self._empty_move = np.zeros(self.num_colors * self.num_cards + 1, dtype=int).view(Move)
        self._empty_move.state_type = STATE_TYPE_OPTION_TO_INDEX['flat']

        # players aren't first so that it makes more sense for AI algorithms
        # with the standard 'channels_last' structure
        self._empty_state = np.zeros((self.num_colors, self.num_cards, self.num_players+1), dtype=int).view(State)
        self._empty_state.task_name = self.task_name
        self._empty_state.state_type = STATE_TYPE_OPTION_TO_INDEX['deeprect']
        self._empty_state.next_player = 0


    def get_random_card(self):
        return random.randint(0, self.num_colors-1), random.randint(0, self.num_cards-1)

    def next_player(self, player):
        return (player + 1) % self.num_players

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
            A state vector initialized to a starting setup
        """

        # a lot of times, AIs expect an "empty state" to actually be the initial state
        state = self._empty_state.copy()

        # initial card in the pile
        c, n = self.get_random_card()
        state[c, n, -1] = 1

        for p in range(self.num_players):
            for c in range(self.init_hand_size):
                c, n = self.get_random_card()
                state[c, n, p] += 1

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

        c_color, c_number = np.unravel_index(np.argmax(state[..., -1]), (self.num_colors, self.num_cards))

        # get all the cards in hand with the current color or number
        for color in range(self.num_colors):
            if state[color, c_number, state.next_player]:
                out[color*self.num_cards + c_number] = 1
        for number in range(self.num_cards):
            if state[c_color, number, state.next_player]:
                out[c_color*self.num_cards + number] = 1

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

        nstate = state.copy()

        # rotate hands around until current player is in slot 0
        nstate[..., :-1] = np.roll(nstate[..., :-1], -1*state.next_player, axis=2)
        # mask other player's hands, replacing all their values with
        # the number of cards they have
        for p in range(1, self.num_players):
            nstate[..., p] = np.sum(nstate[..., p])

        nstate.next_player = 0

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

        legal_mask = self.get_legal_mask(state)
        legal_moves = move * legal_mask

        if not legal_moves.any():
            raise BadMoveException("Error: no legal moves provided in {} for the state {}".format(move, state))

        index = np.argmax(legal_moves)
        nstate = state.copy()

        if index == self.num_cards * self.num_colors:
            # pass and draw
            c, n = self.get_random_card()
            nstate[c, n, state.next_player] += 1
        else:
            color, number = np.unravel_index(index, (self.num_colors, self.num_cards))
            # clear the field
            nstate[..., -1] = 0
            # set the new card
            nstate[color, number, -1] = 1
            # decrease the amount of cards the player has
            nstate[color, number, state.next_player] -= 1

        # get the next player
        nstate.next_player = self.next_player(state.next_player)

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

        for p in range(self.num_players):
            if not state[..., p].any():
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
            A set containing all the winners. Empty if no winners.

        Notes
        -----
        There are no ties in Uno, it is impossible
        """

        winners = set()
        for p in range(self.num_players):
            if not state[..., p].any():
                winners.add(p)
        return winners

    def color_number_to_string(self, color, number):
        if self.num_colors < 26:
            return chr(65+color) + '-' + str(number)
        else:
            return '{}-{}'.format(color, number)

    def string_to_color_number(self, string):
        color, number = string.split('-')
        if self.num_colors < 26:
            color = ord(color) - 65
        else:
            color = int(color)
        number = int(number)

        return color, number

    def state_string_representation(self, state):
        """
        Returns a string representation of a board, fit for printing and/or caching

        Parameters
        ----------
        state : State

        Returns
        -------
        str

        Notes
        -----
        Expects a non-canoncial state, canonicalizes internally
        """

        cstate = self.get_canonical_form(state)

        out = "player {}\n".format(state.next_player)
        out += 'hand: '

        cards_in_hand = []
        for c in range(self.num_colors):
            for n in range(self.num_cards):
                if cstate[c, n, 0]:
                    for x in range(cstate[c, n, 0]):
                        cards_in_hand.append(self.color_number_to_string(c, n))

        out += ' '.join(cards_in_hand) + '\n'
        out += 'field: {}\n'.format(self.color_number_to_string(
            *np.unravel_index(np.argmax(state[..., -1]), (self.num_colors, self.num_cards))
        ))

        out += 'other players: {}\n'.format(' '.join(str(cstate[0, 0, p]) for p in range(1, self.num_players)))

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

        index = np.argmax(move)

        if index == self.num_colors * self.num_cards:
            return 'PASS'
        else:
            color, number = np.unravel_index(index, (self.num_colors, self.num_cards))
            return self.color_number_to_string(color, number)

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

        out = self.empty_move(phase=phase)

        if move_str == 'PASS':
            out[-1] = 1
        else:
            try:
                color, number = self.string_to_color_number(move_str)
                out[color*self.num_cards+number] = 1
            except:
                raise BadMoveException("Error: move {} is formatted incorrectly".format(move_str))

        return out
