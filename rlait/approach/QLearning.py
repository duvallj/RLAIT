from .Approach import Approach
from .Random import Random # to test against
from ..util import STATE_TYPE_OPTION, dotdict, parallelize

import random
from pickle import Pickler, Unpickler
import logging
import os

log = logging.getLogger(__name__)

class QLearning(Approach):
    def __init__(self, argdict):
        """
        Initializes an approach before
        it applies itself to a task

        Parameters
        ----------
        argdict : rlait.util.misc.dotdict
            A dictionary containing all the extra arguments that AlphaZero uses.
            Fully explained in the Notes section

        Notes
        -----
        Full list of possible values that can be provided in argdict,
        as well as their default values:
        * discount : float (0.90)
            A factor detailing how much to prioritize future rewards
        * lr : float (0.2)
            Learning rate, how much to weight new experiences over old ones

        * num_procs : int (4)
            Number of parallel processes to train with
        * games_per_proc : int(16)
            Number of games to run per process. Together with num_procs
            determines how many games to run per training interation
        * chunksize : int(1)
            Chunksize for underlying process pool. For small numbers of games
            can be left at 1.

        * checkpoint_dir : str ("./checkpoints")
            Folder to store the checkpoints in. Must be an absolute path or a
            path relative to the location of the script running
        """
        super().__init__(approach_name="qlearning")

        self.args = dotdict(argdict)

        self.args.lr                              = self.args.get('lr', 0.2)
        self.args.discount                        = self.args.get('discount', 0.90)

        self.args.num_procs                       = self.args.get('num_procs', 4)
        self.args.games_per_proc                  = self.args.get('games_per_proc', 16)
        self.args.chunksize                       = self.args.get('chunksize', 1)

        self.args.checkpoint_dir                  = self.args.get('checkpoint_dir', "./checkpoints")


    def init_to_task(self, task):
        """
        Customizes an approach to work on
        a specific task

        Parameters
        ----------
        task : Task
            The Task object to customize to. Should provide
            all the necessary methods for this approach to customize,
            like length of move vectors for different phases.

        Returns
        -------
        self
            For daisy-chaining purposes. Other methods return
            self for the same reason
        """

        self.task = task
        self.Q = dict()

        # Returning self so that constructs like
        # `ai = CustomApproach(args).init_to_task(Task(more_args))`
        # become possible
        return self

    def get_move(self, state):
        """
        Gets a move the AI wants to play for the passed in state.

        Parameters
        ----------
        state : State
            The state object containing all the state information and
            next player information. Size and shape varies per Task.
            Ideally should be canonicalized.

        Returns
        -------
        move : Move
            The move object containing the move information. Size and
            shape varies per Task.

        Notes
        -----
        Selects the move with the highest Q-value given the state.
        If it's seen the state but not explored any move, it will choose a random
        move from ones it hasn't explored.
        If it hasn't seen the state, it will choose a random move.
        """

        s = self.task.state_string_representation(state)
        moves = list(self.task.iterate_legal_moves(state))

        if s in self.Q:
            best_move = None
            best_Q = float('-inf')

            # don't always choose same unexplored move
            random.shuffle(moves)
            for move in moves:
                a = self.task.move_string_representation(move, state)
                # encourage exploration
                update_Q = self.Q[s].get(a, float('inf'))
                if update_Q > best_Q:
                    best_Q = update_Q
                    best_move = move

            return best_move
        else:
            # We've never seen this state before, return random
            return random.choice(moves)


    def load_weights(self, filename):
        """
        Loads a previous Approach state from a file. Just the
        weights, history is loaded separately.

        Parameters
        ----------
        filename : str
            File to load the weights from. Not having this be a
            list allows for other data encoding schemes.

        Returns
        -------
        self
        """

        filepath = filename
        if not os.path.exists(filepath):
            filepath = os.path.join(self.args.checkpoint_dir, filename)
            if not os.path.exists(filepath):
                raise("No model in local file {} or path {}!".format(filename, filepath))

        new_Q = None
        with open(filepath, "rb") as f:
            new_Q = Unpickler(f).load()

        self.Q = new_Q

        return self

    def save_weights(self, filename):
        """
        Saves the weights of the current state to a file.

        Parameters
        ----------
        filename : str
            File to save the weights to.

        Returns
        -------
        self
        """

        filepath = os.path.join(self.args.checkpoint_dir, filename)
        if not os.path.exists(self.args.checkpoint_dir):
            print("Checkpoint Directory does not exist! Making directory {}".format(self.args.checkpoint_dir))
            os.mkdir(self.args.checkpoint_dir)

        with open(filepath, "wb") as f:
            Pickler(f).dump(self.Q)

        return self

    def load_history(self, filename):
        """
        Loads a game history from a file. A file can optionally
        contain one or many History classes, and this method
        can be extended with optional arguments to specify how
        many histories to load.

        Parameters
        ----------
        filename : str
            File to load history from.

        Returns
        -------
        self

        Notes
        -----
        This does nothing because, by default, Q-Learning has no
        notion of game history, only weights.
        """

        return self

    def save_history(self, filename):
        """
        Saves the current game history to a file. Should generally
        append to the history in the file if it exists.

        Parameters
        ----------
        filename : str
            File to save/append history to.

        Returns
        -------
        self

        Notes
        -----
        This does nothing because, by default, Q-Learning has no
        notion of game history, only weights.
        """

        return self

    def _get_checkpoint_filename(self, iteration):
        return "checkpoint_{}.pth.tar".format(iteration)

    def _play_game(self, n):
        board = self.task.empty_state(0)
        temp_history = []

        while not self.task.is_terminal_state(board):
            move = self.get_move(board)
            s = self.task.state_string_representation(board)
            a = self.task.move_string_representation(move, board)
            temp_history.append((s,a,board.next_player))
            board = self.task.apply_move(move, board)

        winners = self.task.get_winners(board)

        return temp_history, winners

    def train_once(self):
        """
        Runs a a series of training games to update the Q-values.

        Does not create a checkpoint, must be done manually or
        by running test_once.

        Notes
        -----
        How Q-Learning works is that every time it encounters a reward
        throughout a game (in this case, a single reward value for winning
        given at the end of the game), it propagates that value through a
        Q-table that keeps track of the expected future reward from every move
        given a state. By playing the game many times, you can view many states
        and eventually figure out the optimal actions.

        The main issue at hand is that for large state spaces, this approach
        becomes impractical. Deep Q-Learning approaches try to solve this by
        emulate the Q-table with a neural network. Here, though, we just use
        a dictionary.
        """

        history = parallelize(self._play_game, self.args.num_procs,
            self.args.games_per_proc, self.args.chunksize)

        for temp_history, winners in history:
            last_s, last_a, last_np = temp_history[-1]
            if last_s not in self.Q:
                self.Q[last_s] = dict()
            self.Q[last_s][last_a] = (1-self.args.lr) * self.Q[last_s].get(last_a, 0) \
                                + self.args.lr * (10 * (-1)**(last_np not in winners))
                                # reward for winning, or penalty for losing

            for i in range(len(temp_history)-2, -1, -1):
                # no reward for intermediate moves
                # unfortunately hampers learning for anything
                # except the shortest tasks
                s, a, np = temp_history[i]
                if s not in self.Q:
                    self.Q[s] = dict()
                prev_Q = max(self.Q[temp_history[i+1][0]].values()) * (-1)**(np != last_np)
                self.Q[s][a] = (1-self.args.lr) * self.Q[s].get(a, 0) \
                            + self.args.lr * (self.args.discount * prev_Q)


    def test_once(self):
        """
        Runs a single testing interation. Does not change weights, but
        does create a checkpoint.

        Returns
        -------
        score : float
            ELO, win percentage, other number where higher is better
        """

        return None
