from ..Approach import Approach

import numpy as np

class AlphaZero(Approach):
    def __init__(self, argdict, approach_name="alphazero"):
        """
        Initializes an approach before
        it applies itself to a task

        Parameters
        ----------
        argdict : rlait.util.misc.dotdict
            A dictionary containing all the extra arguments that AlphaZero uses.
            Fully explained in the Notes section
        approach_name : str
            Name of the approach, used for printing

        Notes
        -----
        Full list of possible arguments that can be provided in argdict,
        as well as their default values:

        * numEps : int (30)
            The number of playout episodes to run per training iteration
        * tempThreshold : int (15)
            The number of moves to make using weighted plays instead of maximum
            plays when training
        * updateThreshold : float (0.6)
            ??? TODO: figure this out
        * maxlenOfQueue : int (200000)
            ??? TODO: figure this out
        * numMCTSSims : int (30)
            The number of times to run MCTS per move during self-play and actual play
        * arenaCompare : int (11)
            The number of games to play against the previous best AI at the end
            of a training iteration
        * cpuct : float (1.0)
            A factor that determines how likely the MCTS is to explore. (TODO: explain it better)

        * load_checkpoint : bool (False)
            Do we load a checkpoint?
        * checkpoint : str (None)
            Checkpoint to load. Must be set if `load_checkpoint` is set, should
            be a file path relative to the below directory.
        * checkpoint_dir : str ("./checkpoints")
            Folder to store the checkpoints in. Must be an absolute path or a
            path relative to the location of this file.
        * numItersForTrainExamplesHistory : int(30)
            The number of past iterations to store in a single history file.
        """
        super().__init__(approach_name)

        self.args = argdict

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
        """

        return None

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
        """

        return self

    def train_once(self):
        """
        Runs a single training iteration to fine-tune the weights. Possible
        side effects include:

        * Changing the internals weights (duh)
        * Adding to the history (optional)
        * Printing to console
        * Automatically calling `save_history` and `save_weights`

        In implementations, can take custom arguments here in a single dict
        or have arguments passed in an earlier initialization phase.
        Any settings passed in here are expected to override default settings.
        """
        pass


    def test_once(self):
        """
        Runs a single testing interation. Does not change weights, and usually
        does not change the history either.

        Returns
        -------
        score : float
            ELO, win percentage, other number where higher is better
        """

        return None
