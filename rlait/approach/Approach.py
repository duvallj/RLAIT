from ..util import State, Move, Game, History

class Approach:
    def __init__(self, approach_name="empty_approach"):
        """
        Initializes an approach before
        it applies itself to a task

        Parameters
        ----------
        approach_name : str
            Name of the approach, used for printing
        """
        self.approach_name = approach_name

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
