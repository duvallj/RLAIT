from ..Approach import Approach
from ...util import BadMoveException

class InteractivePlayer(Approach):
    def __init__(self, approach_name="interactive"):
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

        self.task = task

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

        move = None
        while move is None:
            move_str = input("Move for {}: ".format(self.task.task_name))
            try:
                move = self.task.string_to_move(move_str, state)
                _ = self.task.apply_move(move, state)
            except BadMoveException as e:
                print(e)
                move = None

        return move

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
