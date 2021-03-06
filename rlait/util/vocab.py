import numpy as np

STATE_TYPE_OPTION = ['flat', 'rect', 'deepflat', 'deeprect']
STATE_TYPE_OPTION_TO_INDEX = {
    'flat': 0,
    'rect': 1,
    'deepflat': 2,
    'deeprect': 3,
}

class State(np.ndarray):
    """
    Attributes
    ----------
    task_name : str
        Which task this state belongs to
    phase : int
        Phase of task this state belongs to
    type : int
        Format of state vector, so approaches know how to properly handle
    next_player : int
        The ID of the next player to move. Corresponds to the index of a player vector

    Notes
    -----
    For every location in the state (be it in `flat` or `rect`) format,
    there is a boolean vector containing all the players in that location.
    A `deepflat` or `deeprect` configuration is similar, only it has
    each player's location be a vector with all the possible types of pieces or
    occupation states any player could have in that spot. This is to support more tasks like Chess
    for example.
    """

    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None,
                task_name="empty_task", phase=0, state_type=0, next_player=0):
        """
        Class initializer. Preferred method of creating States.

        Parameters
        ----------
        task_name : str
            Name of task this state belongs to
        phase : int, optional
            Phase of task this state belongs to
        type : int
            Format of state vector
        first_player : int
            ID of the first player to make a move
        """

        obj = super(State, subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)

        obj.task_name = task_name
        obj.phase = phase
        obj.state_type = state_type
        obj.next_player = next_player

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        self.task_name = getattr(obj, 'task_name', "empty_task")
        self.phase = getattr(obj, 'phase', 0)
        self.state_type = getattr(obj, 'state_type', 0)
        self.next_player = getattr(obj, 'next_player', 0)

    # copied from https://stackoverflow.com/a/26599346
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(State, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.task_name, self.phase, self.state_type, self.next_player,)
        # Return a tuple that replaces the parent's __reduce__
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # Set the info attributes accordingly
        self.task_name = state[-4]
        self.phase = state[-3]
        self.state_type = state[-2]
        self.next_player = state[-1]
        # Call the parent's __setstate__ with the original tuple
        super(State, self).__setstate__(state[0:-4])


class Move(State):
    """
    Notes
    -----
    Very similar to State, even comes with the same format options, only
    difference is that it contains booleans for whether or not a specified
    player can make/makes a move. Because that is a property of the stored
    ndarray, not of the class, this subclasses State.
    """
    pass


class BadMoveException(Exception): pass

class History(np.ndarray):
    """
    Notes
    -----
    Stores the collective history of multiple games, in the folowing format:

    ```
    History = [
        Game,
        Game,
        Game,
        ...
    ]

    Game = [
        (State, None),   # Represents initial state
        (State, Move),
        (State, Move),
        ...
    ]
    ```

    This works because each State contains the next player and phase data within
    it. Also should be fairly efficient because everything  except the
    one state-move tuple is ndarrays.
    """
    pass

class Game(np.ndarray):
    """
    Notes
    -----
    Stores the current progress of a game. See above for implementation details.
    """
    pass
