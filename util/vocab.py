import numpy as np

STATE_TYPE_OPTIONS = ['flat', 'rect', 'deepflat', 'deeprect']

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
    each location contain a vector with all the possible types of pieces or 
    occupation states any player could have in that spot, and then the
    vector of occupying players. This is to support more tasks like Chess 
    for example.
    """
    
    def __array_finalize__(self, obj):
        if isinstance(obj, State):
            self.task_name = obj.task_name
            self.phase = obj.phase
            self.type = obj.type
            self.next_player = obj.next_player
        elif obj is not None:
            raise TypeError("You can only cast States from other States")
            
    def __init__(self, *args, task_name="empty_task", phase=0, type=0, next_player=0, **kwargs):
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
        self.task_name = task_name
        self.phase = phase
        self.type = type
        self.next_player = next_player
            
    
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