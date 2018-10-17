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
    
class Task:
    def __init__(self, task_name="empty_task", num_phases=1):
        """
        Initializes a task
        
        Parmeters
        ---------
        task_name : str
            Name of task, used for printing
        num_phases : int
            Total number of different phases this task can have
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
        pass
        
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
        pass
        
    def get_legal_moves(self, state):
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
        pass
        
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
        pass
        
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
        pass
