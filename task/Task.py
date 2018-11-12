from ..util import State, Move
    
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
