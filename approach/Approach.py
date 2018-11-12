from ..util import State, Move, History

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
            all the necessary methods for this approach to customize, like length of move vectors for different
            phases.
            
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
        Loads a game history from a file. A file can optionally contain one or many History classes, and this method can be extended with optional arguments to specify how many histories to load.
        
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
        Saves the current game history to a file. Should generally append to the history in the file if it exists.
        
        Parameters
        ----------
        filename : str
            File to save/append history to.
            
        Returns
        -------
        self
        """
        
        return self