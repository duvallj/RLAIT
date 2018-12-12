from ..Approach import Approach

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *

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

        * lr : float (0.001)
            Backpropagation learning rate
        * dropout : float (0.3)
            Dropout factor to use in the densely connected layers
        * epochs : int (10)
            Number of epochs to train network on past examples each iteration
        * batch_size : int (64)
            Input batch size to use with neural network
        * cuda : bool (True)
            Use CUDA to speed training?
        * num_channels : int (512)
            Number of features to detect in the convolutional layers
        (The number of layers, activation functions are fixed)

        * numEps : int (30)
            The number of playout episodes to run per training iteration
        * tempThreshold : int (15)
            The number of moves to make using weighted plays instead of maximum
            plays when training
        * updateThreshold : float (0.6)
            Fraction of games a challenger network needs to win in order to
            become the new base.
        * maxlenOfQueue : int (200000)
            The maximum number of training examples to train on.
        * numMCTSSims : int (30)
            The number of times to run MCTS per move during self-play and actual play
        * arenaCompare : int (11)
            The number of games to play against the previous best AI at the end
            of a training iteration
        * cpuct : float (1.0)
            A factor that determines how likely the MCTS is to explore.

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

        self.args.lr                              = self.args.get("lr", 0.001)
        self.args.dropout                         = self.args.get("dropout", 0.3)
        self.args.epochs                          = self.args.get("epochs", 10)
        self.args.batch_size                      = self.args.get("batch_size", 64)
        self.args.cuda                            = self.args.get("cuda", True)
        self.args.num_channels                    = self.args.get("num_channels", 512)

        self.args.numEps                          = self.args.get("numEps", 30)
        self.args.tempThreshold                   = self.args.get("tempThreshold", 15)
        self.args.updateThreshold                 = self.args.get("updateThreshold", 0.6)
        self.args.maxlenOfQueue                   = self.args.get("maxlenOfQueue", 200000)
        self.args.numMCTSSims                     = self.args.get("numMCTSSims", 30)
        self.args.arenaCompare                    = self.args.get("arenaCompare", 11)
        self.args.cpuct                           = self.args.get("cpuct", 1.0)

        self.args.load_checkpoint                 = self.args.get("load_checkpoint", False)
        self.args.checkpoint                      = self.args.get("checkpoint", None)
        self.args.checkpoint_dir                  = self.args.get("checkpoint_dir", "./checkpoints")
        self.args.numItersForTrainExamplesHistory = self.args.get("numItersForTrainExamplesHistory", 30)

    def _reset_mcts(self):
        # Reset MCTS variables (clears cache)
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

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

        Notes
        -----
        Although the parameters for the neural network size are passed through
        the __init__ method, the neural networks are actually created in this
        method. This is to handle different sizes of input and output layers
        required for different Tasks.
        """

        # Assumes that the input board shape will remain constant between phases
        # Unfortunately, there's no good way to do this without that assumption.

        empty_state = task.empty_state(0):
        for phase in range(1, task.num_phases):
            try:
                assert task.empty_state(phase).shape == empty_state.shape
            except AssertionError:
                raise TypeError("{} cannot be applied to tasks with variant board representations!".format(self.task_name))

        if 'flat' in STATE_TYPE_OPTION[empty_state.state_type]:
            raise TypeError("{} currently does not support tasks with board type \"flat\"".format(self.task_name))

        self.input = Input(shape=empty_state.shape)

        if STATE_TYPE_OPTION[empty_state.state_type] == 'deeprect':
            extra_dim = 1
            for i in range(2, len(empty_state.shape)):
                extra_dim *= empty_state.shape[i]
            x_image = Reshape((empty_state.shape[0], empty_state.shape[1], extra_dim))(self.input)
        elif STATE_TYPE_OPTION[empty_state.state_type] == 'rect':
            x_image = Reshape((empty_state.shape[0], empty_state.shape[1], 1))(self.input)
        else:
            raise TypeError("Unknown state type \"{}\"".format(empty_state.state_type))
            x_image = None

        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='valid')(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='valid')(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(self.args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))                # batch_size x 1024
        s_fc2 = Dropout(self.args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))                        # batch_size x 1024
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                                                                        # batch_size x 1

        self.outputs = []
        self.output_sizes = []
        self.action_sizes = []


        for phase in range(task.num_phases):
            empty_move = task.empty_move(phase)
            output_size = 1
            for i in range(len(empty_move.shape)):
                output_size *= empty_move.shape[i]
            output = Dense(output_size, activation='softmax', name='pi{}'.format(phase))(s_fc2)   # batch_size x self.output_size
            self.outputs.append(output)
            self.output_sizes.append(output_size)

            action_size = list(map(task.move_string_representation, task.iterate_all_moves(phase)))
            self.action_sizes.append(action_size)

        self.model = Model(inputs=self.input, outputs=self.outputs)

        self._reset_mcts()

        # Returning self so that constructs like
        # `ai = CustomApproach(args).init_to_task(Task(more_args))`
        # become possible
        return self

    def _nn_predict(self, state):
        pass

    def _search(self, board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Parameters
        ----------
        canonicalBoard : State
            The canonicalized version of the board to search

        Returns
        -------
            v: the negative of the value of the current canonicalBoard

        Notes
        -----
        The return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        """

        canonicalBoard = self.task.get_canonical_form(board)
        phase = board.phase
        s = self.task.string_respresentation(canonicalBoard)

        if s not in self.Es:
            winners = self.task.get_winners(canonicalBoard)

            self.Es[s] = 1-len()
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            # TODO: make sure _nn_predict actually generates a move with correct dimesions + type
            self.Ps[s], v = self._nn_predict(canonicalBoard)
            valids = self.task.get_legal_mask(canonicalBoard)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 1.0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        """
        TODO: how do I iterate over all moves for tasks that accept an entire
        array as their input?
        New idea: legal_moves iterator. Change below loop+check to use that instead
        """

        for a in self.action_sizes[phase]:
            if valids[a]: # TODO: basically, if a is a valid move. I need a better way to check...
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.task.apply_move(board, self.task.string_to_move(a))

        v = self.search(next_s)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v

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
