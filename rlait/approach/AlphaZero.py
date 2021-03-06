from .Approach import Approach
from ..model.SimpleConvNet import SimpleConvNet
from ..util import STATE_TYPE_OPTION, dotdict

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.generic_utils import Progbar
import h5py

import random
import math
from pickle import Pickler, Unpickler
from collections import deque
import logging
import os
import io

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EPS = 1e-8

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

        * startFromEp : int (0)
            The episode number to start from. Useful if a previous run got interrupted
        * numEps : int (30)
            The number of playout episodes to run per training iteration
        * tempThreshold : int (15)
            The number of moves to make using weighted plays instead of maximum
            plays when training
        * updateThreshold : float (0.51)
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
        * maxDepth : int (5000)
            The maximum search depth to explore for one move
        * startWithRandomPlay : bool (false)
            If starting at the first iteration, do we use random play to generate
            some training examples to look at first?
        * numRandomPlayExamples : int (100)
            Number of random game examples to generate

        * load_checkpoint : bool (False)
            Do we load a checkpoint?
        * prevHistory : str (None)
            Previous history to load. Can be set if `load_checkpoint` is set. If
            set, AlphaZero skips the first self-play iteration and jumps straight
            to training a new network on the provided history.
        * checkpoint : str (None)
            Checkpoint to load. Must be set if `load_checkpoint` is set, should
            be a file path relative to the below directory.
        * checkpoint_dir : str ("./checkpoints")
            Folder to store the checkpoints in. Must be an absolute path or a
            path relative to the location of the script running.
        * numItersForTrainExamplesHistory : int(30)
            The number of past iterations to store in a single history file.
        """
        super().__init__(approach_name)

        self.args = dotdict(argdict)

        self.args.lr                              = self.args.get("lr", 0.001)
        self.args.dropout                         = self.args.get("dropout", 0.3)
        self.args.epochs                          = self.args.get("epochs", 10)
        self.args.batch_size                      = self.args.get("batch_size", 64)
        self.args.cuda                            = self.args.get("cuda", True)
        self.args.num_channels                    = self.args.get("num_channels", 512)

        self.args.startFromEp                     = self.args.get("startFromEp", 0)
        self.args.numEps                          = self.args.get("numEps", 30)
        self.args.tempThreshold                   = self.args.get("tempThreshold", 15)
        self.args.updateThreshold                 = self.args.get("updateThreshold", 0.51)
        self.args.maxlenOfQueue                   = self.args.get("maxlenOfQueue", 200000)
        self.args.numMCTSSims                     = self.args.get("numMCTSSims", 30)
        self.args.arenaCompare                    = self.args.get("arenaCompare", 11)
        self.args.cpuct                           = self.args.get("cpuct", 1.0)
        self.args.maxDepth                        = self.args.get("maxDepth", 500)
        self.args.startWithRandomPlay             = self.args.get("startWithRandomPlay", False)
        self.args.numRandomPlayExamples           = self.args.get("numRandomPlayExamples", 100)

        self.args.load_checkpoint                 = self.args.get("load_checkpoint", False)
        self.args.checkpoint                      = self.args.get("checkpoint", None)
        self.args.prevHistory                     = self.args.get("prevHistory", None)
        self.args.checkpoint_dir                  = self.args.get("checkpoint_dir", "./checkpoints")
        self.args.numItersForTrainExamplesHistory = self.args.get("numItersForTrainExamplesHistory", 30)

        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False

    def _reset_mcts(self):
        # Reset MCTS variables (clears cache)
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def init_model(self):
        """
        Initializes AlphaZero with a list of models (one for each task phase)
        in self.models
        """
        # Assumes that the input board shape will remain constant between phases
        # Unfortunately, there's no good way to do this without that assumption.

        ###########################################
        # Define network based on Task parameters #
        ###########################################

        empty_state = self.task.empty_state(0)
        for phase in range(1, self.task.num_phases):
            try:
                assert self.task.empty_state(phase).shape == empty_state.shape
            except AssertionError:
                raise TypeError("{} cannot be applied to tasks with variant board representations!".format(self.approach_name))

        if 'flat' in STATE_TYPE_OPTION[empty_state.state_type]:
            raise TypeError("{} currently does not support tasks with board type \"flat\"".format(self.approeach_name))

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

        conv_model = SimpleConvNet(self.args, x_image)

        self.v = Dense(self.task.num_players, activation='tanh', name='v')(conv_model.output)                                                                        # batch_size x 1

        self.outputs = [self.v]
        self.output_sizes = [self.task.num_players]

        self.models = []

        for phase in range(self.task.num_phases):
            empty_move = self.task.empty_move(phase)
            output_size = 1
            for i in range(len(empty_move.shape)):
                output_size *= empty_move.shape[i]
            output = Dense(output_size, activation='softmax', name='pi{}'.format(phase))(conv_model.output)   # batch_size x self.output_size
            self.outputs.append(output)
            self.output_sizes.append(output_size)

            model = Model(inputs=self.input, outputs=[self.v, output])
            model.compile(loss=['mean_squared_error', 'categorical_crossentropy'], optimizer=Adam(self.args.lr))
            self.models.append(model)

        # Doing the above instead of this because we need to have multiple models
        # in order to train correctly. Unfortunately, that means we cannot share
        # self.model = Model(inputs=self.input, outputs=self.outputs)

    def init_to_task(self, task, make_competetor=True):
        """
        Customizes an approach to work on
        a specific task

        Parameters
        ----------
        task : Task
            The Task object to customize to. Should provide
            all the necessary methods for this approach to customize,
            like length of move vectors for different phases.
        competetor : bool (True)
            Optional, controls whether we initialize a competetor AI for selfplay.
            When creating a competetor, this is turned to False so we don't
            infinitely recur.

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

        self.task = task

        self.init_model()
        self._reset_mcts()

        #######################
        # Load previous model #
        #######################

        self.iteration = self.args.startFromEp

        if self.args.load_checkpoint:
            if not (self.args.checkpoint is None):
                self.load_weights(self.args.checkpoint)
            else:
                try:
                    self.load_weights(self._get_checkpoint_filename(self.iteration))
                except:
                    log.warn("Tried to load checkpoint from starting iteration, could not find it.")
            if not (self.args.prevHistory is None) and make_competetor:
                self.load_history(self.args.prevHistory)
                self.skipFirstSelfPlay = True

        if make_competetor:
            self.pnet = self.__class__(self.args).init_to_task(self.task, make_competetor=False)
        else:
            self.pnet = None

        # Returning self so that constructs like
        # `ai = CustomApproach(args).init_to_task(Task(more_args))`
        # become possible
        return self

    def _nn_predict(self, state):
        v, pi = self.models[state.phase].predict(np.reshape(state, (1,)+state.shape))
        return v[0], pi[0]

    def _search(self, board, depth, maxdepth=float('inf')):
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
        board : State
            The board to search

        Returns
        -------
            v: the player value vector corresponding to the searched board

        Notes
        -----
        Because of the internal use of canonical boards, every v returned is
        relative to the perspective of the first player.
        """

        if depth > maxdepth: return np.zeros((self.task.num_players,))

        canonicalBoard = self.task.get_canonical_form(board)
        phase = board.phase
        s = self.task.state_string_representation(board)

        if s not in self.Es:
            if self.task.is_terminal_state(board):
                winners = self.task.get_winners(board)
                self.Es[s] = np.array([1 if (i in winners) else -1 for i in range(self.task.num_players)])
            else:
                self.Es[s] = np.zeros((self.task.num_players,))

        if not (self.Es[s] == 0).all():
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            # NOTE: this section is the only place canonical boards are introduced
            # the rest of the searching takes place on non-canonical boards
            # for ease of passing hidden information.
            # NOTE: but I shouldn't even be passing around hidden information because it should be hidden even when searching???
            v, self.Ps[s] = self._nn_predict(canonicalBoard)
            # correcting ordering of v
            v = np.concatenate((v[(self.task.num_players-board.next_player):], v[:(self.task.num_players-board.next_player)]))
            self.Ps[s] = np.reshape(self.Ps[s], self.task.empty_move(board.phase).shape)
            # unfortunately, this requires that canonical forms do nothing
            # like flipping the board, which is usually normal...
            valids = self.task.get_legal_mask(board)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0.0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.warn("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = list(self.task.iterate_legal_moves(board))
            self.Ns[s] = 1
            #print("For state {}".format(self.task.state_string_representation(canonicalBoard)))
            #print("Predicted move: {} and value: {}".format(self.Ps[s], v))
            return v

        valid_moves = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for move in valid_moves:
            a = self.task.move_string_representation(move, board)
            if (s,a) in self.Qsa:
                u = self.Qsa[(s,a)][board.next_player] + self.args.cpuct*(move*self.Ps[s]).sum()*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
            else:
                u = self.args.cpuct*(move*self.Ps[s]).sum()*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        #print("For state {}".format(self.task.state_string_representation(board)))
        #print("Best move is {}".format(a))
        next_s = self.task.apply_move(self.task.string_to_move(a, board), board)

        try:
            v = self._search(next_s, depth+1, maxdepth=maxdepth)
        except RecursionError:
            # NOTE: this might cause some issues, but it hits the
            # recursion limit pretty infrequently anyways (most likely due
            # to inadequate tie handling anyways, which should be fixed)
            return 0

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v

    def _get_action_prob(self, state, temp=1):
        """
        Returns the probabilities for each legal move given a state.
        Also returns the legal moves themselves

        Parameters
        ----------
            state : State

        Returns
        -------
            probs : list(float)
                A policy vector where the probability of the ith action is
                proportional to Nsa[(s,a)]**(1./temp)
            avail_moves : list(Move)
                A list of available moves where each entry corresponds to the
                same index in probs
        """
        for i in range(self.args.numMCTSSims):
            self._search(state, 0, maxdepth=self.args.maxDepth)

        s = self.task.state_string_representation(state)
        avail_moves = list(map(lambda x: self.task.move_string_representation(x, state),
                     self.task.iterate_legal_moves(state)))
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 \
                for a in avail_moves]

        probs = []
        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA] = 1
        else:
            probs = [x**(1./temp) for x in counts]

        return probs, avail_moves

    def _get_random_action_prob(self, state, temp=1):
        """
        Just like _get_action_prob, only completely random
        """
        s = self.task.state_string_representation(state)
        avail_moves = list(map(lambda x: self.task.move_string_representation(x, state),
                     self.task.iterate_legal_moves(state)))
        probs = [1.0/len(avail_moves)] * len(avail_moves)

        return probs, avail_moves

    def get_move(self, state, temp=0):
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

        probs, avail_moves = self._get_action_prob(state, temp)

        chosen_move = random.choices(
            population=avail_moves,
            weights=probs,
            k=1
        )[0]

        return self.task.string_to_move(chosen_move, state)

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

        all_model_files = []
        with open(filepath, "rb") as f:
            all_model_files = Unpickler(f).load()

        for i in range(len(all_model_files)):
            buf = all_model_files[i]
            self.models[i].load_weights(buf)

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

        # This was the best thing I could think of to fit multiple models in one file:
        # Keras's h5py backend supports writing data directly to a BytesIO object
        # So, we just tell it to do that for all the models and write the resulting
        # list to a pickle file.
        # Will most certainly fail for large enough models
        all_model_files = []
        for i in range(len(self.models)):
            buf = io.BytesIO()
            self.models[i].save_weights(buf, overwrite=True)
            all_model_files.append(buf)

        with open(filepath, "wb") as f:
            Pickler(f).dump(all_model_files)

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

        filepath = filename
        if not os.path.exists(filepath):
            filepath = os.path.join(self.args.checkpoint_dir, filename)
            if not os.path.exists(filepath):
                raise("No checkpoint in local file {} or path {}!".format(filename, filepath))

        with open(filepath, "rb") as f:
            log.info(f"Loading History from {filepath}")
            self.trainExamplesHistory = Unpickler(f).load()

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

        folder = self.args.checkpoint_dir
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            Pickler(f).dump(self.trainExamplesHistory)

        return self

    def _get_checkpoint_filename(self, iteration):
        return "checkpoint_{}.pth.tar".format(iteration)

    def _run_one_selfplay(self, action_prob_func):
        trainExamples = []
        board = self.task.empty_state(phase=0)
        episodeStep = 0

        while not self.task.is_terminal_state(board):
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)

            probs, avail_moves = action_prob_func(board, temp=temp) #self._get_action_prob(board, temp=temp)
            # not applicable to all games, but might include later
            #sym = self.task.getSymmetries(canonicalBoard, pi)
            #for b,p in sym:

            pi = np.asarray([probs[i]*self.task.string_to_move(avail_moves[i],board) for i in range(len(probs))])
            pi = np.sum(pi, axis=0).flatten()

            trainExamples.append((board, board.next_player, pi))

            action = random.choices(
                population=avail_moves,
                weights=probs,
                k=1
            )[0]
            board = self.task.apply_move(self.task.string_to_move(action, board), board)

        # Game is over
        winners = self.task.get_winners(board)

        #print("SELFPLAY GAME OVER. Board: {}".format(self.task.state_string_representation(board)))
        #print("Winners: {}. Final player: {}.".format(winners, board.next_player))
        # This might be biased towards or against ties depending on the last player
        # to move, but in sufficiently complex games this should result in a 50/50
        # split anyways
        fmtTrainExamples = []
        for old_board, player, pi in trainExamples:
            r = np.array([1 if (i in winners) else -1 for i in range(self.task.num_players)])
            fmtTrainExamples.append((old_board, pi, r))
        return fmtTrainExamples

    def _run_selfplay(self):

        # bookkeeping
        log.info('------ITER ' + str(self.iteration) + '------')
        # examples of the iteration
        if self.iteration == 0 and self.args.startWithRandomPlay:
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            log.info("Running Random Play to start")
            bar = Progbar(self.args.numRandomPlayExamples)
            bar.update(0)

            for eps in range(self.args.numRandomPlayExamples):
                iterationTrainExamples += self._run_one_selfplay(self._get_random_action_prob)
                bar.add(1)

            self.trainExamplesHistory.append(iterationTrainExamples)

        elif self.iteration > self.args.startFromEp or not self.skipFirstSelfPlay:
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            bar = Progbar(self.args.numEps)
            bar.update(0)

            for eps in range(self.args.numEps):
                self._reset_mcts()
                iterationTrainExamples += self._run_one_selfplay(self._get_action_prob)
                # bookkeeping + plot progress
                bar.add(1)

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

        if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            log.debug("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
            self.trainExamplesHistory.pop(0)
        # backup history to a file
        # NB! the examples were collected using the model from the previous iteration
        self.save_history(self._get_checkpoint_filename(self.iteration)+".examples")

    def _match_phase(self, phase):
        def _internal_match_phase(a):
            return a.phase == phase
        return _internal_match_phase

    def _train_nnet(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        #input_boards, target_pis, target_vs = list(zip(*examples))

        # for each phase, filter out all the examples from that phase
        # and train the corresponding model on them
        for phase in range(self.task.num_phases):
            match_phase = self._match_phase(phase)
            f_input_boards, f_target_pis, f_target_vs = list(zip(*
                list(filter(
                    lambda x: match_phase(x[0]),
                    examples
                ))
            ))
            f_input_boards = np.asarray(f_input_boards)
            f_target_pis = np.asarray(f_target_pis)
            f_target_vs = np.asarray(f_target_vs)
            #f_target_vs = f_target_vs.reshape(f_target_vs.shape + (1,))
            self.models[phase].fit(
                x=f_input_boards,
                y=[f_target_vs, f_target_pis],
                batch_size=self.args.batch_size,
                epochs=self.args.epochs
            )

    def _arena_play_once(self, first_player, second_player, verbose=False):
        # TODO: Currently only supports 2-player games, should probably fix that
        if verbose:
            assert(self.task.state_string_representation)
        board = self.task.empty_state(phase=0)

        first_player._reset_mcts()
        second_player._reset_mcts()

        it = 0
        first_player_number = board.next_player

        while not self.task.is_terminal_state(board):
            it += 1
            if verbose:
                print(f"Turn {it} Player {board.next_player}")
                print(self.task.state_string_representation(board))

            temp = it < self.args.tempThreshold
            # can't count on consistent player numbers
            # or even consisten player counts
            if board.next_player == first_player_number:
                #print("FIRST PLAYER")
                move = first_player.get_move(board, temp)
                board = self.task.apply_move(move, board)
            else:
                #print("SECOND PLAYER")
                move = second_player.get_move(board, temp)
                board = self.task.apply_move(move, board)

        winners = self.task.get_winners(board)
        if verbose:
            print(f"Game Over: Turn {it} Winners {winners}")
            print(self.task.state_string_representation(board))
        r = 0
        if first_player_number in winners:
            r = 1
        elif first_player_number not in winners and len(winners)>0:
            r = -1

        return r

    def _arena_play(self, num, verbose=True):
        """
        Plays num games in which player 1 and 2 both start num/2 times each

        Parameters
        ----------
            num : int

        Returns
        -------
            oneWon : int
                games won by player1
            twoWon : int
                games won by player2
            draws:
                games won by nobody
        """
        num = int(num/2)
        eps_time = Progbar(2*num, stateful_metrics=["wins", "draws", "losses"])
        eps_time.update(0, values=[("wins",0),("draws",0),("losses",0)])
        oneWon = 0
        twoWon = 0
        draws = 0

        for _ in range(num):
            #import pdb; pdb.set_trace()
            result = self._arena_play_once(self, self.pnet, verbose=verbose)
            if result == 1:
                oneWon += 1
            elif result == -1:
                twoWon += 1
            else:
                draws += 1
            eps_time.add(1, [("wins", oneWon), ("losses", twoWon), ("draws", draws)])

            result = self._arena_play_once(self.pnet, self, verbose=verbose)
            if result == 1:
                twoWon += 1
            elif result == -1:
                oneWon += 1
            else:
                draws += 1
            eps_time.add(1, [("wins", oneWon), ("losses", twoWon), ("draws", draws)])

        return oneWon, twoWon, draws

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

        self._run_selfplay()

        # shuffle examlpes before training
        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        random.shuffle(trainExamples)

        # training new network, keeping a copy of the old one
        self.save_weights('temp.pth.tar')
        self.pnet.load_weights('temp.pth.tar')
        self.pnet._reset_mcts()

        self._train_nnet(trainExamples)
        #print(self.trainExamplesHistory)
        self._reset_mcts()

        log.info('PITTING AGAINST PREVIOUS VERSION')
        nwins, pwins, draws = self._arena_play(self.args.arenaCompare)

        log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
            log.info('REJECTING NEW MODEL')
            self.load_weights('temp.pth.tar')
        else:
            log.info('ACCEPTING NEW MODEL')
            self.save_weights(self._get_checkpoint_filename(self.iteration))
            self.save_weights('best.pth.tar')

        self.iteration += 1

    def test_once(self):
        """
        Runs a single testing interation. Does not change weights, and usually
        does not change the history either.

        Returns
        -------
        score : float
            ELO, win percentage, or another number where higher is better
        """

        return None
