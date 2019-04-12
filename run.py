# only use 1 gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from rlait.task.Go import Go
from rlait.approach.Random import Random
from rlait.approach.InteractivePlayer import InteractivePlayer
from rlait.util import dotdict, BadMoveException
from rlait.task.SimplestTask import SimplestTask

from time import sleep
import numpy as np

class run_test:
    def main(self, task, ai1, ai2):
        board = task.empty_state(phase=0)
        print(task.state_string_representation(board))

        first_player_number = board.next_player

        while not task.is_terminal_state(board):
            move = None
            if board.next_player == first_player_number:
                print("To move: {} ({})".format(board.next_player, ai1.approach_name))
                move = ai1.get_move(board)
            else:
                print("To move: {} ({})".format(board.next_player, ai2.approach_name))
                move = ai2.get_move(board)

            print(task.move_string_representation(move, board))
            board = task.apply_move(move, board)
            to_print = task.state_string_representation(board)
            print(to_print)

        print('The winner is player {}'.format(task.get_winners(board)))

    def play_noprint(self, task, ai1, ai2):
        board = task.empty_state(phase=0)

        first_player_number = board.next_player

        while not task.is_terminal_state(board):
            move = None
            if board.next_player == first_player_number:
                move = ai1.get_move(board)
            else:
                move = ai2.get_move(board)
            board = task.apply_move(move, board)

        winners = task.get_winners(board)
        if first_player_number in winners: return 1
        elif len(winners) != 0: return -1
        return 0

class GoRunner:

    az_train_iterations = 100
    az_start_iteration  = 0

    ql_train_iterations = 10
    ql_start_iteration = 0

    task = Go(board_size=5)

    def run_interactive(self):
        ai1 = InteractivePlayer().init_to_task(self.task)
        ai2 = Random().init_to_task(self.task)

        run_test.main(self.task, ai1, ai2)

    def train_alphazero(self):
        from rlait.approach.AlphaZero import AlphaZero
        az = AlphaZero({
            "numEps": 200,
            "startFromEp": az_start_iteration,
            "numMCTSSims": 40,
            "tempThreshold": 6,
            "maxDepth": 1000,
            "arenaCompare": 16,
            "load_checkpoint": False,
            "checkpoint": None,
            "prevHistory": None,
            "checkpoint_dir": "./go5_checkpoints",
        }).init_to_task(task)

        for x in range(az_start_iteration, az_train_iterations+1):
            az.train_once()

    def train_qlearning(self):
        from rlait.approach.QLearning import QLearning
        ql = QLearning({
            "checkpoint_dir": "./qlgo5_checkpoints",
            "lr": 0.2,
            "discount": 0.9,
            "num_procs": 4,
            "games_per_proc": 20,
        }).init_to_task(task)

        ql.load_weights("checkpoint_1.pkl")

        for x in range(ql_start_iteration, ql_train_iterations):
            ql.train_once()

        ql.save_weights("checkpoint_2.pkl")

class SimpleRunner:
    task = SimplestTask()

    def train(self):
        from rlait.approach.AlphaZero import AlphaZero
        az = AlphaZero({
            "numEps": 10,
            "startFromEp": 0,
            "numMCTSSims": 5,
            "tempThreshold": 0,
            "maxDepth": 100,
            "arenaCompare": 10,
            "load_checkpoint": False,
            "checkpoint": None,
            "prevHistory": None,
            "checkpoint_dir": "./simple_checkpoints",
        }).init_to_task(self.task)

        for x in range(10):
            az.train_once()

    def run_interactive(self):
        from rlait.approach.AlphaZero import AlphaZero
        ai1 = InteractivePlayer().init_to_task(self.task)
        ai2 = AlphaZero({
            "load_checkpoint": True,
            "checkpoint": "temp.pth.tar",
            "prevHistory": None,
            "checkpoint_dir": "./simple_checkpoints",
        }).init_to_task(self.task)

        run_test.main(None, self.task, ai1, ai2)


if __name__ == "__main__":
    a = SimpleRunner()
    a.train()
