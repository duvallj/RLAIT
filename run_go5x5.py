# only use 1 gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from rlait.task.Go import Go
from rlait.approach.Random import Random
from rlait.approach.InteractivePlayer import InteractivePlayer
import run_test

az_train_iterations = 100
az_start_iteration  = 0

def run_interactive():
    task = Go(board_size=5)
    ai1 = InteractivePlayer().init_to_task(task)
    ai2 = Random().init_to_task(task)

    run_test.main(task, ai1, ai2)

def train_alphazero():
    task = Go(board_size=5)
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


if __name__ == '__main__':
    train_alphazero()
