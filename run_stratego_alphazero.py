from rlait.task.stratego.Stratego import Stratego
from rlait.approach.alphazero.AlphaZero import AlphaZero
from rlait.util import dotdict, BadMoveException

import sys
sys.setrecursionlimit(100000)

total_iterations = 30
start_from_iteration = 0

def run():
    N = 10
    task = Stratego(N)
    ai = AlphaZero({
        "numEps": 6,
        "numMCTSSims": 3,
        "startFromEp": start_from_iteration,
        "load_checkpoint": False,
    }).init_to_task(task)

    for i in range(start_from_iteration, total_iterations+1):
        ai.train_once()

    print("Done training!")

if __name__ == "__main__":
    run()
