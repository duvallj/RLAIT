#! /usr/bin/env python3

# only use 1 gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from rlait.task.Stratego import Stratego
from rlait.approach.AlphaZero import AlphaZero
from rlait.util import dotdict, BadMoveException

import sys
sys.setrecursionlimit(10000)

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

total_iterations = 30
start_from_iteration = 0

def run():
    N = 10
    task = Stratego(N, max_moves=650)
    ai = AlphaZero({
        "numEps": 6,
        "numMCTSSims": 3,
        "maxDepth": 300,
        "arenaCompare": 6,
        "startFromEp": start_from_iteration,
        "load_checkpoint": True,
        "prevHistory": "checkpoint_0.pth.tar.examples",
    }).init_to_task(task)

    for i in range(start_from_iteration, total_iterations+1):
        ai.train_once()

    print("Done training!")

if __name__ == "__main__":
    run()
