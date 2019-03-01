#! /usr/bin/env python3

# only use 1 gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from rlait.task.Othello import Othello
from rlait.approach.QLearning import QLearning
from rlait.approach.AlphaZero import AlphaZero
from rlait.util import dotdict, BadMoveException
import run_test

import sys
sys.setrecursionlimit(10000)

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

total_az_iterations = 30
start_from_az_iteration = 10

total_ql_iterations = 3000
start_from_ql_iteration = 0

def run():
    task = Othello(4)
    az = AlphaZero({
        "numEps": 100,
        "numMCTSSims": 3,
        "maxDepth": 300,
        "arenaCompare": 10,
        "startFromEp": start_from_az_iteration,
        "load_checkpoint": True,
        "checkpoint": "checkpoint_28.pth.tar",
        "checkpoint_dir": "./az_checkpoints",
        "prevHistory": "checkpoint_10.pth.tar.examples",
    })

    ql = QLearning({
        "checkpoint_dir": "./ql_checkpoints",
    })

    az.init_to_task(task)
    ql.init_to_task(task)
    ql.load_weights("checkpoint_1.pkl")

    for x in range(3):
        run_test.main(task, az, ql)
        run_test.main(task, ql, az)
"""
    for i in range(start_from_ql_iteration, total_ql_iterations):
        ql.train_once()

    ql.save_weights("checkpoint_1.pkl")

    for i in range(start_from_az_iteration, total_az_iterations):
        az.train_once()

    print("Done training!")
"""
if __name__ == "__main__":
    run()
