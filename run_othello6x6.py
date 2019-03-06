#! /usr/bin/env python3

# only use 1 gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from itertools import product
import pickle

from rlait.util import dotdict, BadMoveException
import run_test

import sys
sys.setrecursionlimit(10000)

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

total_az_iterations = 100
start_from_az_iteration = 30

total_ql_iterations = 10
start_from_ql_iteration = 0

AZ_CHECKPOINTS = [3, 4, 6, 9, 10, 11, 13, 14, 15, 27]
QL_CHECKPOINTS = [1, 2, 3]

games_per_round = 13

def run():
    from rlait.task.Othello import Othello
    from rlait.approach.QLearning import QLearning
    from rlait.approach.AlphaZero import AlphaZero

    task = Othello(6)
    az = AlphaZero({
        "numEps": 100,
        "numMCTSSims": 20,
        "maxDepth": 300,
        "arenaCompare": 10,
        "startFromEp": start_from_az_iteration,
        "load_checkpoint": False,
        "checkpoint": "checkpoint_27.pth.tar",
        "checkpoint_dir": "./az6_checkpoints",
        "prevHistory": "checkpoint_29.pth.tar.examples",
    })

    ql = QLearning({
        "checkpoint_dir": "./ql6_checkpoints",
        "lr": 0.2,
        "discount": 0.9,
        "num_procs": 8,
        "games_per_proc": 100,
    })

    az.init_to_task(task)
    ql.init_to_task(task)

    train(ql, az)

def test(al, az):

    game_history = dict()

    for ql_n, az_n in product(QL_CHECKPOINTS, AZ_CHECKPOINTS):
        log.info("Now running QL({}) vs AZ({})".format(ql_n, az_n))
        ql.load_weights("checkpoint_{}.pkl".format(ql_n))
        az.load_weights("checkpoint_{}.pth.tar".format(az_n))

        az_deficit = 0

        for x in range(games_per_round):
            log.info("Game {}/{}".format(x+1, games_per_round))
            az_deficit += run_test.play_noprint(task, az, ql)
            az_deficit -= run_test.play_noprint(task, ql, az)

        game_history[(ql_n, az_n)] = az_deficit

        with open("t6x6-1.pkl", 'wb') as f:
            pickle.dump(game_history, f)

def train(ql, az):
    
    ql.load_weights("checkpoint_3.pkl")

    for i in range(start_from_ql_iteration, total_ql_iterations):
        if i%100==0: log.info("QLearning iteration: {}".format(i))
        ql.train_once()

    ql.save_weights("checkpoint_4.pkl")


    for i in range(start_from_az_iteration, total_az_iterations):
        az.train_once()

    log.info("Done training!")

if __name__ == "__main__":
    run()
