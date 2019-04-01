from rlait.task.Go import Go
from rlait.approach.Random import Random
from rlait.approach.InteractivePlayer import InteractivePlayer
from run_test import main

def run():
    task = Go(board_size=5)
    ai1 = InteractivePlayer().init_to_task(task)
    ai2 = Random().init_to_task(task)

    main(task, ai1, ai2)

if __name__ == '__main__':
    run()
