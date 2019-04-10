import multiprocessing
import os, sys

def parallelize(function, num_procs=4, runs_per_proc=1, chunksize=1):
    """
    Runs a function in `num_procs` parallel processes, `runs_per_proc` times
    per process. Automatically concatenates all return values of the function,
    if all a built-in `dict` or `list`, or returns an unordered list
    of all the return values otherwise.

    Arguments
    ---------
    function : Runnable
        The function to run. Must take no arguments.
    num_procs : int (4)
    runs_per_proc : int (1)
    chunksize : int(1)
        How large of chunks to chop the iterable into when submitting the
        job to the process bool. For small runs, is OK to leave at 1

    Returns
    -------
    output : (variable)
        Type depends on function return type, see description above.
    """

    assert callable(function)
    assert type(num_procs) is int and type(runs_per_proc) is int
    assert num_procs > 0 and runs_per_proc > 0

    results = None
    with multiprocessing.Pool(num_procs) as p:
        results = p.map(function, range(num_procs*runs_per_proc), chunksize)

    full_results = None
    if all(type(x) is dict for x in results):
        full_results = dict()
        for x in results:
            full_results.update(x)
    elif all(type(x) is list for x in results):
        full_results = list()
        for x in results:
            full_results.extend(x)
    else:
        # not concatenating tuples
        full_results = results

    return full_results
