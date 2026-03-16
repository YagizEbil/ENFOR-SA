import torch
import sys
import os
import random
import cProfile
import io
import pstats
import random


# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.experiment.experiment_sequential as exp_seq
import src.experiment.experiment as exp

import src.utils.dataset_loader as dataloader
import src.definitions as defs
import src.gemmini.gemmini_config as conf
import src.sim_options as simopt
import src.utils.utils as u
import src.experiment.logger as logger

from src.models.instrumented_model import InstrumentedModel
from src.models.base_model import BaseModel

# How to generate the call stack for python apps
# 1. python -m cProfile -o output.prof main.py
# 2. gprof2dot -f pstats output.prof | dot -Tsvg -o output.svg 
#  python -m cProfile -s tottime -o output.prof main.py 4


def profile_this():
    model_golden = BaseModel(defs.MODEL_NAME)
    model_faulty = InstrumentedModel(defs.MODEL_NAME)

    experiment = exp_seq.ExperimentSequential(
        model_faulty, 
        model_golden) 

    input_indices = u.random_indices(
        size=defs.BATCHES*defs.BATCH_SIZE, 
        max=dataloader.dataset_len-1)

    critical_faults = experiment.run_experiment(input_indices, trials=defs.INJECTIONS);


def main():

    simopt.parse_args()

    print(f"{u.Co['fg'][65]}[Setting threads]:{u.R}\n- Inter: {defs.N_INTER_THREADS}\n- Intra: {defs.N_INTRA_THREADS}\n")
    torch.set_num_interop_threads(defs.N_INTER_THREADS)
    torch.set_num_threads(defs.N_INTRA_THREADS)

    random.seed(defs.SEED)

    dataloader.load_dataset_imagenet()

    logger.SKIP_LOGS = True

    command = (
        "profile_this();"
    )


    """
    Prints top N methods, sorted by time.
    Equivalent to:
    python -m cProfile -o data/profile.txt main.py -n 100
    Options:
    time, cumulative, line, name, nfl, calls
    -----------
    ncalls - for the number of calls.
    time/tottime - for the total time spent in the given function
    (and excluding time made in calls to sub-functions)
    cumulative/cumtime - is the cumulative time spent in this and all subfunctions
    (from invocation till exit). This figure is accurate even for recursive functions.
    """

    profile_file = f"./profiler/prof/{defs.MODEL_NAME}-{conf.CONFIG_KEY}.prof"
    sort = "time"

    cProfile.run(command, filename=profile_file, sort=sort)
    s = io.StringIO()
    stats = pstats.Stats(profile_file, stream=s)
    stats.sort_stats(sort).print_stats(100)

    with open(f'./profiler/txt/{defs.MODEL_NAME}-{conf.CONFIG_KEY}.txt', 'w') as f:
        f.write(s.getvalue())

if __name__ == '__main__':
    main()


