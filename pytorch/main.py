import torch
import sys
import random

import src.definitions as defs
from src.models.instrumented_model import InstrumentedModel
from src.models.base_model import BaseModel
import src.utils.dataset_loader as dataloader
import src.experiment.experiment_sequential as exp_seq
import src.experiment.experiment_parallel as exp_par
import src.experiment.experiment as exp
import src.sim_options as simopt
import src.utils.utils as u
import src.toy_inputs as ti

def init():
    print(f"{u.Co['fg'][65]}[Setting threads]:{u.R}")
    print(f"- Inter:  {defs.N_INTER_THREADS}")
    print(f"- Intra:  {defs.N_INTRA_THREADS}")
    print(f"- Device: {defs.DEVICE}\n")

    #torch.use_deterministic_algorithms(True)
    torch.set_num_interop_threads(defs.N_INTER_THREADS)
    torch.set_num_threads(defs.N_INTRA_THREADS)

    random.seed(defs.SEED)
    torch.manual_seed(defs.SEED)
    torch.cuda.manual_seed(defs.SEED)
    dataloader.load_dataset_imagenet()


def main():
    init()

    model_golden = BaseModel(defs.MODEL_NAME)
    model_faulty = InstrumentedModel(defs.MODEL_NAME)
    
    if defs.TREE_FI_MODE:
        experiment = exp_par.ExperimentParallel(
            model_faulty=model_faulty, 
            model_golden=model_golden
            )
    else:
        experiment = exp_seq.ExperimentSequential(
            model_faulty=model_faulty, 
            model_golden=model_golden
            ) 

    #generates a random set of indices. inputs are loaded from the dataset based on the indices
    input_indices = u.random_indices(
        size=defs.BATCHES*defs.BATCH_SIZE, 
        max=dataloader.dataset_len-1
        )

    #input_indices = ti.toy_inputs_low_avf[0:8]

    critical_faults = experiment.run_experiment(input_indices)

    return 0


if __name__ == '__main__':
    u.register_sig_handlers()
    sys.exit(main())

