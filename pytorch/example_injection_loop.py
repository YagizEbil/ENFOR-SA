import torch
import random
from src.models.instrumented_model import InstrumentedModel
from src.models.base_model import BaseModel
import src.utils.dataset_loader as dataloader
import src.definitions as defs
from src.conv import cached_tensors as tcache
from src.flist.fl import fl
import src.gemmini.gemmini_config as c
from src.utils import utils as u
from src import sim_options as opt

# parse args
args = opt.parse_args()

# loads a golden, reference model
model_golden = BaseModel(args.model).to(defs.DEVICE)

# loads the instrumented model for fault injections
model_faulty = InstrumentedModel(args.model).to(defs.DEVICE)

# gets a handle to iterate over the dataset
val_loader = dataloader.load_dataset_imagenet(batch_size=args.bsize)

# customize the desired fault targets. this custom filter will be valid for RTL injections only
fault_target = {
    #
    # the target conv layer (QuantizedConv2d, QuantizedConvReLU2d, etc...)
    #
    'layer': args.layer,
    
    #
    # target PE position
    #
    'pe_col': 0,
    'pe_row': 0, #targets only PEs of row 0
    #'pe_col':(u.ANY, [1, 2, 3]), #targets all columns except for columns 0 and 2

    #
    # the target bit in each PE
    #
    #'bit': (u.ANY, [5, 6, 7]),  # inject only bits 5, 6, and 7
    #'bit': (u.ANY, [0, 1, 2]),  # inject only bits 0, 1, and 2

    #
    # the target signal in each PE (inputs, outputs or ctrl signals)
    #
    #'target': (u.ANY, [c.IN_A, c.IN_B, c.OUT_C]), # inject only in the PE inputs and outputs
    #'target': (u.ANY, [c.SIG_PROPAG, c.SIG_VALID]), # inject only in the control signals
}

# the filter can also be empty
#fault_target = {}

# important: assure the 'layer' filter matches exaclty what was passed as args
fault_target['layer'] = args.layer

# loads the fault list with the desired targets. this will load the rows [0, args.injections-1]
fault_list = fl.load_fault_list(
    args.faultlist, 
    (0, args.injections-1), 
    filters=fault_target
    )

# loop over the dataset
for i, (inputs, gt_target) in enumerate(val_loader):
    if i == args.batches:
        break

    # ViT models can run on CUDA
    inputs = inputs.to(defs.DEVICE, non_blocking=True)
    
    # the ground truth label. this could be used to compute the model accuracy
    gt_target = gt_target.to(defs.DEVICE, non_blocking=True)

    # runs the golden inference. returns the output logits
    logits_golden = model_golden.run_inference(inputs)

    # compute the golden softmax probabilities over the logits
    probabilities_golden = torch.nn.functional.softmax(logits_golden, dim=1)

    # extracts the computed golden labels
    predicted_labels_golden = torch.argmax(probabilities_golden, dim=1)
    
    # per-batch counters
    critical_faults, total_sdcs = 0, 0

    # runs the faulty mode by iterating over each fault in the list
    for fault in fault_list:
        # faulty inference run
        logits_faulty = model_faulty.run_inference(inputs, fault=fault)
        
        # compute the faulty softmax probabilities over the logits
        probabilities_faulty = torch.nn.functional.softmax(logits_faulty, dim=1)
        
        # extracts the computed faulty labels
        predicted_labels_faulty = torch.argmax(probabilities_faulty, dim=1)

        # count the number of inputs for which the faulty model != golden reference
        count_inputs_failed = (predicted_labels_faulty != predicted_labels_golden).sum().item()

        # the fault is critical if at least one input failed
        critical_faults += count_inputs_failed != 0

        # the total number of inputs with faults
        total_sdcs += count_inputs_failed

    print(f"Finished batch {i} with {critical_faults} critical faults ({100*critical_faults/args.injections:.2f}%)")

    # fault injections run on repeated inputs/layers. we store the conv outputs in LUTs to improve injection time. 
    # the LUTs must be cleared whenever new inputs are to be simulated
    tcache.clear_luts()