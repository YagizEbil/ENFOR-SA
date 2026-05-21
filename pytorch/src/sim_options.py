import argparse
import torch
from src import definitions as defs
from src.gemmini import gemmini_config as conf
from src.utils import utils as u


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation options")

    # [General options]
    parser.add_argument("-m", "--model",  type=str, default=defs.MODEL_NAME, help="The DNN model name. See available options in src/models/common.py")
    parser.add_argument("-c", "--config", type=str, default=conf.CONFIG_KEY, help="The Gemmini configuration (possible cases listed in gemmini_config.py)")
    parser.add_argument("-o", "--output", type=str, default=defs.EXP_FOLDER, help="The report output folder)")

    # [Fault injection options]
    parser.add_argument("-b", "--bsize",      type=int, default=defs.BATCH_SIZE,   help="The batch size")
    parser.add_argument("-B", "--batches",    type=int, default=defs.BATCHES,      help="The number of batches")
    parser.add_argument("-l", "--layer",      type=int, default=defs.TARGET_LAYER, help="The target injection layer")
    parser.add_argument("-n", "--injections", type=int, default=defs.INJECTIONS,   help="The number of fault injections (sampled from the fault list)")
    parser.add_argument("-f", "--faultlist",  type=str, default=defs.FAULT_LIST,   help="The fault list file. It must be stored in fault_lists/<model>")
    parser.add_argument("-F", "--fmodel",     type=str, default="rtl", choices=["sw", "rtl", "gl"], help="The fault model")

    # removed for open source
    parser.add_argument("-t", "--tree",  action='store_true', default=defs.TREE_FI_MODE, help="Use the tree fault injection mode")
    parser.add_argument("-p", "--prune", action='store_true', default=defs.PRUNE_INPUTS, help="Prune golden inputs in the fault tree mode")

    # other options
    parser.add_argument("-E", "--th_inter", type=int, default=defs.N_INTER_THREADS, help="The number inter_op threads")
    parser.add_argument("-A", "--th_intra", type=int, default=defs.N_INTRA_THREADS, help="The number intra_op threads")
    parser.add_argument("-a", "--alias",    type=str, default=defs.CAMP_ALIAS,      help="The simulation alias (will be the report file name)")

    parser.add_argument("-g", "--gold",  action='store_true', default=defs.RUN_GOLDEN_MODE, help="Run golden mode mode")
    parser.add_argument("-s", "--seed",  type=int, default=defs.SEED, help="A simulation seed")

    # [Debug options]
    #parser.add_argument("-d", "--ofm",  action='store_true', default=defs.DUMP_OFM,   help="Debug mode on/off")
    #parser.add_argument("-q", "--seqp", action='store_true', default=defs.RUN_PAR_FI, help="Debug: runs sequential fault list, multiple faults in parallel")
    #parser.add_argument("-i", "--input",  type=int, default=0, help="A toy input index (from imagenet indexes)")
    
    args = parser.parse_args()
    return args


#
# all ViT models available
#
vit_models = [
    "deit_tiny", "deit_small", "deit_base",
    "swin_tiny", "swin_small", "swin_base",
    "vit_base", "vit_large"]
              

def set_definitions(args):

    # removed for open source
    #if args.seqp and args.tree:
        #print("[Error]: cannot run with args <--seqp> and <--tree> set at the same time")
        #exit(0)

    if args.model in ["swin_tiny", "swin_small", "swin_base"]:
        print("Swin models are not supported yet!")
        exit(0)

    print(f"{u.Co['fg'][65]}\n[Running with args]:{u.R}")

    for key, value in vars(args).items():
        print(f"-{key} {u.Co['fg'][95]}{value}{u.R} ", end='')
   
    print("\n")

    # overwrites the default options with the parsed arguments
    conf.CONFIG_KEY = args.config
    defs.MODEL_NAME = args.model
    defs.RUN_GOLDEN_MODE = args.gold
    defs.TARGET_LAYER = args.layer
    defs.EXP_FOLDER = args.output # e.g., reports/ResNet18/exp-v1/sequential (the whole path before 'trace' or 'batch')
    defs.SEED = args.seed
    defs.BATCH_SIZE = args.bsize
    defs.BATCHES = args.batches
    defs.INJECTIONS = args.injections
    defs.FAULT_LIST = args.faultlist
    defs.CAMP_ALIAS = f"{args.alias}"

    defs.TREE_FI_MODE = args.tree
    defs.PRUNE_INPUTS = args.prune

    defs.N_INTER_THREADS = args.th_inter
    defs.N_INTRA_THREADS = args.th_intra

    n_th_inter_max = torch.get_num_interop_threads()
    n_th_intra_max = torch.get_num_threads()

    if args.th_inter > n_th_inter_max:
        print(f"{u.C['red']}Warning: excessive number of inter op threads ({args.th_inter}). Setting it to max ({n_th_inter_max}){u.C['reset']}")
        defs.N_INTER_THREADS = n_th_inter_max

    if args.th_intra > n_th_intra_max:
        print(f"{u.C['red']}Warning: excessive number of intra op threads ({args.th_intra}). Setting it to max ({n_th_intra_max}){u.C['reset']}")
        defs.N_INTRA_THREADS = n_th_intra_max

    #
    # check if we need Gemmini for RTL injection or not
    #
    defs.FI_GEMM = True if args.fmodel in ["rtl", "gl"] else False
    #defs.FI_GEMM = True if args.fmodel in ["rtl", "gl"] else False

    #
    # check if the model is any transformer model, and asserts defs.VIT if True
    #
    defs.VIT = True if args.model in vit_models else False

    #
    # use CUDA only for the ViT models iff there's a GPU available
    #
    defs.CUDA = torch.cuda.is_available() if defs.VIT else False

    #
    # use cuda if we can (ViTs only)
    #
    defs.DEVICE = "cuda" if defs.CUDA else "cpu"

    #
    # SW fault injection mode. just an inverted alias...
    #
    defs.FI_SW = not defs.FI_GEMM

    # removed for open source
    defs.ENABLE_GL_FAULT_MODEL = True if args.fmodel == "gl" else False

