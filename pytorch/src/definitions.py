import os

#
# Full path to the ImageNet dataset
#
PATH_IMAGENET = os.environ.get("PATH_IMAGENET")

if PATH_IMAGENET is None:
    raise RuntimeError("PATH_IMAGENET is not set. Run 'export PATH_IMAGENET=<Path to ImageNet>'")

"""
    Fixed default options. These options are overwritten in src/flist/fl.py. do not touch them here
"""    

#
# Fauls in Gemmini (if False, faults are injected in SW only)
#
FI_GEMM = True

#
# Use VIT models
#
VIT = False

#
# use CUDA only for the ViT models iff there's a GPU available
#
CUDA = False

#
# Only ViT models run on GPUs iff CUDA is available
#
DEVICE = "cuda" if CUDA else "cpu"

#
# SW fault injection mode. just an inverted alias...
#
FI_SW = not FI_GEMM

ENABLE_GL_FAULT_MODEL = False
TREE_FI_MODE = False

#
# Available CNN and ViT models
#
MODEL_NAME = "ResNet18"
#MODEL_NAME = "ResNet50"
#MODEL_NAME = "ResNeXt101_32X8D"
#MODEL_NAME = "ResNeXt101_64X4D"
#MODEL_NAME = "MobileNet_V2"
#MODEL_NAME = "MobileNet_V3_Large"
#MODEL_NAME = "GoogLeNet"    # QuantizedConvReLU2d, no conv2d
#MODEL_NAME = "Inception_V3" # QuantizedConvReLU2d, no conv2d
#MODEL_NAME = "ShuffleNet_V2_X0_5"
#MODEL_NAME = "ShuffleNet_V2_X1_0"
#MODEL_NAME = "ShuffleNet_V2_X1_5"
#MODEL_NAME = "ShuffleNet_V2_X2_0"
#MODEL_NAME = "deit_tiny"
#MODEL_NAME = "deit_small"
#MODEL_NAME = "deit_base"

# An alias to the campaign. this is attatched to the report file names
CAMP_ALIAS = "experiment-xxx"

# The fault list. This must reside in the path fault_lists/<model>
# Fault lists can be generated with the script gen_fault_list.py
FAULT_LIST = "fl_os_dim_8.csv" if FI_GEMM else "fl_sw.csv"

# Destination path to the report file names
EXP_FOLDER = "reports/ResNet18/experiment-xxx/sequential"

# The number of batches
BATCHES = 1

# The number of inputs for each batch
BATCH_SIZE = 8

# The target layer for injection
TARGET_LAYER = 0

# The number of injections per layer per input
INJECTIONS = 10

# Simulation seed
SEED = 0

# Skips fault injections
RUN_GOLDEN_MODE = False

# configures the number of PyTorch threads - https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
# inter op: One or more inference threads execute a model’s forward pass on the given inputs.
# intra op: Each thread in the forward pass further forks the computation into multiple threads (e.g., a dot product can be split into multiple threads)
N_INTER_THREADS = 2
N_INTRA_THREADS = 2