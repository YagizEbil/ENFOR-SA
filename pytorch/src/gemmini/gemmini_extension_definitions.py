import os
import torch.utils.cpp_extension as cpp_extension
from src.gemmini import gemmini_config as conf
from src import definitions as defs

# This file defines the parameters to compile the C++ extensions of the Gemmini interface (e.g., in gemmini = cpp_extension.load(...))

#
# ROOT_PATH: The root path of the framework
#
ROOT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "..")

#
#  VERILATED_MODULE: Verilated modules - points to the archive (.a) of the verilated design
#
VERILATED_MODULE = 'VMesh__ALL.a'    # The Gemmini module

#
# The GL design has a different DUT name:
#
MAC_MODULE=f"{ROOT_PATH}/rtl/verilated/MacUnit_netlist_hdfit/VMacUnit_netlist_hdfit__ALL.a"

#
# VERILATED_PATH: The base path of the verilated designs (e.g, where verilator creates the verilated designs)
#
VERILATED_PATH = f"{ROOT_PATH}/rtl/verilated/"

#
# The verilator include path (v4.224 available in this repo)
#
VERILATOR_INC_PATH = f"{ROOT_PATH}/rtl/lib/verilator-include/include"

#
# all Gemmini config keys
#
CONFIG_KEYS = [k for k in conf.CONFIG_PARAMS.keys()]

#
# all available paths in the "verilated" folder
#
DESIGN_PATHS = [cfg["path"] for cfg in conf.CONFIG_PARAMS.values()]

#
# the verilator.o object file of all configs (this requires that the verilated dir folder names match exactly the names in conf.configs)
#
VERILATED_OBJ_DIR = [os.path.join(VERILATED_PATH, d, "verilated.o") for d in DESIGN_PATHS] # = VERILATED_PATH + VERILATED_OBJ_DIR + "/verilated.o"

#
# the verilated_threads.o object file (if using multi-threaded verilator)
#
#VERILATED_MT_OBJ_DIR = [os.path.join(VERILATED_PATH, d, "verilated_threads.o") for d in conf.VERILATED_DIRS] # only for mt verilator

#
# the VMesh__ALL.a file of all configs
#
VERILATED_ARCH = [os.path.join(VERILATED_PATH, d, VERILATED_MODULE) for d in DESIGN_PATHS] # = VERILATED_PATH + VERILATED_OBJ_DIR + "/" + VERILATED_MODULE

# if using single-threaded verilator
CONFIGS_ST = { key: [obj, arch] for key, obj, arch in zip(CONFIG_KEYS, VERILATED_OBJ_DIR, VERILATED_ARCH) }

# if using multi-threaded verilator
#CONFIGS_MT = { key: [obj, obj_mt, arch] for key, obj, obj_mt, arch in zip(CONFIG_KEYS, VERILATED_OBJ_DIR, VERILATED_MT_OBJ_DIR, VERILATED_ARCH) }


# ------------ Compiler flags ------------ #

#
# INCLUDE_FLAG: The flag to include the path to verilated.h
#
INCLUDE_FLAG = f"-I/{VERILATOR_INC_PATH}"

# if using multi-threaded verilator
#OPT_FLAG_MT = "-O3 -std=c++17 -DVL_THREADED"

# two options (single- and multi-threaded verilator)
CONFIGS = CONFIGS_ST

# Enable this for multi-threaded verilator
#CONFIGS = CONFIGS_MT

# ------------ Interface definitions ------------ #
# The extension name
EXT_NAME = "gemmini_interface"

#The interface sources
FILE_SER = "./src/gemmini/cpp/gemmini_interface.cpp"
#FILE_PAR = "src/cpp/gemmini_interface_parallel.cpp"

SRC_SER = [FILE_SER]
#SRC_PAR = [FILE_PAR]

#
# Compiles and loads the C++ extension with the required config_key Gemmini config
#
def load_extension(config_key):
    conf.set_config(config_key)

    # Enable this for multi-threaded verilator
    #OPT_FLAG = f"-O3 -D{conf.CONFIG_KEY} -std=c++17 -DVL_THREADED"

    #
    # OPT_FLAG: Opt flag for the main interface file (gemmini_interface.cpp) and the files in the Gemmini "headers" folder)
    #
    if conf.GEMM_MODE == conf.MODE_OS:
        OPT_FLAG = f"-O3 -DGEMM_OS -DCONFIG={conf.DESIGN_PATH}"
    else:
        OPT_FLAG = f"-O3 -DGEMM_WS -DCONFIG={conf.DESIGN_PATH}"

    objectFiles = CONFIGS[config_key]

    # attach -DUSE_GL_INJECTION if GL is True
    if defs.ENABLE_GL_FAULT_MODEL:
        OPT_FLAG = f"{OPT_FLAG} -DUSE_GL_INJECTION"
        objectFiles.append(MAC_MODULE)

    FLAGS = [OPT_FLAG, INCLUDE_FLAG]

    """
    PyTorch computes a hash based on:
        sources
        compiler version
        PyTorch version 
        all compiler flags (cflags + ldflags)

    If any of these change — even just the extra_ldflags string — PyTorch creates a new hash directory, and recompiles everything from scratch.
    This way, if we use a single extension name for each config, we can keep individual cached bindings for each config
    """
    return cpp_extension.load(
        name=f"{EXT_NAME}_{config_key}_{defs.ENABLE_GL_FAULT_MODEL}", # attach defs.ENABLE_GL_FAULT_MODEL, so we can easily alternate between rtl/gl cases for the same config
        sources=SRC_SER, 
        extra_cflags=FLAGS, 
        extra_ldflags=objectFiles, 
        verbose=False)
