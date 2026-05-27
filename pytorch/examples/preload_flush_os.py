import sys
import os
import torch

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.gemmini.gemmini_extension_definitions as ext
import src.gemmini.gemmini_config as conf
import src.definitions as defs


# Available configs
#
# OS configs
#
#CONFIG_KEY = "OSDIM4"
CONFIG_KEY = "OSDIM8"
#CONFIG_KEY = "OSDIM16"
#CONFIG_KEY = "OSDIM32"
#CONFIG_KEY = "OSDIM64"

defs.ENABLE_GL_FAULT_MODEL = False

#
# Loads the Gemmini module - the ahead-of-time extension to interface with the verilated Gemmini module (this lib is designed in /rtl/lib/Gemmini)
#
gemmini = ext.load_extension(CONFIG_KEY)
gemmini.init()
gemmini.print_info()

DIM = conf.DIM
INPUT_TYPE, OUTPUT_TYPE = conf.GEMM_INPUT_DTYPE, conf.GEMM_OUTPUT_DTYPE
MIN_INT, MAX_INT = -128, 127

D = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
C_ref = D.clone() 

C = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)

steps = gemmini.preload(D)
steps = gemmini.flush_gemm(C, False)

print(torch.equal(C, C_ref))