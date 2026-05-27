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
STREAM_SIZE = conf.CONFIG_PARAMS[CONFIG_KEY]["dim"] + 4 

assert(STREAM_SIZE >= conf.DIM)

"""
 Test the computation of C = A1*B1 + A2*B2 + A3*B3 + A4*B4 + D 
"""
A1 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, STREAM_SIZE]), dtype=torch.int)
A2 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, STREAM_SIZE]), dtype=torch.int)
A3 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, STREAM_SIZE]), dtype=torch.int)
A4 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, STREAM_SIZE]), dtype=torch.int)
B1 = torch.randint(MIN_INT, MAX_INT, torch.Size([STREAM_SIZE, conf.DIM]), dtype=torch.int)
B2 = torch.randint(MIN_INT, MAX_INT, torch.Size([STREAM_SIZE, conf.DIM]), dtype=torch.int)
B3 = torch.randint(MIN_INT, MAX_INT, torch.Size([STREAM_SIZE, conf.DIM]), dtype=torch.int)
B4 = torch.randint(MIN_INT, MAX_INT, torch.Size([STREAM_SIZE, conf.DIM]), dtype=torch.int)

D = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
C = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)

steps = gemmini.preload(D)
steps = gemmini.stream(A1, B1)  # Note: the partial sums are kept stored in the PE accumulators 
steps = gemmini.stream(A2, B2) 
steps = gemmini.stream(A3, B3)
steps = gemmini.stream(A4, B4) 
steps = gemmini.flush_gemm(C, False)

C_ref = torch.mm(A1,B1) + torch.mm(A2,B2) + torch.mm(A3,B3) + torch.mm(A4,B4) + D

print(torch.equal(C, C_ref))