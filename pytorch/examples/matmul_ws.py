import sys
import os
import torch

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.gemmini.gemmini_extension_definitions as ext
import src.gemmini.gemmini_config as conf
import src.definitions as defs


#
# WS configs
#
#CONFIG_KEY = "WSDIM4"
CONFIG_KEY = "WSDIM8"
#CONFIG_KEY = "WSDIM64"

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


"""
    Test the computation of
    C1 = A1*B + D1
    C2 = A2*B + D2
    ...
    Cn = An*B * Dn
"""

A1 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
A2 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
A3 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
A4 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
D1 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
D2 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
D3 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
D4 = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)

B = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)

C1 = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)
C2 = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)
C3 = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)
C4 = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)

steps_pre = gemmini.preload(B)
steps_mm = gemmini.stream_bias(A1, D1, C1)
steps_mm = gemmini.stream_bias(A2, D2, C2)
steps_mm = gemmini.stream_bias(A3, D3, C3)
steps_mm = gemmini.stream_bias(A4, D4, C4)

C1_ref = torch.mm(A1, B) + D1
C2_ref = torch.mm(A2, B) + D2
C3_ref = torch.mm(A3, B) + D3
C4_ref = torch.mm(A4, B) + D4        

print(torch.equal(C1_ref, C1))
print(torch.equal(C2_ref, C2))
print(torch.equal(C3_ref, C3))
print(torch.equal(C4_ref, C4))