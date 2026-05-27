import sys
import os
import torch
import random

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.gemmini.gemmini_extension_definitions as ext
import src.gemmini.gemmini_config as conf
import src.definitions as defs
from src.gemmini import gemmini_utils as gu
from src.gemmini.gemmini_config import *

# Available configs
#
# OS configs
#
#CONFIG_KEY = "OSDIM4"
CONFIG_KEY = "OSDIM8"
#CONFIG_KEY = "OSDIM16"
#CONFIG_KEY = "OSDIM32"
#CONFIG_KEY = "OSDIM64"

#
# WS configs
#
#CONFIG_KEY = "WSDIM4"
#CONFIG_KEY = "WSDIM8"
#CONFIG_KEY = "WSDIM64"

defs.ENABLE_GL_FAULT_MODEL = False # note, only logic-level masking is modeled

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


# for gate-level faults (logic masking only)
MAC_UNIT_HDFIT_FIRST_TARGET_ID = 2  # First one is assign _0056_[16] =( io_in_a[8] & io_in_b[8]) ^ ((fiEnable && (2 == GlobalFiNumber)) ? GlobalFiSignal[0] : {1{1'b0}});
MAC_UNIT_HDFIT_LAST_TARGET_ID  = 824 # 8/32-bit inputs/outputs (signed mac)

if CONFIG_KEY in ["WSDIM4", "WSDIM8", "WSDIM64"]: # as in WS we preload the B tensor, we must assure the stream size is same size of the SA
    STREAM_SIZE = conf.CONFIG_PARAMS[CONFIG_KEY]["dim"]
else:
    STREAM_SIZE = conf.CONFIG_PARAMS[CONFIG_KEY]["dim"] + 4 


TARGET_SIGNALS = {
    # data signals  - PE inputs
    "IN_A":  (IN_A, PE_IN_BITS),  # input A signal id is 0, with PE_IN_BITS bits
    "IN_B":  (IN_B, PE_IN_BITS),  # input B signal id is 1, with PE_IN_BITS bits

    # data signals - PE outputs
    "OUT_B": (OUT_B, PE_OUT_BITS),  # WS: this would be the partial sum flowing downstream
    "OUT_C": (OUT_C, PE_OUT_BITS),  # WS: no effect. the outputs are streamed through out_b. OS: affects the accumulators

    # data signals - each PE has two registers to store: 1. accumulators in OS, or 2. weights in WS - in each case, only one reg. is actually used
    "C1":   (C1, PE_OUT_BITS), # OS: faults (must use WILL_PE_INPUT_BE_ASSIGNED so the fault is not overwritten) WS: no faults (preloaded in C2)
    "C2":   (C2, PE_OUT_BITS), # OS: no faults (preloaded in C1)  WS: yes faults

    # control signals
    "SIG_PROPAG": (SIG_PROPAG, 1),
    "SIG_VALID":  (SIG_VALID, 1),
}

target = TARGET_SIGNALS['IN_A']

fiSilent = False


"""
 Test the computation of C = A*B + D with faults injected
"""
A = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, STREAM_SIZE]), dtype=torch.int)
B = torch.randint(MIN_INT, MAX_INT, torch.Size([STREAM_SIZE, conf.DIM]), dtype=torch.int)
D = torch.randint(MIN_INT, MAX_INT, torch.Size([conf.DIM, conf.DIM]), dtype=torch.int)
C = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)

C_ref = torch.mm(A,B) + D

tgt_sig = target[0]
bits = target[1]

pe_row = random.randint(0, DIM-1)
pe_col = random.randint(0, DIM-1)
tgt_bit = random.randint(0, bits)
ficycle = gu.get_pe_active_rand_cycle(pe_row, pe_col, DIM)
cell = random.randint(MAC_UNIT_HDFIT_FIRST_TARGET_ID, MAC_UNIT_HDFIT_LAST_TARGET_ID)
#pol = random.randint(0, 1) # for permanent faults only 

gemmini.clear_fault_list()

gemmini.add_transient_fault(
    tgt_sig, pe_row, pe_col, tgt_bit, ficycle, cell, fiSilent) 

steps = gemmini.preload(D)
steps = gemmini.stream(A, B)  # Note: the partial sums are kept stored in the PE accumulators 

steps = gemmini.flush_gemm(C, False)

print(torch.eq(C, C_ref))