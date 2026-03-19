#
# This is a test to validate our fault injection method against HDFIT
# the validation is done against a OS DIM8 config only.
# to enable HDFIT injections, we need set the HDFIT variables (GlobalFiModInstNr, GlobalFiNumber and GlobalFiSignal) according to the fault position and targets
# see the file /rtl/lib/Gemmini/HDFIT-fault-map-os-DIM8.txt, which is a sketch showing how this was done
#

import random 
import sys
import os
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as cpp_extension

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.gemmini.gemmini_extension_definitions as ext
import src.gemmini.gemmini_config as conf
import src.gemmini.gemmini_utils as gu
import src.definitions as defs


def validate_against_hdfit():

    # we validate our RTL injection approach vs. HDFT RTL approach
    #defs.ENABLE_GL_FAULT_MODEL = False

    TRIALS = 1000
    FLUSH_OUT_TRANSP = False
    FI_SILENT = True
    MIN_INT, MAX_INT = 1, 127 # i do not want any zeros in the inputs so that no faults are masked

    gemmini_enfor = ext.load_extension("OSDIM8")
    gemmini_hdfit = ext.load_extension("OSDIM8_HDFIT")
        
    gemmini_enfor.init()
    gemmini_hdfit.init()

    gemmini_enfor.print_info()
    gemmini_hdfit.print_info()

    DIM = conf.DIM

    A = torch.zeros((DIM, DIM), dtype=conf.GEMM_INPUT_DTYPE)
    B = torch.zeros((DIM, DIM), dtype=conf.GEMM_INPUT_DTYPE)
    C_enfor = torch.zeros((DIM, DIM), dtype=conf.GEMM_OUTPUT_DTYPE)
    C_hdfit = torch.zeros((DIM, DIM), dtype=conf.GEMM_OUTPUT_DTYPE)

    injections, errors_enfor, errors_hdfit, count_hdfit_mismatch = 0, 0, 0, 0

    # targer signals evaluated. signal name: (signal id, signal bits)
    TARGET_SIGNALS = {
        # data signals
        "IN_A":  (0, 8), # we're considering 8 bits for fault injection
        "IN_B":  (1, 8),
        "OUT_C": (4, 32), # the accumulator output

        # control signals
        "SIG_PROPAG":   (5, 1),
        "SIG_VALID":    (6, 1),
    }

    # tests all target signals
    for key, sig_id_bits_tuple in TARGET_SIGNALS.items():
        target   = sig_id_bits_tuple[0]
        sig_bits = sig_id_bits_tuple[1]

        for i in range(TRIALS):
            row = random.randint(0, DIM-1)
            col = random.randint(0, DIM-1)
            bit = random.randint(0, sig_bits-1)

            A.random_(MIN_INT, MAX_INT)
            B.random_(MIN_INT, MAX_INT)

            gemmini_enfor.clear_fault_list()
            gemmini_hdfit.clear_fault_list()

            ficycle = gu.pe_first_cycle(row, col) + 1 # the cycle hdfit injects

            gemmini_enfor.add_transient_fault(target, row, col, bit, ficycle, 0, FI_SILENT)
            gemmini_hdfit.add_transient_fault(target, row, col, bit, ficycle, 0, FI_SILENT)

            _ = gemmini_enfor.stream(A, B)
            _ = gemmini_hdfit.stream(A, B)
            
            _ = gemmini_enfor.flush_gemm(C_enfor, FLUSH_OUT_TRANSP)
            _ = gemmini_hdfit.flush_gemm(C_hdfit, FLUSH_OUT_TRANSP)

            C_gold = torch.mm(A, B)

            errors_enfor += int (not torch.equal(C_gold, C_enfor))
            errors_hdfit += int (not torch.equal(C_gold, C_hdfit))
            count_hdfit_mismatch += not torch.equal(C_enfor, C_hdfit)
            injections += 1

    print(f"Injections:   {injections}")
    print(f"ENFOR-SA: Output tensors with errors: {errors_enfor}")
    print(f"HDFIT:    Output tensors with errors  {errors_hdfit}")
    print(f"RESULT:   Trials not matching HDFIT:  {count_hdfit_mismatch}")




validate_against_hdfit()