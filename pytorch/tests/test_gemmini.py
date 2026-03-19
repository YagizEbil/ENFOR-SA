#
# test cases for the Gemmini operations
#

import unittest
import inspect
import random 
import statistics
import sys
import os
import time
import torch
import torch.nn.functional as F

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.gemmini.gemmini_extension_definitions as ext
import src.gemmini.gemmini_config as conf
from src.gemmini.gemmini_config import *
from src.gemmini import gemmini_utils as gu
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

#
# Versions instrumented for RTL injection with the HDFIT tool
#
#CONFIG_KEY = "OSDIM4_HDFIT"
#CONFIG_KEY = "OSDIM8_HDFIT"
#CONFIG_KEY = "OSDIM16_HDFIT"
#CONFIG_KEY = "OSDIM32_HDFIT"
#CONFIG_KEY = "OSDIM64_HDFIT"

#
# WS configs
#
#CONFIG_KEY = "WSDIM4"
#CONFIG_KEY = "WSDIM8"
#CONFIG_KEY = "WSDIM64"

#CONFIG_KEY = "TVU_SA"
#CONFIG_KEY = "TestConfig"


#CONFIG_KEY = sys.argv[1]
#print("CONFIG_KEY = ", CONFIG_KEY)
#
# Loads the Gemmini module - the ahead-of-time extension to interface with the verilated Gemmini module (this lib is designed in /rtl/lib/Gemmini)
#

defs.ENABLE_GL_FAULT_MODEL = False
#defs.ENABLE_GL_FAULT_MODEL = True

gemmini = ext.load_extension(CONFIG_KEY) # if CONFIG_KEY is a direct config name
gemmini.init()
gemmini.print_info()

DIM = conf.DIM
INPUT_TYPE  = conf.GEMM_INPUT_DTYPE
OUTPUT_TYPE = conf.GEMM_OUTPUT_DTYPE
MIN_INT, MAX_INT = -128, 127

# HDFIT assignment IDs for GL fault injection
MAC_UNIT_HDFIT_FIRST_TARGET_ID = 2     # First one is assign _0056_[16] =( io_in_a[8] & io_in_b[8]) ^ ((fiEnable && (2 == GlobalFiNumber)) ? GlobalFiSignal[0] : {1{1'b0}});
#MAC_UNIT_HDFIT_LAST_TARGET_ID = 3419  # 32/32-bit inputs/outputs (unsigned mac): Last one is assign _0184_[0] =( _0182_[0]) ^ ((fiEnable && (3419 == GlobalFiNumber)) ? GlobalFiSignal[0] : {1{1'b0}});
#MAC_UNIT_HDFIT_LAST_TARGET_ID  = 591  # 8/32-bit inputs/outputs (unsigned mac) 
MAC_UNIT_HDFIT_LAST_TARGET_ID  = 824   # 8/32-bit inputs/outputs (signed mac)


if CONFIG_KEY in ["WSDIM4", "WSDIM8", "WSDIM64"]: # as in WS we preload the B tensor, we must assure the stream size is same size of the SA
    STREAM_SIZE = conf.CONFIG_PARAMS[CONFIG_KEY]["dim"]
else:
    #STREAM_SIZE = 8
    # the shape of the left->right input stream must be DIM x STREAM_SIZE
    # the shape of the top->bottom input stream must be STREAM_SIZE x DIM
    # in all cases, we must assure STREAM_SIZE >= DIM
    STREAM_SIZE = conf.CONFIG_PARAMS[CONFIG_KEY]["dim"] + 4 


#unittest.skip
class TesterGemmini(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TesterGemmini, self).__init__(*args, **kwargs)

        assert (STREAM_SIZE >= conf.DIM)

        self.A = torch.zeros((DIM, STREAM_SIZE), dtype=INPUT_TYPE)

        self.B = torch.zeros((STREAM_SIZE, DIM), dtype=INPUT_TYPE)
        self.D = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)
        self.C = torch.zeros((DIM, DIM), dtype=OUTPUT_TYPE)

        self.A1 = torch.zeros((DIM, STREAM_SIZE), dtype=INPUT_TYPE)
        self.B1 = torch.zeros((STREAM_SIZE, DIM), dtype=INPUT_TYPE)

        self.A2 = torch.zeros((DIM, STREAM_SIZE), dtype=INPUT_TYPE)
        self.B2 = torch.zeros((STREAM_SIZE, DIM), dtype=INPUT_TYPE)

        self.A3 = torch.zeros((DIM, STREAM_SIZE), dtype=INPUT_TYPE)
        self.B3 = torch.zeros((STREAM_SIZE, DIM), dtype=INPUT_TYPE)

        self.A4 = torch.zeros((DIM, STREAM_SIZE), dtype=INPUT_TYPE)
        self.B4 = torch.zeros((STREAM_SIZE, DIM), dtype=INPUT_TYPE)

    def setUp(self):
        pass


    def tearDown(self):
        pass    

    #@unittest.skip
    def test_matmul_os(self):
        if conf.GEMM_MODE != conf.MODE_OS:
            return

        nFailed, nTrials = 0, 1000

        """
            Test the computation of C = A1*B1 + A2*B2 + A3*B3 + A4*B4 + D 
        """
        for i in range(nTrials):
            self.D.random_(MIN_INT, MAX_INT)
            self.A1.random_(MIN_INT, MAX_INT)
            self.A2.random_(MIN_INT, MAX_INT)
            self.A3.random_(MIN_INT, MAX_INT)
            self.A4.random_(MIN_INT, MAX_INT)
            self.B1.random_(MIN_INT, MAX_INT)
            self.B2.random_(MIN_INT, MAX_INT)
            self.B3.random_(MIN_INT, MAX_INT)
            self.B4.random_(MIN_INT, MAX_INT)

            # computes C_gold = A1*B1 + A2*B2 + A3*B3 + A4*B4 + D
            C_gold = self.D.clone()  # important. this must be cloned, otherwise changes in C_gold change D, which will change the preloaded matrix in Gemmini
            C_gold += torch.mm(self.A1, self.B1)
            C_gold += torch.mm(self.A2, self.B2)
            C_gold += torch.mm(self.A3, self.B3)
            C_gold += torch.mm(self.A4, self.B4)

            steps = gemmini.preload(self.D)
            steps = gemmini.stream(self.A1, self.B1)  # Note: the partial sums are kept stored in the PE accumulators 
            steps = gemmini.stream(self.A2, self.B2) 
            steps = gemmini.stream(self.A3, self.B3)
            steps = gemmini.stream(self.A4, self.B4) 
            steps = gemmini.flush_gemm(self.C, False)

            nFailed += not torch.equal(C_gold, self.C)

        fn = inspect.currentframe().f_code.co_name
        print(f"{fn} iterations: {nTrials} - failed: {nFailed}")
        self.assertTrue(nFailed == 0)


    #@unittest.skip
    def test_preload_then_flush(self):  # this is for OS only!
        if conf.GEMM_MODE != conf.MODE_OS:
            return

        nFailed, nTrials = 0, 1000
       
        for i in range(nTrials):
            self.D.random_(MIN_INT, MAX_INT)

            C_gold = self.D.clone()  
            C_gold_t = C_gold.clone().t()  

            steps = gemmini.preload(self.D)
            steps = gemmini.flush_gemm(self.C, False)

            nFailed += not torch.equal(C_gold, self.C)

            steps = gemmini.preload(self.D)
            steps = gemmini.flush_gemm(self.C, True) 
            nFailed += not torch.equal(C_gold_t, self.C)

        fn = inspect.currentframe().f_code.co_name
        print(f"{fn} iterations: {nTrials} - failed: {nFailed}")
        self.assertTrue(nFailed == 0)


    #@unittest.skip
    def test_matmul_ws(self):  
        if conf.GEMM_MODE != conf.MODE_WS:
            return

        """
            Test the computation of
            C1 = A1*B + D1
            C2 = A2*B + D2
            ...
            Cn = An*B * Dn
        """

        n = 10
        nFailed, nTrials = 0, 10

        for i in range(nTrials):
            self.B.random_(MIN_INT, MAX_INT)
            steps_pre = gemmini.preload(self.B)
             
            for j in range(n): # here we always reuse weights
                self.A.random_(MIN_INT, MAX_INT)
                self.D.random_(MIN_INT, MAX_INT)

                C_gold = torch.mm(self.A, self.B) + self.D
                steps_mm = gemmini.stream_bias(self.A, self.D, self.C)

                nFailed += not torch.equal(C_gold, self.C)

        fn = inspect.currentframe().f_code.co_name
        print(f"{fn} iterations: {nTrials} - failed: {nFailed}")
        self.assertTrue(nFailed == 0)


    #@unittest.skip
    def test_matmul_rtl(self):
        nFailed, nTrials = 0, 100

        for i in range(nTrials):
            self.A.random_(MIN_INT, MAX_INT)
            self.B.random_(MIN_INT, MAX_INT)
            self.D.random_(MIN_INT, MAX_INT)

            C_gold = torch.mm(self.A, self.B) + self.D

            gemmini.reset()  # Resets the accumulators to zero. For OS mode, this must be called if no preloads are performed before

            if conf.GEMM_MODE == conf.MODE_OS:
                gemmini.preload(self.D)
                steps = gemmini.stream(self.A, self.B)
                steps = gemmini.flush_gemm(self.C, False) 
            else: # MODE_WS
                steps_pre = gemmini.preload(self.B)
                #steps_mm  = gemmini.stream(self.A, self.C) # if D is zero
                steps_mm  = gemmini.stream_bias(self.A, self.D, self.C)

            nFailed += not torch.equal(C_gold, self.C)
            
        fn = inspect.currentframe().f_code.co_name
        print(f"{fn} iterations: {nTrials} - failed: {nFailed}")

        self.assertTrue(nFailed == 0)


    #@unittest.skip
    def test_fi_eval(self):
        #print("[Testing]:", inspect.currentframe().f_code.co_name)

        nFailed, nTrials = 0, 200
        fiSilent = True

        PE_IN_BITS, PE_OUT_BITS = 8, 32

        # targer signals evaluated. signal name: (signal id, signal bits)

        """
            Notes for faults in the IN_D input:
                No effects if faults are injected during streaming (only if injected during preloads).  
                The registers taking the in_d inputs are only vulnerable during preloads, whose value are then stored in c1. you can target the c1 register during stream
                WS: no effects, because the D matrix is fed to the in_b pins. the in_d pins were already used in preloads. injecting faults during stream will affect c1 (preloads stored in c2)
        """
        TARGET_SIGNALS = {
            # data signals  - PE inputs
            "IN_A":  (IN_A, PE_IN_BITS),  # input A signal id is 0, with PE_IN_BITS bits
            "IN_B":  (IN_B, PE_IN_BITS),  # input B signal id is 1, with PE_IN_BITS bits
            #"IN_D":  (IN_D, PE_OUT_BITS), # should not have any effects as faults are not injected during preloads
            
            # data signals - PE outputs
            #"OUT_A": (OUT_A, PE_IN_BITS),
            #"OUT_B": (OUT_B, PE_OUT_BITS),  # WS: this would be the partial sum flowing downstream
            #"OUT_C": (OUT_C, PE_OUT_BITS),  # WS: no effect. the outputs are streamed through out_b. OS: affects the accumulators

            # data signals - each PE has two registers to store: 1. accumulators in OS, or 2. weights in WS - in each case, only one reg. is actually used
            #"C1":   (C1, PE_OUT_BITS), # OS: faults (must use WILL_PE_INPUT_BE_ASSIGNED so the fault is not overwritten) WS: no faults (preloaded in C2)
            #"C2":   (C2, PE_OUT_BITS), # OS: no faults (preloaded in C1)  WS: yes faults

            # control signals
            #"SIG_PROPAG": (SIG_PROPAG, 1),
            #"SIG_VALID":  (SIG_VALID, 1),
        }

        faulty_outputs = 0

        for key, sig_id_bits_tuple in TARGET_SIGNALS.items():  
            target = sig_id_bits_tuple[0]
            bits = sig_id_bits_tuple[1]

            for i in range(nTrials):
                row = random.randint(0, DIM-1)
                col = random.randint(0, DIM-1)
                bit = random.randint(0, bits-1)
                ficycle = gu.get_pe_active_rand_cycle(row, col, DIM)

                cell = random.randint(
                    MAC_UNIT_HDFIT_FIRST_TARGET_ID, 
                    MAC_UNIT_HDFIT_LAST_TARGET_ID)

                #pol = random.randint(0, 1) # for permanent faults only 

                self.A.random_(1, MAX_INT)
                self.B.random_(1, MAX_INT)
                self.D.random_(1, MAX_INT)

                #self.A.random_(MIN_INT, MAX_INT)
                #self.B.random_(MIN_INT, MAX_INT)
                #self.D.random_(MIN_INT, MAX_INT)

                gemmini.clear_fault_list()
                
                gemmini.add_transient_fault(
                    target, row, col, bit, ficycle, cell, fiSilent) 
                
                #gemmini.add_permanent_fault(
                #target, row, col, bit, pol, cell, False) # Important: for permanents, one must clear the fault list for the next tests...

                C_gold = torch.mm(self.A, self.B)  # + self.D
                
                if conf.GEMM_MODE == conf.MODE_OS:
                    #steps_pre = gemmini.preload(self.D)
                    steps_str = gemmini.stream(self.A, self.B)
                    steps_flu = gemmini.flush_gemm(self.C, False)
                else:
                    steps_pre = gemmini.preload(self.B)
                    steps_str  = gemmini.stream(self.A, self.C) # if D=0
                    #steps_str  = gemmini.stream_bias(self.A, self.D, self.C)
                gemmini.clear_fault_list()
                
                faulty_outputs += not torch.equal(C_gold, self.C)

        injections = len(TARGET_SIGNALS)*nTrials
        fn = inspect.currentframe().f_code.co_name

        print(f"{fn} Injections: {injections}\nOutput tensors with errors: {faulty_outputs}")


if __name__ == '__main__':
    unittest.main()
    gemmini.finish()
