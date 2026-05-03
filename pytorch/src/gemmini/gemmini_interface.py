import torch
import random
from typing import Optional

from src.gemmini import gemmini_config as conf
from src.gemmini import gemmini_utils as gu
from src.gemmini import gemmini_extension_definitions as ext
from src.flist import fault_list as flist
from src.utils import utils as u
from src import definitions as defs

"""
 A Gemmini wrapper to express basic Gemmini operations (preload, stream inputs and flushing outputs)    
 This interface binds PyTorch to the C++ Gemmini interface. 
 This is done as a C++ extension defined in gemmini_interface.cpp
 The C++ extension is accessed through the 'device' object.
"""

# prints the type and position of each fault, in the device level, in the cycle the fault is injected
#FI_SILENT = False
FI_SILENT = True

class Gemmini:
    def __init__(self, config_key):
        self.device = self.load_module(config_key)

        #resetself.device.print_info()

        # a tensor to store the resulting matmul tile
        self.C_tile = torch.zeros((conf.DIM, conf.DIM), dtype=conf.GEMM_OUTPUT_DTYPE)

        print(self)


    # initializes, Gemmini, the RTL DUT, and MxM solver in the 'gemmini' C++ extension
    def load_module(self, config_key):
        print()
        print(f"{u.Co['fg'][65]}[Loading Gemmini extension]:{u.R}")

        #
        # Loads the Gemmini module - the ahead-of-time extension to interface with the verilated Gemmini module (this lib is designed in /rtl/lib/Gemmini)
        # The Gemmini conifg must be set in conf.CONFIG_KEY
        #
        device = ext.load_extension(config_key)
       
        # Initializes Gemmini, the RTL DUT, and MxM solver in the 'gemmini' C++ extension
        if not device.init():
            raise RuntimeError("[FAILED]: igemm.load_module()")

        return device


    # deletes the gemmini module and all associated objects in the C++ side
    def finish(self):
        self.device.finish()


    # resets the Gemmini mesh signals
    def reset(self):
        return self.device.reset()


    def __str__(self):
        mode = "OS" if conf.GEMM_MODE == conf.MODE_OS else "WS"
        return f"- Config: {self.device.get_config_name()}\n- Mode: {mode}\n- conf.DIM: {conf.DIM}"


    #Preloads a tile tensor to the SA. If OS, preloads to the PE accumulators. In WS the tile is preloaded to the weight registers
    def preload(self, tile):
        return self.device.preload(tile)


    # updates the fault list in Gemmini with the positions previously (globaly) stored in "fault"
    def update_fault_list(self, fault: Optional[flist.GemminiPos]):
        self.device.clear_fault_list()
        
        # the iteraction in which the PE is injected
        ficycle = gu.pe_first_cycle(fault.pe_row, fault.pe_col) + 1
        #ficycle = gu.pe_last_cycle(fault.pe_row, fault.pe_col, conf.DIM)
        #ficycle = gu.get_pe_active_rand_cycle(fault.pe_row, fault.pe_col, conf.DIM)

        self.device.add_transient_fault(
            fault.target, 
            fault.pe_row, 
            fault.pe_col, 
            fault.bit,
            ficycle,
            fault.cell, 
            FI_SILENT
        )
        #gemmini_device.add_permanent_fault(fault.target, fault.pe_row, fault.pe_col, fault.bit, 0, FI_SILENT)


    # adds a random fault generated at runtime
    def update_fault_list_random(self):
        self.device.clear_fault_list()
        new_fault = gu.gen_random_fault()
        ficycle = gu.pe_first_cycle(fault.pe_row, fault.pe_col)
        
        self.device.add_transient_fault(
            new_fault.target, 
            new_fault.pe_row, 
            new_fault.pe_col, 
            new_fault.bit,
            ficycle,
            fault.cell, 
            FI_SILENT
        )


# Gemmini OS mode
class GemminiOS(Gemmini):
    def __init__(self, config_key):
        super().__init__(config_key=config_key)


    # flushes the PE accumulators
    def flush_gemm(self, out_tile, transpose_output=False):
        return self.device.flush_gemm(out_tile, transpose_output)


    # performs matrix multiplication of the two tensors A[conf.DIM][conf.DIM] and B[conf.DIM][conf.DIM] with the Gemmini systolic array. The outputs are kept in the PE accumulators
    def stream(self, A_tile, B_tile):        
        return self.device.stream(A_tile, B_tile)


    # performs matrix multiplication of the two tensors A[conf.DIM][conf.DIM] and B[conf.DIM][conf.DIM] with the Gemmini systolic array, flushes the PEs and returns the results to a tensor
    def matmul(self, A_tile, B_tile, trans_out=False):
        _ = self.stream(A_tile, B_tile)
        _ = self.flush_gemm(self.C_tile, trans_out)
        return self.C_tile


    # performs tiled matmul for tensors of generic sizes. in practice, this is not used for fault injections 
    def tiled_matmul(self, A, B, D=None, stream_size=conf.DIM):
        assert(len(A.shape) == 2)
        assert(len(B.shape) == 2)
        assert(A.shape[1] == B.shape[0])

        I, K = A.shape
        K, J = B.shape

        # Pad A and B so that all tiles are conf.DIM x conf.DIM
        I_pad = (conf.DIM - I % conf.DIM) % conf.DIM
        #K_pad = (conf.DIM - K % conf.DIM) % conf.DIM
        K_pad = (stream_size - K % stream_size) % stream_size
        J_pad = (conf.DIM - J % conf.DIM) % conf.DIM

        A_p = torch.nn.functional.pad(A, (0, K_pad, 0, I_pad))  # (left, right, top, bottom)
        B_p = torch.nn.functional.pad(B, (0, J_pad, 0, K_pad))
        #D_p = torch.nn.functional.pad(D, (0, I_pad, 0, J_pad)) if D != None else None

        if D is not None:
            assert(len(D.shape) == 2)
            assert(D.shape == (A.shape[0], B.shape[1]))
            D_p = torch.nn.functional.pad(D, (0, J_pad, 0, I_pad))
        else:
            D_p = None

        I_full, K_full = A_p.shape
        _, J_full = B_p.shape

        # Output matrix (will be cropped later)
        C_p = torch.zeros((I_full, J_full), dtype=A.dtype, device=A.device)

        cycles_compute, cycles_flush = 0, 0

        # Iterate over conf.DIM×conf.DIM tiles
        for i in range(0, I_full, conf.DIM):
            for j in range(0, J_full, conf.DIM):
                if D != None:
                    D_tile = D_p[i:i + conf.DIM, j:j+conf.DIM]
                    steps = self.preload(D_tile.contiguous())

                for k in range(0, K_full, stream_size):  # flows every  A col/B row. reuses the SA output in every iteration
                    A_tile = A_p[i:i+conf.DIM, k:k+stream_size]
                    B_tile = B_p[k:k+stream_size, j:j+conf.DIM]
                    cycles_compute += self.stream(A_tile.contiguous(), B_tile.contiguous())
                
                cycles_flush += self.flush_gemm(self.C_tile, False) # we don't need to flush_gemm every k iteration due to OS output reuse

                C_p[i:i + conf.DIM, j:j+conf.DIM] = self.C_tile

        # Remove padding to return the true result shape
        C = C_p[:I, :J]

        return C


# Gemmini WS mode
class GemminiWS(Gemmini):
    def __init__(self, config_key):
        super().__init__(config_key=config_key)


    # performs matrix multiplication of the two tensors A[conf.DIM][conf.DIM] and B[conf.DIM][conf.DIM] with the Gemmini systolic array. The outputs are kept in the PE accumulators
    def stream(self, A_tile, C_tile):        
        return self.device.stream(A_tile, C_tile)


    # performs matrix multiplication of the two tensors A[conf.DIM][conf.DIM] and B[conf.DIM][conf.DIM] with the Gemmini systolic array. The outputs are kept in the PE accumulators
    def stream_bias(self, A_tile, D_tile, C_tile):        
        return self.device.stream_bias(A_tile, D_tile, C_tile)


    # WS matmul. A[conf.DIM][conf.DIM] and B[conf.DIM][conf.DIM]. this requires preloading the B matrix first. 
    def matmul(self, A_tile, B_tile, D_tile=None):
        # WS only passes the A and D (if not None). B should be preloaded first. C_tile is passed here just to get the outputs directly from the PE partial sum (last row) - there's no flush_gemm for WS mode
        preload_cycles = self.preload(B_tile)

        if D_tile == None:
            compute_cycles = self.stream(A_tile, self.C_tile)
        else:
            compute_cycles = self.stream_bias(A_tile, D_tile, self.C_tile)

        return self.C_tile


    # performs tiled matmul for tensors of generic sizes. in practice, this is not used for fault injections
    def tiled_matmul(self, A, B, D=None): # TODO: WS only supports streaming sizes the same size of the SA
        assert(len(A.shape) == 2)
        assert(len(B.shape) == 2)
        assert(A.shape[1] == B.shape[0])

        if D != None:
            assert(len(D.shape) == 2)
            assert(D.shape == (A.shape[0], B.shape[1]))

        I, K = A.shape
        _, J = B.shape

        I_pad = (conf.DIM - I % conf.DIM) % conf.DIM
        K_pad = (conf.DIM - K % conf.DIM) % conf.DIM
        J_pad = (conf.DIM - J % conf.DIM) % conf.DIM

        A_p = torch.nn.functional.pad(A, (0, K_pad, 0, I_pad))
        B_p = torch.nn.functional.pad(B, (0, J_pad, 0, K_pad))
        D_p = torch.nn.functional.pad(D, (0, J_pad, 0, I_pad)) if D != None else None

        I_full, K_full = A_p.shape
        _, J_full = B_p.shape

        C_p = torch.zeros((I_full, J_full), device=A.device, dtype=A.dtype)

        D_tile = None # default

        for k in range(0, K_full, conf.DIM):
            for j in range(0, J_full, conf.DIM):
                # 1) Preload weights ONCE (stationary)
                B_tile = B_p[k:k+conf.DIM, j:j+conf.DIM].contiguous() # B_tile(k,j)

                self.preload(B_tile)

                for i in range(0, I_full, conf.DIM):  # loops through all A tile rows
                    A_tile = A_p[i:i+conf.DIM, k:k+conf.DIM].contiguous() # A_tile (i,k)  -> so we have A_tile (i,k) * B_tile(k,j)
                
                    if k == 0 and D_p != None:  # D_tile should be added only once per output tile   C_tile(i,j) = SUM_k{A_tile(i,k)*B_tile(k,j)} + D_tile(i,j)
                        D_tile = D_p[i:i+conf.DIM, j:j+conf.DIM].contiguous() # we dont have to do this for every k (indexes i and j)
                        r = self.stream_bias(A_tile, D_tile, self.C_tile) # stream this tile only in the first iteration
                    else:
                        r = self.stream(A_tile, self.C_tile)

                    C_p[i:i+conf.DIM, j:j+conf.DIM] += self.C_tile

        return C_p[:I, :J]



#
# gemmini_device: the (global) main (and only) Gemmini handler. Access to Gemmini can only be done through this interface
#
gemmini_device: Gemmini=None
