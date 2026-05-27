#
# Gemmini *host* performance tests
#
import sys
import os
import torch

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.gemmini.gemmini_config as conf
import src.gemmini.gemmini_interface as gemmini_interface

MIN_INT, MAX_INT = -128, 127
INPUT_TYPE  = conf.GEMM_INPUT_DTYPE
OUTPUT_TYPE = conf.GEMM_OUTPUT_DTYPE


def run_tiled_matmul(gemmini):
    w_r, w_c = 64, 147
    i_r, i_c = w_c, 12544
    b_r, b_c = w_r, i_c

    stream_size = conf.DIM + 4

    W = torch.randint(MIN_INT, MAX_INT, torch.Size([w_r, w_c]), dtype=torch.int)
    I = torch.randint(MIN_INT, MAX_INT, torch.Size([i_r, i_c]), dtype=torch.int)
    bias = torch.randint(MIN_INT, MAX_INT, torch.Size([b_r, b_c]), dtype=torch.int)

    if conf.GEMM_MODE == conf.MODE_WS:
        Wt = W.t().contiguous() # very important: all tensors sent to Gemmini must be contiguous
        It = I.t().contiguous() # tranpose operations may result in non-contiguous tensors!!!
        biast =  bias.t().contiguous()

    # OS mode
    if conf.GEMM_MODE == conf.MODE_OS:
        O = gemmini.tiled_matmul(A=W, B=I, D=bias, stream_size=stream_size)  # computes W*I + bias (which is what we want)

    # WS mode
    else: # In WS, we have to make the *weight* stationary, so we must feed the weights to the B pins in Gemmini, so we compute as follows:
        O = gemmini.tiled_matmul(A=It, B=Wt, D=biast).t() # computes Ot = It*Wt + biast => (It*Wt + biast)t = (It*Wt)t + (biast)t = W*I + bias (which is what we want)

    O_ref = torch.mm(W, I)

    if bias != None: 
        O_ref += bias

    print(f"{CONFIG_KEY}: Tiled matmul results match: {torch.equal(O, O_ref)}")


def run_simple_matmul(gemmini):
    DIM = conf.CONFIG_PARAMS[CONFIG_KEY]["dim"]

    A = torch.randint(MIN_INT, MAX_INT, torch.Size([DIM, DIM]), dtype=torch.int)
    B = torch.randint(MIN_INT, MAX_INT, torch.Size([DIM, DIM]), dtype=torch.int)
    D = torch.randint(MIN_INT, MAX_INT, torch.Size([DIM, DIM]), dtype=torch.int)

    C_ref = torch.mm(A, B) + D
    C = gemmini.matmul(A, B, D)

    print(f"{CONFIG_KEY}: Simple matmul results match: {torch.equal(O, O_ref)}")


# sweeps through all configs keys
tested_configs = [
    "OSDIM4",
    "OSDIM8",
    "OSDIM16",
    "OSDIM32",
    "OSDIM64",
    "WSDIM4",
    "WSDIM8"
]

for CONFIG_KEY in tested_configs:
    print(f"Testing with config: {CONFIG_KEY}")
    
    mode = conf.CONFIG_PARAMS[CONFIG_KEY]["mode"]

    if mode == conf.MODE_OS:
        gemmini = gemmini_interface.GemminiOS(CONFIG_KEY)

    elif mode == conf.MODE_WS:
        gemmini = gemmini_interface.GemminiWS(CONFIG_KEY)
    
    else:
        raise("Invalid Gemmini mode")

    run_simple_matmul(gemmini)
    run_tiled_matmul(gemmini)

    gemmini.finish()
