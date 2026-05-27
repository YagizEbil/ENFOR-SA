import torch
import utils
import sys
import os
import numpy as np
from tqdm import tqdm

# set a dummy path, as this is not required for this experiment 
os.environ["PATH_IMAGENET"] = ""


import fault_list as fl 
import logger

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.gemmini.gemmini_extension_definitions as ext
import src.gemmini.gemmini_config as conf
import src.definitions as defs

from src.conv import tile_ops as tile_ops
from src.gemmini.gemmini_config import *
from src.gemmini import gemmini_utils as gu

#experiment = "experiment-sc26-v1"
experiment = "debug"

conv_out_samples_path = f"tests/exp_conv2d_tpu/logs/{experiment}/conv_out_samples"
trace_file_path = f"tests/exp_conv2d_tpu/logs/{experiment}/trace.tsv"

os.makedirs(conv_out_samples_path, exist_ok=True)

input_path  = "tests/exp_conv2d_tpu/experiments/input_simple_conv_2d_1_1024_1024_1_40_40_1_1.pt"
golden_path = "tests/exp_conv2d_tpu/experiments/golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy"
fault_list = "./tests/exp_conv2d_tpu/experiments/WSDIM64_sc26.csv"


# the gemmini config, resembling the TPU
CONFIG_KEY = "WSDIM64"

defs.ENABLE_GL_FAULT_MODEL = False
#defs.ENABLE_GL_FAULT_MODEL = True

# saves the conv outputs after each injection
SAVE_CONV_OUTPUTS=True

# loads the gemmini extension
gemmini = ext.load_extension(CONFIG_KEY) # if CONFIG_KEY is a direct config name

# the SA has shape DIMxDIM
DIM = conf.DIM


# loads a reference np tensor
def load_np_tensor_as_pt(tensor_path):
    data = np.load(tensor_path)
    data = np.squeeze(data, axis=1)
    return torch.from_numpy(data)


# runs the injection loop approach for the Standard Conv2d case. Faults are injected in the 64x64 WS array
def run_injection_loop():
    trace_logger = logger.Logger(
        output_fn=trace_file_path, 
        log_type=logger.TYPE_STATS_PER_FAULT_GEMMINI, 
        skip_log=False
        )

    injections = 17000 # this suffices for both RTL and GL cases
    fi_silent = True

    # loads the fault list
    base_fault_list = fl.load_fault_list(
        file_path=fault_list, 
        itv_tuple=(0, injections-1), 
        shuffle_list=False
        ) 

    ninjections = len(base_fault_list)

    input_pt = torch.load(input_path) #.contiguous()

    # the setup used in the radiation experiments
    w_shape = (1, 1, 40, 40)
    w_value = 127
    scale= 4.999999873689376e-06

    weights = torch.full(w_shape, w_value, dtype=torch.int)

    input_im2col = utils.im2col_tf_same_t(input_pt, weights.shape).to(torch.int)  # torch.Size([1048576, 1600])  -> Important: input_im2col is NOT contiguous
    weights_flat = utils.flatten_weight_t(weights).to(torch.int)  # torch.Size([1600, 1])

    # golden conv out reference
    full_conv_out_gold = load_np_tensor_as_pt(golden_path).to(torch.uint8)
    full_conv_out_gold_original_shape = full_conv_out_gold.shape

    full_conv_out_gold = full_conv_out_gold.reshape(
        input_im2col.shape[0], 
        weights_flat.shape[1]
        )

    # loops over each fault
    for i in tqdm(range(0, ninjections), desc="Processing fault list", unit="iter", ncols=0):
    #for i in range(ninjections):

        fault = base_fault_list[i]

        # extracts a an input tile (DIM x DIM)
        i_tile = tile_ops.extract_tile(
            input_im2col, 
            fault.tile.a_row, 
            fault.tile.a_col, 
            DIM
            )

        # extracts a kernel tile (DIM x DIM)
        w_tile = tile_ops.extract_tile(
            weights_flat, 
            fault.tile.b_row,  # Note: fault.tile.b_row == fault.tile.a_col always
            fault.tile.b_col, 
            DIM
            ) 

        # faults will be injected in random active cycle in each target PE
        ficycle = gu.get_pe_active_rand_cycle(
            fault.gemm.pe_row, 
            fault.gemm.pe_col, 
            DIM)

        gemmini.clear_fault_list()

        gemmini.add_transient_fault(
            fault.gemm.target, 
            fault.gemm.pe_row, 
            fault.gemm.pe_col, 
            fault.gemm.bit,
            ficycle,
            fault.gemm.cell, 
            fi_silent
            ) 

        # the Gemmini tile output (DIM x DIM)
        tile_mm_out_gemm = torch.empty((DIM, DIM), dtype=torch.int)
        tile_mm_out_gemm.zero_()

        # preloads the kernel tile
        gemmini.preload(w_tile)

        # streams the input. the output tile from Gemmini is stored in tile_mm_out_gemm
        gemmini.stream(i_tile, tile_mm_out_gemm)

        # computes the golden tile
        tile_mm_out_gold = torch.mm(i_tile, w_tile)

        valid_rows = min(DIM, full_conv_out_gold.shape[0] - fault.tile.a_row * DIM)
        valid_cols = min(DIM, full_conv_out_gold.shape[1] - fault.tile.b_col * DIM)

        # extracts the valid region, from the tile output, that can affect the conv output
        tile_mm_out_gemm = tile_mm_out_gemm[:valid_rows, :valid_cols]
        tile_mm_out_gold = tile_mm_out_gold[:valid_rows, :valid_cols]

        fault.status.msk_gemm = torch.equal(
            tile_mm_out_gemm, 
            tile_mm_out_gold)

        # scales, rounds and clamps the Gemmini output
        tile_mm_out_gemm = tile_mm_out_gemm*scale
        tile_mm_out_gold = tile_mm_out_gold*scale

        fault.status.msk_scale = not fault.status.msk_gemm and \
            torch.equal(
                tile_mm_out_gemm, 
                tile_mm_out_gold)

        # applies rounding
        tile_mm_out_gemm = tile_mm_out_gemm.round()
        tile_mm_out_gold = tile_mm_out_gold.round()

        fault.status.msk_round = not fault.status.msk_gemm and \
            not fault.status.msk_scale and \
            torch.equal(
                tile_mm_out_gemm, 
                tile_mm_out_gold)

        # clamps to the proper interval
        tile_mm_out_gemm = tile_mm_out_gemm.clamp(0, 255).to(torch.uint8)
        tile_mm_out_gold = tile_mm_out_gold.clamp(0, 255).to(torch.uint8)

        fault.status.msk_clamp = not fault.status.msk_gemm and \
            not fault.status.msk_scale and \
            not fault.status.msk_round and \
            torch.equal(
                tile_mm_out_gemm, 
                tile_mm_out_gold)

        tiles_match = fault.status.msk_gemm or \
            fault.status.msk_scale or \
            fault.status.msk_round or \
            fault.status.msk_clamp 

        fault.status.corrupted_elements = 0
        fault.status.critical = False

        if not tiles_match:
            # subtracts the golden tile, in the fault position, from the golden output
            full_conv_out_work = tile_ops.sub_tile(  
                full_conv_out_gold,   
                tile_mm_out_gold,#_valid_region, 
                fault.tile.a_row, 
                fault.tile.b_col, 
                DIM
                )

            # sums the "faulty" gemmini tile, in the fault position, to the golden output
            full_conv_out_work = tile_ops.sum_tile(
                full_conv_out_work,  
                tile_mm_out_gemm,#_valid_region,
                fault.tile.a_row, 
                fault.tile.b_col, 
                DIM
                )

            fault.status.corrupted_elements = torch.sum(
                full_conv_out_gold != full_conv_out_work).item()
            
            fault.status.critical = fault.status.corrupted_elements != 0
           
            if SAVE_CONV_OUTPUTS:    
                np.save(
                    f"{conv_out_samples_path}/fault_{fault.tag}.npy", 
                    full_conv_out_work.reshape(full_conv_out_gold_original_shape).numpy())
                
                # save as compressed np tensors. compresses a lot, but reduces injection time by around 3x
                """
                np.savez_compressed(
                    f"radiate/trace/conv_out_samples/fault_{fault.tag}.npz", 
                    full_conv_out_work.reshape(full_conv_out_gold_original_shape).numpy())
                """
        trace_logger.try_dump_item(
            logger.StatsPerFaultGemmini(fault))

        del tile_mm_out_gemm
        del tile_mm_out_gold
        del i_tile
        del w_tile

    trace_logger.flush()


# tests if we can re-compute the exact same conv. but expressed as a matmul
def test_parameters_match_golden():
    input_pt = torch.load(input_path)  # NCHW

    # simple conv case
    w_shape = (1, 1, 40, 40) # PyTorch shape (out_channels, in_channels, kernel_height, kernel_width)
    
    w_value = 127

    weights = torch.full(w_shape, w_value, dtype=torch.int32)

    print(f"Input shape  {input_pt.shape}")
    print(f"Weight shape {weights.shape}")

    input_im2col = utils.im2col_tf_same_t(input_pt, weights.shape).to(torch.int) # torch.Size([1048576, 1600])
    weights_flat = utils.flatten_weight_t(weights) # torch.Size([1600, 1])

    print(f"input_im2col shape  {input_im2col.shape}")
    print(f"weights_flat shape {weights_flat.shape}")

    full_conv_out_work = torch.mm(input_im2col, weights_flat)
    full_conv_out_gold = load_np_tensor_as_pt(golden_path)

    print(f"matmul out shape: ", full_conv_out_work.shape)
    print(f"full_conv_out_gold shape: ", full_conv_out_gold.shape)

    # simple conv
    scale1= 4.999999873689376e-06
    scale2 = 1.0
    scale3 = 1.0

    full_conv_out_work = full_conv_out_work*scale1*scale2/scale3
    full_conv_out_work = full_conv_out_work.round()
    full_conv_out_work = full_conv_out_work.clamp(0, 255).to(torch.uint8)

    print("full_conv_out_work.shape: ", full_conv_out_work.shape)
    print("full_conv_out_gold.shape: ", full_conv_out_gold.shape)

    # 1024*1024 = 1048576
    full_conv_out_work = full_conv_out_work.reshape(full_conv_out_gold.shape)

    total_diff = torch.sum(full_conv_out_gold != full_conv_out_work).item()

    print("Total diverging items: ", total_diff)
    print("Results match: ", total_diff == 0)


if __name__ == '__main__':
    
    #test_parameters_match_golden()
    #exit(0)

    max_inter_threads, max_intra_threads = 2, 2
    torch.set_num_interop_threads(max_inter_threads)
    torch.set_num_threads(max_intra_threads)

    gemmini.init()
    gemmini.print_info()

    run_injection_loop()
    
    gemmini.finish()


