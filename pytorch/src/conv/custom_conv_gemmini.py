import torch
import random 
import numpy as np

from collections import defaultdict
from typing import Optional
from torch._ops import ops

from src.gemmini import gemmini_interface as igemm
from src.gemmini import gemmini_config as conf
from src.utils import utils as u
from src.conv import tile_ops as tile_ops
from src.conv import flattener as flattener
from src.conv import cached_tensors as tcache
from src import definitions as defs
from src.flist.fl import fl

#
# the type of convolution the custom_conv object represents (also needed in instrumented_model for the layer replacement)
#
CONV, CONV_RELU = 0, 1

#
# faults can be masked in four different levels during the convolution (gemm_msk, scale_msk, round_msk, clamp_msk)
#
MASK_LEVELS = 4

#
# this class models the convolution as a matmul
# we extract a tile from the input and kernel. this single tile is offloaded to gemmini to simulate SEUs. 
# the output tile is then re-scaled ("re-quantized"). the golden tile in the full output conv tensors is then replaced by the tile computed by gemmini
#
class CustomConv:
    def __init__(self):
        self.input_ids = np.arange(0, defs.BATCH_SIZE+1) # for now we set this externally
        self.batch_id = 0
        self.conv_type = -1

        # list of faults with collisions for the parallel approach
        # if two faults hit the same tile, we avoid one of them, and add it to this list
        # the 'collided' faults are then added back to the fault list for further trials until the list is empty
        self.fault_list_collision: defaultdict[int, fl.Fault] = defaultdict(fl.Fault)
        self.prepared = False #  this must be set to False for every new target conv layer. See in InstrumentedModel how this used...


    # extracts and prepares the weight, biases and quantization paramters for this layer. repeat this for every new layer
    # this must be called only once for every replaced layer (for fault injection) - this does not change for different inputs, so we save a lot of performance doing it once...
    def load_params(self, conv_model, layer_id=0):
        self.layer_id = layer_id

        ### Extract some parameters needed for the manual convolution ###
        self.weights_flat = flattener.flatten_weight(conv_model)
        self.weights = conv_model.weight()
        self.bias = conv_model.bias()
        self.weight_scale, _ = flattener.get_scales_and_zero_points(self.weights)
        self.output_scale = conv_model.scale
        self.z_zero_point = conv_model.zero_point
       
        self.is_per_tensor_qtz = self.weights.qscheme() == torch.per_tensor_affine

        # extract only the portion of the weight scale that matters - i'm computing a single matmul tile
        # this way i'm expanding the weight scale to the same shape of the weight tensor, so that i can extract the proper weight scale tile
        if self.is_per_tensor_qtz:  # MobileNet_V2 and MobileNet_V3_Large use per-tensor quantization, so the scale parameter is a single float value
            self.weight_scale_exp = self.weight_scale
        else: # e.g., ResNet18 and most other models use per-channel quantization, so weight_scale is a tensors of scale factors (e.g, shape([64])) 
            self.weight_scale_exp = self.weight_scale.unsqueeze(1).expand(-1, self.weights_flat.shape[1])

    """
    gemmini_os_compute_tile():  models a single tile multiplication on Gemmini (a golden matmul is also computed for reference
        - OS version. preloads the bias matrix, and flows inputs and weights through Gemmini
         (this was the version used in the DAC/DATE/VTS results)

        - Receives the input (im2col format) and the flattened weight
        - Extracts a single random tile from the weight (i, k) and input (k, j)
        - Offloads a single random tile for matmul in Gemmini
        - Fault injection is performed in Gemmini in the position specified in gemmini_fault
        - The bias matrix may or not be preloaded to Gemmini (preload_bias_hw). This is optional
        - The Gemmini output is re-scaled back according to the procedure shown below
        - The function keeps track where the fault may or may be not masked (e.g., gemm_msk, scale_msk, round_msk, clamp_msk )
        - The function returns the tile result computed by Gemmini, the golden tile (computed in PyTorch SW), and a list of masking levels

    The quantization approach to scale the conv outputs are taken from:
        https://discuss.pytorch.org/t/how-does-quantized-conv2d-handle-scale-and-zero-point/129437

    The basic procedure is:
        z = qconv(wq, xq)  -> this is the matmul done in integer
        # z is at scale (weight_scale*input_scale) and at int32
        # Convert to int32 and perform 32 bit add
        bias_q = round(bias/(input_scale*weight_scale))
        z_int = z + bias_q
        # rounding to 8 bits
        z_out  = round[(z_int)*(input_scale*weight_scale)/output_scale) - z_zero_point] 
        z_out = saturate(z_out)

        (I think it is actually ** + z_zero_point ** in the forumula above
         This can be confirmed in the function debug_conv() of ./toys/gemmini_toy_conv_model.py
         (https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf))
    """

    def gemmini_os_compute_tile(
        self, 
        input_im2col,  # The input in the im2col format
        tile_w_row,    # The weight tile row positon
        tile_w_col,    # The weight tile col positon
        tile_i_row,    # The input tile row positon
        tile_i_col,    # The input tile col positon
        preload_bias_hw=True, # preloads the bias matrix in the SA (for FI, this is not actually mandatory)
        gemmini_fault=None    # The Gemmini fault position of type fl.GemminiPos
        ): #: Optional[fl.GemminiPos] = None):

        # extracts the weight and input tiles
        # note that the zero points are already subtracted from self.weights_flat and input_im2col
        w_tile = tile_ops.extract_tile(self.weights_flat, tile_w_row, tile_w_col, conf.DIM)
        i_tile = tile_ops.extract_tile(input_im2col,      tile_i_row, tile_i_col, conf.DIM)

        # keeps track of the different levels in which fault masking can occur during convolution
        gemm_msk = scale_msk = round_msk = clamp_msk = False

        if self.is_per_tensor_qtz:
            weight_scale_tile = self.weight_scale_exp   
        else:
            weight_scale_tile = tile_ops.extract_fp_tile(self.weight_scale_exp, tile_w_row, tile_w_col, conf.DIM)

        # expands the bias so that i can extract the tile that matters for this position
        # TODO: this could be done offline once per layer in load_params(), but i need the input shape (solve this somehow) 
        if preload_bias_hw:
            bias_exp = self.bias.unsqueeze(1).expand(-1, input_im2col.shape[1]) # expands the bias to the shape [A_rows, B_cols]

            # extracts only the bias portion that matters - as i'm computing a single matmul tile
            bias_tile = tile_ops.extract_fp_tile(bias_exp, tile_w_row, tile_i_col, conf.DIM)

            # Convert to int32 and perform 32 bit add
            bias_tile_q = bias_tile/(self.input_scale*weight_scale_tile)
            bias_tile_q = bias_tile_q.round().to(torch.int32)

            # Preloads the tile to the PE accumulators (maybe use preload_bias_hw=False for large arrays to improve performance)
            igemm.gemmini_device.preload(bias_tile_q) 
        else:
            igemm.gemmini_device.reset()
            
        # updates the Gemmini fault list
        if not defs.RUN_GOLDEN_MODE:
            igemm.gemmini_device.update_fault_list(gemmini_fault)

        # performs the tile matmul in Gemmini
        z_gemm = igemm.gemmini_device.matmul(w_tile, i_tile)
        #print(f"{igemm.gemmini_device.cycles_preload}\t{igemm.gemmini_device.cycles_compute}\t{igemm.gemmini_device.cycles_flush}")

        # computes the golden tile result
        z_gold = torch.mm(w_tile, i_tile) 

        del w_tile
        del i_tile

        if preload_bias_hw:
            z_int_gold = z_gold + bias_tile_q
            z_int_gemm = z_gemm # + bias_tile_q  # Important: iff the bias is not preloaded in Gemmini first, we should add it to the output here, in sw
        else:
            z_int_gold = z_gold 
            z_int_gemm = z_gemm

        gemm_msk = torch.equal(z_int_gemm, z_int_gold) # was the fault masked in the matmul?
        
        if gemm_msk:
            return None, None, [gemm_msk, scale_msk, round_msk, clamp_msk]

        # TODO: check if this should be compiled ahead of time
        z_out_gold = z_int_gold*self.input_scale*weight_scale_tile/self.output_scale + self.z_zero_point
        z_out_gemm = z_int_gemm*self.input_scale*weight_scale_tile/self.output_scale + self.z_zero_point

        #if (z_out_gold > 255).sum() > 0: print("[Debug]: case 5") # never happens. z_out_gold is alwas < 255 due to proper scaling and zero_point
        # QuantizedConv2d:     z_out_gold is always positive because the + self.z_zero_point
        # QuantizedConvReLU2d: z_out_gold can have negative values - the + self.z_zero_point does not shift them to positive -> these will be clamped by the activation

        scale_msk = not gemm_msk and torch.equal(z_out_gemm, z_out_gold)  # was the fault masked in the scaling?
        
        if scale_msk: 
            return None, None, [gemm_msk, scale_msk, round_msk, clamp_msk]

        z_out_gold = z_out_gold.round()
        z_out_gemm = z_out_gemm.round()

        round_msk = not (gemm_msk or scale_msk) and torch.equal(z_out_gemm, z_out_gold) # was the fault masked in the rounding?

        if round_msk:
            return None, None, [gemm_msk, scale_msk, round_msk, clamp_msk]

        if self.conv_type == CONV_RELU: # QuantizedConvReLU2d
            # here, z_out_gold can have negative values - the + self.z_zero_point does not shift them to positive -> these will be clamped by the activation
           
            #if (z_out_gold > 255).sum() > 0: print("[Debug]: case 1") # z_out_gold never overflows - due to proper scaling and zp
            #if (z_out_gold < 0).sum() > 0: print("[Debug]: case 2")  # can happen

            # saturate. this is, in effect, the activation function - here, faults in negative outputs that do not change their signal will always be masked
            z_out_gold = z_out_gold.clamp(0, 255)
            z_out_gemm = z_out_gemm.clamp(0, 255)

            clamp_msk = not (gemm_msk or scale_msk or round_msk) and torch.equal(z_out_gemm, z_out_gold) # was the fault masked in the clamping (a.k.a. the activation function)
            
            if clamp_msk:
                return None, None, [gemm_msk, scale_msk, round_msk, clamp_msk]
       
        #else: # QuantizedConv2d. Note: QuantizedConv2d does not have a clamp, therefore it can propagate more faults
            #here, z_out_gold is always positive because the + self.z_zero_point. 
            #if (z_out_gold > 255).sum() > 0: print("[Debug]: case 3") # z_out_gold never overflows - due to proper scaling and zp
            #if (z_out_gold < 0).sum() > 0: print("[Debug]: case 4")   # never happens

        # rounding to 8 bits
        z_out_gold = z_out_gold.to(torch.uint8)
        z_out_gemm = z_out_gemm.to(torch.uint8)

        # here i check again if rounding to 8 bits masked the fault. if this is CONV_RELU, this will not be checked if it was masked during clamping (msk_levels[3])            
        round_msk = not (gemm_msk or scale_msk or clamp_msk) and torch.equal(z_out_gemm, z_out_gold) # was the fault masked in the rounding?
        
        if round_msk:
            return None, None, [gemm_msk, scale_msk, round_msk, clamp_msk]
    
        return z_out_gemm, z_out_gold, [gemm_msk, scale_msk, round_msk, clamp_msk]


    def matmul_random_tile(self, input_im2col):
        # alias
        fault = fl.next_fault

        # takes a random fault position at runtime instead of reading it from the (csv file) fault list
        #fault = fl.gen_random_fault(A.shape, B.shape, layer_id=2, possible_targets=[defs.SIGNAL['IN_A'], defs.SIGNAL['IN_B']])
        #print(f"Next fault: {fault}")
        #fault.tile.a_row, fault.tile.a_col, fault.tile.b_row, fault.tile.b_col = 0, 0, 0, 0
        
        C_tile_gemm, C_tile_gold, msk_levels = self.gemmini_os_compute_tile(
            input_im2col, 
            fault.tile.a_row, 
            fault.tile.a_col, 
            fault.tile.b_row, 
            fault.tile.b_col,
            preload_bias_hw=True,
            gemmini_fault=fault.gemm
            )

        return C_tile_gemm, C_tile_gold, msk_levels, fault.tile.a_row, fault.tile.b_col


    # computes multiple tile convolutions with Gemmini, sweeping the fault list and applying one fault per matmul
    def matmul_multiple_tiles(self, input_im2col, ret_tensor):
        injected_tiles = {}  # flags each affected tile

        ret_tensor = ret_tensor.reshape(self.weights_flat.shape[0], -1)

        self.fault_list_collision.clear()
   
        msk_levels = [True, True, True, True] # marks as True only if True for all faults in the loop below
   
        while fl.fault_list:
            # reads the next fault in the list
            fault = fl.fault_list.popleft()

            tile_key = (fault.tile.a_row, fault.tile.b_col)

            # prevents injecting in the same tile twice
            if tile_key not in injected_tiles:
                C_tile_gemm, C_tile_gold, msk_levels_fault = self.gemmini_os_compute_tile(
                    input_im2col, 
                    fault.tile.a_row, 
                    fault.tile.a_col, 
                    fault.tile.b_row, 
                    fault.tile.b_col,
                    preload_bias_hw=True,
                    gemmini_fault=fault.gemm
                    )
                
                injected_tiles[tile_key] = 1

                masked_at_any_level = any(msk_levels_fault)

                for i in range(MASK_LEVELS):
                    msk_levels[i] = msk_levels[i] and msk_levels_fault[i]

                if not masked_at_any_level:
                    ret_tensor = tile_ops.sub_tile(
                        ret_tensor, 
                        C_tile_gold, 
                        fault.tile.a_row, 
                        fault.tile.b_col, 
                        conf.DIM
                    )

                    ret_tensor = tile_ops.sum_tile(
                        ret_tensor, 
                        C_tile_gemm, 
                        fault.tile.a_row, 
                        fault.tile.b_col, 
                        conf.DIM
                    )

                del C_tile_gemm
                del C_tile_gold

            else: # if this tile was already injected in a previous iteraction, then this fault must be added back to the fault list for injection latter
                self.fault_list_collision[fault.tag] = fault

        return ret_tensor, msk_levels


    # the custom convolution operation
    def conv(self, conv_model, input_tensor, input_id, layer_id=0):
        # load the flattened input from the LUT. the input is flattened if it's not in the LUT
        input_im2col = tcache.load_cached_inputs(
            conv_model, 
            input_tensor, 
            input_id
        )

        # the inputs use per-tensor qtz. so the input is a single float value (has no shape)
        self.input_scale, _  = flattener.get_scales_and_zero_points(input_tensor)

        # computes the correct convolution
        #gold_tensor = ops.quantized.conv2d(input_tensor, conv_model._packed_params, conv_model.scale, conv_model.zero_point)
        gold_tensor = tcache.load_conv_ofm(input_id, conv_model, input_tensor)

        #[COLLECT LAYER SHAPES]
        #prints the shapes of the filter and input tensors for this conv layer. this is used to generate the fault list of each layer
        # remove this comment to collect the layer shapes (use REPL_ALL=True)
        #shape = (self.weights_flat.shape, input_im2col.shape); print(f"shape_layer[{layer_id}] = {shape}"); return gold_tensor, [False, False, False, False]
        
        # SW fault list
        #shape = (self.weights.shape, input_tensor.shape); print(f"shape_layer_sw[{layer_id}] = {shape}"); return gold_tensor, [False, False, False, False]

        shape_old = gold_tensor.shape

        if defs.TREE_FI_MODE or defs.RUN_PAR_FI:
            ret_tensor, msk_levels = self.matmul_multiple_tiles(input_im2col, gold_tensor)
            masked_at_any_level = any(msk_levels)

            if masked_at_any_level:  # not defs.RUN_GOLDEN_MODE can be set to False for debug/sanity purposes only
                return gold_tensor, msk_levels
        else:
            # Calculates the matmul (for both Gemmini and golden) of a single conv tile. the random tile positions are returned in repl_row, repl_col
            mmul_gemm, mmul_gold, msk_levels, repl_row, repl_col = self.matmul_random_tile(input_im2col)
            masked_at_any_level = any(msk_levels)

            if masked_at_any_level:
                del mmul_gold
                del mmul_gemm
                return gold_tensor, msk_levels

            gold_tensor = gold_tensor.reshape(self.weights_flat.shape[0], -1)

            # replaces the gold tile by the faulty one
            ret_tensor = tile_ops.sub_tile(
                gold_tensor, 
                mmul_gold, 
                repl_row, 
                repl_col, 
                conf.DIM
            )

            ret_tensor = tile_ops.sum_tile(
                ret_tensor,  
                mmul_gemm, 
                repl_row, 
                repl_col, 
                conf.DIM
            )

            del mmul_gold
            del mmul_gemm

        ret_tensor = ret_tensor.reshape(shape_old)
        return ret_tensor, msk_levels


    # for a batch of image inputs in the model's forward pass: conv_batch receives a batch of input tensors for the conv layer
    # this computes the convolution for each input tensor
    # returns the merged tensors as the conv. output (an output batch)
    # TODO: the custom convs are performed serially for each input in the batch - this becomes a bottleneck. All inputs can be computed in parallel
    def conv_batch(self, conv_model, batch_input_tensor, layer_id=0): 
        #return ops.quantized.conv2d(batch_input_tensor, conv_model._packed_params, conv_model.scale, conv_model.zero_point), msk_levels_inputs

        tensors = []
        msk_levels_inputs = [[False, False, False, False] for _ in range(batch_input_tensor.shape[0])]
        
        if defs.TREE_FI_MODE: 
            # in the tree approach we must keep track of fault colisions
            # initially we store the full fault list to be injected in base_fl
            # every call to self.conv removes faults from the fault list, 
            # so we need to load the fault list again for every new input (by reading from base_fl again)
            # after all inputs are processed, any faults that collided are kept in the self.fault_list_collision list
            # we must add these faults back to the list

            base_fl = fl.fault_list.copy()
        
            for i, input_tensor in enumerate(batch_input_tensor):  
                # sets the same fault list for the next batch 
                fl.fault_list = base_fl.copy()
                
                # runs the conv for this input batch. this will make the fault list empty
                ret_tensor, msk_levels = self.conv(
                    conv_model, 
                    input_tensor.unsqueeze(0), 
                    self.input_ids[i], 
                    layer_id
                )

                tensors.append(ret_tensor)

                for m in range(MASK_LEVELS):
                    msk_levels_inputs[i][m] = msk_levels[m]

            # add faults with collisions back to the fault list
            # any repeated tiles (collision) in the fault list were not injected due to tile collision
            # if there was collisions, the fault list should be run again (externally) until it is exhausted
            fl.fault_list.extend(self.fault_list_collision.values())

        else: # sequential approach. there's a single fault stored in fl.next_fault (no fault lists are used here)
            for i, input_tensor in enumerate(batch_input_tensor):                  
                ret_tensor, msk_levels = self.conv(
                    conv_model, input_tensor.unsqueeze(0), 
                    self.input_ids[i], 
                    layer_id
                )

                tensors.append(ret_tensor)
                for m in range(MASK_LEVELS):
                    msk_levels_inputs[i][m] = msk_levels[m]

        merged_int = torch.cat([t.int_repr() for t in tensors], dim=0)
        
        merged_tensors = torch._make_per_tensor_quantized_tensor(
            merged_int, 
            tensors[0].q_scale(), 
            tensors[0].q_zero_point()
        )

        return merged_tensors, msk_levels_inputs