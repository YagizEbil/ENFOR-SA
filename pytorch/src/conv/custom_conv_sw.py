import torch
from torch._ops import ops
from src.flist.fl import fl
import src.definitions as defs
#
# the type of convolution the custom_conv object represents (also needed in instrumented_model for the layer replacement)
#
CONV, CONV_RELU = 0, 1


# sw fault injection 
def conv_batch_sw_fi(layer, input_tensor):
    if defs.TREE_FI_MODE: 
        input_tensor = fl.tensor_fi_input_batch_quantized_all_list(input_tensor)
    else:
        input_tensor = fl.tensor_fi_input_batch_quantized(input_tensor)       
    
    ret_tensor = ops.quantized.conv2d(input_tensor, layer._packed_params, layer.scale, layer.zero_point)

    return ret_tensor
