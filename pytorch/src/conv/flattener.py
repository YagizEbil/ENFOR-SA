import torch
import os
import torch.nn.functional as F
from src import definitions as defs

"""
def extract_conv_inputs(conv_model, input_tensor):
    if input_tensor.is_quantized:
        input_unfolded = im2col_quant(conv_model, input_tensor)
        weight_flat    = flatten_weight(conv_model)
    #else:
        #input_unfolded, weight_flat = im2col(conv_model, input_tensor)

    return input_unfolded.to(defs.GEMM_INPUT_DTYPE), weight_flat.to(defs.GEMM_INPUT_DTYPE)
"""

def im2col_quant(conv_model, input_tensor):
    #print(input_tensor.int_repr()); exit(0); # dtype=torch.uint8

    conv_weight = conv_model.weight()

    _, input_zp  = get_scales_and_zero_points(input_tensor)

    conv_weight  = conv_weight.int_repr()
    input_tensor = input_tensor.int_repr()
    
    if isinstance(input_zp, (int, float)):
        input_tensor = input_tensor - input_zp
    else:
        input_tensor = input_tensor - input_zp.view(-1, 1, 1, 1)

    # Output shape after unfolding: [batch_size, in_channels * kernel_height * kernel_width, output_height * output_width]
    input_unfolded = F.unfold(
        input_tensor.float(),
        kernel_size=conv_model.kernel_size, 
        stride=conv_model.stride, 
        padding=conv_model.padding, 
        dilation=conv_model.dilation)

    batch_size = input_tensor.shape[0]
    out_channels, in_channels, kernel_height, kernel_width = conv_weight.shape

    input_unfolded = input_unfolded.contiguous().view(
        batch_size, 
        in_channels * kernel_height * kernel_width, 
        -1)

    #return torch.transpose(input_unfolded[0], 0, 1), torch.transpose(conv_weight_flat, 0, 1)
    #return conv_weight_flat, input_unfolded[0]
    return input_unfolded[0]


def flatten_weight(conv_model):
    conv_weight = conv_model.weight()

    _, weight_zp = get_scales_and_zero_points(conv_weight)

    conv_weight  = conv_weight.int_repr()
    
    if isinstance(weight_zp, (int, float)): # e.g., MobileNet_V2 weight_zp is a scalar 
        conv_weight = conv_weight - weight_zp   
    else:
        conv_weight = conv_weight - weight_zp.view(-1, 1, 1, 1)  # reshape to (N, C, H, W)  

    out_channels = conv_weight.shape[0]

    # Reshape conv_weight to [out_channels, in_channels * kernel_height * kernel_width]
    conv_weight_flat = conv_weight.contiguous().view(out_channels, -1)  # Flatten weight for matmul

    return conv_weight_flat


def im2col(conv_model, input_tensor):
    conv_weight = conv_model.weight

    # Output shape after unfolding: [batch_size, in_channels * kernel_height * kernel_width, output_height * output_width]
    input_unfolded = F.unfold(
        input_tensor, 
        kernel_size=conv_model.kernel_size, 
        stride=conv_model.stride, 
        padding=conv_model.padding, 
        dilation=conv_model.dilation)

    batch_size = input_tensor.shape[0]
    out_channels, in_channels, kernel_height, kernel_width = conv_weight.shape

    # Reshape conv_weight to [out_channels, in_channels * kernel_height * kernel_width]
    conv_weight_flat = conv_weight.contiguous().view(out_channels, -1)  # Flatten weight for matmul

    input_unfolded = input_unfolded.contiguous().view(
        batch_size, 
        in_channels * kernel_height * kernel_width, 
        -1)

    return torch.transpose(input_unfolded[0], 0, 1), torch.transpose(conv_weight_flat, 0, 1)
    #return conv_weight_flat, input_unfolded[0]


def get_scales_and_zero_points (tensor):
    # extracts the scales and zero points for the input and the weight kernel (note the input and weights can have different quantization schemes)
    if tensor.qscheme()  == torch.per_tensor_affine: # here, scale and zero point are just scalars
        return tensor.q_scale(), tensor.q_zero_point()
  
    return tensor.q_per_channel_scales(), tensor.q_per_channel_zero_points()































# backups

""" this is not worth it. it makes the app about 13% slower
@torch.compile
def jit_unfold(tensor, kernel_size, stride, padding, dilation):
    return F.unfold(tensor, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation) 
"""
