import torch
import torch.nn.functional as F


def flatten_weight_t(conv_weight):
    out_channels = conv_weight.shape[0]

    # Reshape conv_weight to [out_channels, in_channels * kernel_height * kernel_width]
    conv_weight_flat = conv_weight.contiguous().view(out_channels, -1)  # Flatten weight for matmul

    return conv_weight_flat.t_()


# im2col with padding="same"
# F.unfold doc: https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html

def im2col_tf_same_t(
    input_tensor, 
    conv_weight_shape, 
    stride=1, 
    dilation=1):
    
    assert input_tensor.shape[1] == conv_weight_shape[1]

    out_channels, in_channels, kh, kw = conv_weight_shape
    batch_size, _, H, W = input_tensor.shape
    
    if isinstance(stride, int):
        stride = (stride, stride)
    
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    sh, sw = stride
    dh, dw = dilation

    # Output spatial size (TensorFlow SAME)
    H_out = (H + sh - 1) // sh
    W_out = (W + sw - 1) // sw

    # Effective kernel
    kh_eff = dh * (kh - 1) + 1
    kw_eff = dw * (kw - 1) + 1

    pad_h = max((H_out - 1) * sh + kh_eff - H, 0)
    pad_w = max((W_out - 1) * sw + kw_eff - W, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    """
    print("pad_top:    ", pad_top)
    print("pad_bottom: ", pad_bottom)
    print("pad_left:   ", pad_left)
    print("pad_right:  ", pad_right)
    """
    
    # Apply asymmetric padding
    input_padded = F.pad(
        input_tensor,
        (pad_left, pad_right, pad_top, pad_bottom)
    )

    # Now unfold with zero padding
    input_unfolded = F.unfold(
        input_padded,
        kernel_size=(kh, kw),
        stride=stride,
        padding=0,
        dilation=dilation
    )

    return input_unfolded[0].t_()


def pe_first_cycle(row, col):
    return row + col


def pe_last_cycle(row, col, stream_size):
    return row + col + stream_size - 1