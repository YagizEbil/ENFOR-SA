from src.gemmini import gemmini_config as c
from src.utils import utils as u
from src import definitions as defs

"""
    This file defines a global filter for desired the fault injection targets ('fault_target' below).
    The desired targets are defined by filtering by the column attributes in a fault list.
    Currently, the fault lists are defined according to the following headers:

     CNNs, RTL injection:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     tag   layer   tile_a_row  tile_a_col  tile_b_row  tile_b_col  target  pe_row  pe_col  bit cell
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     CNNs - SW injection:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     tag layer   target  N   C   H   W   bit
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     Transformers (ViT) - RTL injection:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     tag layer   block   head    tile_a_row  tile_a_col  tile_b_row  tile_b_col  target  pe_row  pe_col  bit cell
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     Transformers (ViT) - SW injection:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     tag layer   block   head    x   y   target  bit
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# fault targets are configured with a customized filter on the rows of the fault list, as shown in the examples below
# if a filter value is not set, it's ignored. e.g., by not setting the 'bit' value, all bits are considered for fault injection
ANY, NONE, ITVL, SIGNAL = u.ANY, u.NONE, u.ITVL, c.SIGNAL

example_generic_target = {
    #
    # the target conv layer (QuantizedConv2d, QuantizedConvReLU2d, etc...)
    #
    'layer': 2, # only layer 2 considered
    
    #
    # target PE position
    #
    'pe_row': 0, # targets only PEs of row 0
    'pe_col':(ANY, [0, 3, 5]), # e.g., considers only PE cols 0, 3 and 5

    #
    # the target bit in each PE
    #
    'bit': (NONE, [0, 3, 5]),  # e.g., ignores bits 0, 3 and 5

    #
    # the target signal in each PE (inputs, outputs or ctrl signals)
    #
    'target': (ANY, [c.IN_A, c.IN_B]), # inject only in the PE inputs
    #'target': SIGNAL["OUT_C"][0], # inject only in the PE outputs
    #'target': (ANY, [SIGNAL["SIG_PROPAG"][0], SIGNAL["SIG_VALID"][0], SIGNAL["SIG_DATAFLOW"][0]]), # inject only control signals
}

target_inputs_lsb = {
    'bit': 0,
    'target': (ANY, [c.IN_A, c.IN_B])
}

target_output_msb = {
    'bit': 7,
    'target': c.OUT_C
}

target_inputs = {
    'target': (ANY, [c.IN_A, c.IN_B])
}

target_outputs = {
    'target': c.OUT_C # OS only
}

target_control = {
    'target': (ANY, [c.SIG_PROPAG, c.SIG_VALID]) # inject only control signals
}

# SW only filters
target_sw_weights = {
    'target': 0
}

target_sw_activation = {
    'target': 1
}


""" Return a customized fault target. The target layer is passed as simul. paramenter --layer and stored in defs.TARGET_LAYER """
def setup_target():
    if defs.FI_GEMM:  # RTL injections in selected signal types
        #fault_target = target_inputs
        fault_target = target_outputs
        #fault_target = target_control
    else: # SW injections in activations only
        fault_target = target_sw_activation
    
    fault_target['layer'] = defs.TARGET_LAYER  # the target layer is passed as a simul. option at runtime.
     
    return fault_target
