import torch

MODE_OS = 0  # Output stationary
MODE_WS = 1  # Weight stationary


#
# CONFIG_PARAMS: stores the Gemmini configurations and the associated parameters
#   When porting new configs:
#       1 - the parameters must match the definitions in the configuration defined in the headers in rtl/lib/Gemmini/headers/configs/<config header file>
#       2 - the PE input and output data types must match the Input_t and Output_t definitions in the config header (rtl/lib/Gemmini/headers/configs/<config header file>)
#       3 - the path key must **match exactly** the folders of the verilated designs (defined in the script rtl/verilate.sh -> variable CONFIG)
CONFIG_PARAMS = {
    #
    # OS configs
    #
    "OSDIM4": 
    {
        "path":     "CustomGemminiOSTinySoCConfig", # the path (in folder "rtl/verilated") associated with the config (defined in rtl/verilate.sh, variable CONFIG)
        "dim":       4,                             # the SA dimention (used for tile extraction and manipulation)
        "in_bits":   8,                             # the SA MAC input bits (verilated model only). used to generate the fault lists
        "out_bits":  32,                            # the SA MAC output bits. used to generate the fault lists
        "in_dtype":  torch.int32,                   # the data type for the verilated model MAC input signals
        "out_dtype": torch.int32,                   # the data type for the verilated model MAC output signal
        "mode":      MODE_OS                        # the array execution mode (OS or WS)
    },

    "OSDIM8": 
    {
        "path":     "CustomGemminiOSMediumSoCConfig",
        "dim":       8,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    }, 

    "OSDIM16": 
    {
        "path":     "CustomGemminiOSBigSoCConfig",
        "dim":       16,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },

    "OSDIM32": 
    {
        "path":     "CustomGemminiOSHugeSoCConfig",
        "dim":       32,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },

    "OSDIM64": 
    {
        "path":     "CustomGemminiOSDIM64SoCConfig",
        "dim":       64,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },

    # Versions instrumented with the HDFIT tool for FI
    "OSDIM4_HDFIT": 
    {
        "path":     "CustomGemminiOSTinySoCConfig_RTL_HDFIT",
        "dim":       4,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },

    "OSDIM8_HDFIT": 
    {
        "path":     "CustomGemminiOSMediumSoCConfig_RTL_HDFIT",
        "dim":       8,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    }, 

    
    #"OSDIM8_HDFIT_GL": 
    #{
    #    "path":     "CustomGemminiOSMediumSoCConfig_GL_HDFIT",
    #    "dim":       8,  
    #    "in_bits":   8, 
    #    "out_bits":  32, 
    #    "in_dtype":  torch.int32, 
    #    "out_dtype": torch.int32, 
    #    "mode":      MODE_OS
    #}, 
   

    "OSDIM16_HDFIT": 
    {
        "path":     "CustomGemminiOSBigSoCConfig_RTL_HDFIT",
        "dim":       16,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },

    "OSDIM32_HDFIT": 
    {
        "path":     "CustomGemminiOSHugeSoCConfig_RTL_HDFIT",
        "dim":       32,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },

    "OSDIM64_HDFIT": 
    {
        "path":     "CustomGemminiOSDIM64SoCConfig_RTL_HDFIT",
        "dim":       64,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },

    # a toy fault-tolerant version with TMR
    #"OSDIM8_TMR": 
    #{
    #    "path":     "CustomGemminiOSMediumFTSoCConfig",
    #    "dim":       8,  
    #    "in_bits":   8, 
    #    "out_bits":  32, 
    #    "in_dtype":  torch.int32, 
    #    "out_dtype": torch.int32, 
    #    "mode":      MODE_OS
    #},

    #
    # WS configs
    #
    "WSDIM4": 
    {
        "path":     "CustomGemminiWSTinySoCConfig",
        "dim":       4,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_WS
    },

    "WSDIM8": 
    {
        "path":     "CustomGemminiWSMediumSoCConfig",
        "dim":       8,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_WS
    },

    "WSDIM64": 
    {
        "path":     "CustomGemminiWSTPUSoCConfig",
        "dim":       64,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_WS
    },

    "TestConfig": 
    {
        "path":     "TestConfig",
        "dim":       4,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },

    "TVU_SA": 
    {
        "path":     "TVU_SA",
        "dim":       4,  
        "in_bits":   8, 
        "out_bits":  32, 
        "in_dtype":  torch.int32, 
        "out_dtype": torch.int32, 
        "mode":      MODE_OS
    },
}

#
# Gemmini config options. This default value is overwritten in sim_options.py. Pass the --config <key> option flag to choose a config.
# All avalilable keys are in the dictionary CONFIG_PARAMS
# This key is used to create the gemmini hw interface, as in gemmini_device = gemmini_interface.GemminiOS(conf.CONFIG_KEY)

# OS configs
#CONFIG_KEY = "OSDIM4"
CONFIG_KEY =  "OSDIM8" 
#CONFIG_KEY = "OSDIM16"
#CONFIG_KEY = "OSDIM32"
#CONFIG_KEY = "OSDIM64"

# OS with TMR PEs
#CONFIG_KEY = "OSDIM8_TMR"

# OS configs with HDFIT FI instrumentation
#CONFIG_KEY = "OSDIM4_HDFIT"
#CONFIG_KEY = "OSDIM8_HDFIT"
#CONFIG_KEY = "OSDIM16_HDFIT"
#CONFIG_KEY = "OSDIM32_HDFIT"
#CONFIG_KEY = "OSDIM64_HDFIT"

# WS configs
#CONFIG_KEY = "WSDIM4"
#CONFIG_KEY = "WSDIM8"


#
# Allows to access the parameters in CONFIG_PARAMS based on the config key
#
PARAM=CONFIG_PARAMS[CONFIG_KEY]

#
# The dimesion of the systolic array
#
DIM = PARAM["dim"]

#
# PE input and output bits
#
PE_IN_BITS  = PARAM["in_bits"]
PE_OUT_BITS = PARAM["out_bits"]

#
# PE input and output data types
#
GEMM_INPUT_DTYPE  = PARAM["in_dtype"]
GEMM_OUTPUT_DTYPE = PARAM["out_dtype"]

#
# The type of Gemmini config (OS/WS)
#
GEMM_MODE = PARAM["mode"]

#
# The design path for the chosen config in the "verilated" folder
#
DESIGN_PATH = PARAM["path"] 


# re-defines all parameters above based on the config. conf_key must be any of the configs listed in ext.CONFIG_KEYS (see CONFIG_PARAMS)
def set_config(conf_key):
    global CONFIG_KEY
    global PARAM
    global DIM
    global PE_IN_BITS
    global PE_OUT_BITS
    global GEMM_INPUT_DTYPE
    global GEMM_OUTPUT_DTYPE
    global GEMM_MODE
    global DESIGN_PATH

    CONFIG_KEY = conf_key
    PARAM = CONFIG_PARAMS[CONFIG_KEY]
    DESIGN_PATH = PARAM["path"] 
    DIM = PARAM["dim"]
    PE_IN_BITS  = PARAM["in_bits"]
    PE_OUT_BITS = PARAM["out_bits"]
    GEMM_INPUT_DTYPE  = PARAM["in_dtype"]
    GEMM_OUTPUT_DTYPE = PARAM["out_dtype"]
    GEMM_MODE = PARAM["mode"]




"""
SIGNAL: The target inputs for fault injection. For Gemmini, this must follow the values defined in the FaultList.h enum 
(see in /rtl/lib/Gemmini/headers/FaultList.h)

The signal identifiers are
enum {
    IDX_io_in_a,   // 0 - Gemmini input A (flows left->right)
    IDX_io_in_b,   // 1 - Gemmini input B (flows top->bottom)
    IDX_io_in_d,   // 2 - Gemmini input D (flows top->bottom)
    IDX_io_out_b,  // 3 - OS: input b passes through during stream and preloads. WS: partial sums during stream and preloads
    IDX_io_out_c,  // 4 - OS: in_d in preloads, MAC out_d during stream. WS: in_d during preloads (actuall input B), MAC out_d during stream
    IDX_propagate, // 5 - propagate ctrl signals to assert in_b and in_d pins (flows top->bottom)
    IDX_valid,     // 6 - valid signals to assert in_b and in_d pins (flows top->bottom)
    IDX_io_out_a   // 7
    IDX_c1,        // 8 - the c1/c2 registers holding preloaded values - check PE.scala to understand this better
    IDX_c2         // 9
};
"""

IN_A=0
IN_B=1
IN_D=2

OUT_A=7
OUT_B=3
OUT_C=4

C1=8
C2=9

SIG_PROPAG=5
SIG_VALID=6

SIG_NAMES = {
    IN_A: "IN_A",
    IN_B: "IN_B",
    IN_D: "IN_D",
    OUT_A: "OUT_A",
    OUT_B: "OUT_B",
    OUT_C: "OUT_C",
    C1: "C1",
    C2: "C2",
    SIG_PROPAG: "PROPAG",
    SIG_VALID: "VALID",
}


PE_IN_BITS = 8
PE_OUT_BITS = 32

#
# Maps in the form <signal name>: (<enum index value as shown above>, <number of bits for injection in the signal>)
#
SIGNAL = {
    # data signals  - PE inputs
    "IN_A":  (IN_A, PE_IN_BITS),  # input A signal id is 0, with PE_IN_BITS bits
    "IN_B":  (IN_B, PE_IN_BITS),  # input B signal id is 1, with PE_IN_BITS bits
    "IN_D":  (IN_D, PE_OUT_BITS), # input D
    
    # data signals - PE outputs
    "OUT_A": (OUT_A, PE_IN_BITS),  
    "OUT_B": (OUT_B, PE_OUT_BITS), 
    "OUT_C": (OUT_C, PE_OUT_BITS),

    # data signals - each PE has two registers to store: 1. accumulators in OS, or 2. weights in WS - in each case, only one reg. is actually used
    "C1":   (C1, PE_OUT_BITS),
    "C2":   (C2, PE_OUT_BITS),

    # control signals
    "SIG_PROPAG":   (SIG_PROPAG, 1),
    "SIG_VALID":    (SIG_VALID,  1),
}

#TARGETS_IDS  = [t[0] for t in SIGNAL.values()]
#TARGETS_BITS = [t[1] for t in SIGNAL.values()]


