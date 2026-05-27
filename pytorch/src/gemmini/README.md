## Basic Gemmini API definitions
Using Gemmini modules from PyTorch calls can be trivially done using the simple API described in the snippets below. The core idea is to interface PyTorch with verilated models in C++ by using [C++ extensions for PyTorch]("https://docs.pytorch.org/docs/stable/cpp_extension.html"). PyTorch channels function calls to the verilated Gemmmini models through a binding interface (C++ extensions with pybind11) defined in [`pytorch/src/gemmini/cpp/gemmini_interface.cpp`](cpp/gemmini_interface.cpp). 

### Example on how to use the API for matmuls
To do so, one needs to first load the Gemmini module by specifying a given config. After this, just feed input tensors through the API and read back the resulting matmul. Examples are provided below:

Import the extension modules:
```
import torch
import src.gemmini.gemmini_extension_definitions as ext
import src.gemmini.gemmini_config as conf
```

Load the Gemmini extension modules:
```
# Example of OS mode, DIM 8×8
gemmini_os = ext.load_extension("OSDIM8")

# Example of WS mode, DIM 8×8
gemmini_ws = ext.load_extension("WSDIM8")

gemmini_os.init()
gemmini_ws.init()

gemmini_os.print_info()
gemmini_ws.print_info()
```

The `ext.load_extension(<>)` function call expects a configuration key as parameter. Check all available keys in [pytorch/src/gemmini/gemmini_config.py](gemmini_config.py)

Generate some random inputs:
```
# The dimensions must match the size of the array
A = torch.randint(-128, 127, (conf.DIM, conf.DIM), dtype=conf.GEMM_INPUT_DTYPE)
B = torch.randint(-128, 127, (conf.DIM, conf.DIM), dtype=conf.GEMM_INPUT_DTYPE)
D = torch.randint(-128, 127, (conf.DIM, conf.DIM), dtype=conf.GEMM_OUTPUT_DTYPE)
C = torch.zeros((conf.DIM, conf.DIM), dtype=conf.GEMM_OUTPUT_DTYPE)
```

Compute C = A×B + D with the OS mode:
```
# Preloads the D matrix
steps_pre = gemmini_os.preload(D)

# Streams the inputs
steps_mul = gemmini_os.stream(A, B)

# Flushes the PE outputs to the C tensor. Use True to flush the output transposed
steps_flu = gemmini_os.flush_gemm(C, False)
```

Compute C = A×B + D with the WS mode:
```
# Preloads the B matrix
steps_pre = gemmini_ws.preload(B)

# Streams the A and D inputs
steps_mul = gemmini_ws.stream(A, C) # use this if D is zero
steps_mul = gemmini_ws.stream_bias(A, D, C) # use this D is not zero
```

Free the RTL modules and interfaces
```
gemmini_os.finish()
gemmini_ws.finish()
```

### Adding transient faults
Transient faults can be added to Gemmini through an interface. The possible target signals are identified by a dictionary in [`/pytorch/src/gemmini/gemmini_config.py`](gemmini_config.py), in the variable `SIGNAL`, as follows:

```
IN_A=0
IN_B=1
IN_D=2
...
...
SIGNAL = {
    # Data signals  - PE inputs
    "IN_A":  (IN_A, PE_IN_BITS),  # input A signal id is 0, with PE_IN_BITS bits
    "IN_B":  (IN_B, PE_IN_BITS),  # input B signal id is 1, with PE_IN_BITS bits
    "IN_D":  (IN_D, PE_OUT_BITS), # input D signal (during preloads only)
    
    # Data signals - PE outputs
    "OUT_A": (OUT_A, PE_IN_BITS),
    "OUT_B": (OUT_B, PE_OUT_BITS),
    "OUT_C": (OUT_C, PE_OUT_BITS), # OS: accumulator. WS: Partial sum

    # Data signals - each PE has two state registers to store: 1. accumulators in OS, or 2. weights in WS - in each case, only one reg. is actually used
    "C1":   (C1, PE_OUT_BITS),
    "C2":   (C2, PE_OUT_BITS),

    # Control signals
    "SIG_PROPAG":   (SIG_PROPAG, 1), # the propagate ctrl signal
    "SIG_VALID":    (SIG_VALID,  1), # the valid ctrl signal
}
```

To add faults to Gemmini, do as follows:

```
import src.gemmini.gemmini_config as conf
from src.gemmini.gemmini_config import *

#
# Select a target signal. Examples are:
#
signal = SIGNAL["IN_A"]       # input register A
signal = SIGNAL["IN_B"]       # input register B
signal = SIGNAL["OUT_C"]      # accumulator output, for OS only
signal = SIGNAL["SIG_PROPAG"] # propagate control register
signal = SIGNAL["SIG_VALID"]  # valid control register

# the target register id
target = signal[0]

# the total number of bits in the register
nbits = signal[1]

# Select a PE position
pe_row = random.randint(0, conf.DIM-1)
pe_col = random.randint(0, conf.DIM-1)

# Select a random bit
tgt_bit = random.randint(0, nbits-1)

# prints the injected fault in the moment it's injected
fiSilent = True 

# Add the fault before streaming the inputs
gemmini_os.add_transient_fault(target, pe_row, pe_col, tgt_bit, 0, fiSilent) 

# Stream the inputs...
```

### Loading faults from the fault list
Faults can be loaded directly from lists and added to the Gemmini interface as shown in the following examples.

First, configure fault targets, as shown in [`pytorch/src/flist/fi_target.py`](../flist/fi_target.py), then:

Import the global fault list and fault targets
```
from src.flist import fault_list as fl
from src.flist import fi_target as fit
```

Setup the global fault targets ba calling
```
fit.setup_target()
```

Load the fault list
```
row_start, row_end = 0, 1000 # choose an interval to load over the full list
fault_list = fl.load_fault_list(
    <some fault list>.csv,
    (row_start, row_end),
    filters=fit.fault_target,
    shuffle=False)
```

Iterate over each fault in the list
```
for fault in fault_list:
    # Add the fault before streaming the inputs
    gemmini_os.add_transient_fault(
        fault.gemm.target,
        fault.gemm.pe_row, 
        fault.gemm.pe_col, 
        fault.gemm.bit,
        0, # reserved
        fiSilent)

    # Proceed using gemmini_os for matmuls
    ...
```

### Tips on OS
In OS  mode, one should reuse the outputs by keeping them in the internal PE accumulators. To do so, only flush the PE outputs when the final result is computed. For example, to compute:

```
# Example: C = A1×B1 + A2×B2 + A3×B3 + ... + An×Bn + D

# Preload the D matrix
steps_pre = gemmini_os.preload(D)

# Stream the inputs, and keep the partial outputs in the PE accumulators for reuse
steps_mul = gemmini_os.stream(A1, B1) 
steps_mul = gemmini_os.stream(A2, B2) 
steps_mul = gemmini_os.stream(A3, B3)
...
steps_mul = gemmini_os.stream(An, Bn)

# Only now you can flush the outputs
steps_flu = gemmini_os.flush_gemm(C, False)
```

### Streaming in OS
In OS mode, there's support for streaming the tensor A (left to right) and tensor B (top to bottom) with varying streaming sizes. The number of rows in A and the number of columns in B should always match the size of the systolic array (DIM). The 'streaming size', which is number of columns in A (matching the number of rows in B) can be of any size >= DIM. 

Using large streaming sizes allows one to compute C = A[DIM][N]×B[N][DIM], for N >>> DIM. This works by streaming the full A and B tensors to the SA, hence there's no need to tile A across columns or B across rows. 

For OS tiled matrix multiplication, this allows to offload the full matrix A row, and the full matrix B column to the SA.


### Tips on WS
In WS, reuse the weights as much as possible. For example, to compute multiple matmuls that share a common weights, like in the example below, do:

```
"""
Example:
    C1 = A1×W + D1
    C2 = A2×W + D2
    C3 = A3×W + D3
    ...
    Cn = An×W + Dn
"""

# Preload the W matrix
steps_pre = gemmini_ws.preload(W)

# Reuse the W tensor for all matmuls
steps_mul = gemmini_ws.stream_bias(A1, D1, C1)
steps_mul = gemmini_ws.stream_bias(A2, D2, C2)
steps_mul = gemmini_ws.stream_bias(A3, D3, C3)
...
steps_mul = gemmini_ws.stream_bias(An, Dn, Cn)
```

### Tiled matmuls
Computing large matrix multiplications with Gemmini requires tiling. 
The mapping of the tiles to the SA depends on the Gemmini mode (OS or WS). These tiling and mapping functions are available in [`pytorch/src/gemmini/gemmini_interface.py`](gemmini_interface.py). Check the `tiled_matmul` function. These tiling functions assure output reuse (for OS) or weight reuse (for WS). Check the examples provided [here](../../tests/test_perf.py), function `perf_measure_resnet50_first_layer()`.

## Higher level API
A high-level Gemmini API is available in [`pytorch/src/gemmini/gemmini_interface.py`](gemmini_interface.py). This is an alternative way to use Gemmini instead of making calls directly to the C++ extension interface. Follow the examples:

Load the API module:
```
import src.gemmini.gemmini_interface as gemmini_interface
```

Create Gemmini instances:
```
gemmini_os = gemmini_interface.GemminiOS("OSDIM8") # if the key is for OS mode
gemmini_ws = gemmini_interface.GemminiWS("WSDIM8") # if the key is for WS mode
```

Load large tensors:
```
A = torch.randint(-128, 127, (120, 42))
B = torch.randint(-128, 127, (42, 240))
D = torch.randint(-128, 127, (120, 240))
```

Multiply the tensors with tiled matmuls:
```
C = gemmini_os.tiled_matmul(A, B, D) 
C = gemmini_ws.tiled_matmul(A, B, D) 
```