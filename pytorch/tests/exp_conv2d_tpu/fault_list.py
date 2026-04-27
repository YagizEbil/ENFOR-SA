import random
import csv
import os
import sys
import random
import pandas as pd
from dataclasses import dataclass
from collections import deque
from collections import defaultdict
from typing import Deque, List, Union, Tuple, Optional

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.conv import tile_ops as tile_ops
from src.utils import utils as u


# HDFIT assignment IDs for GL fault injection
HDFIT_FIRST_TARGET_ID = 2     # First one is assign _0056_[16] =( io_in_a[8] & io_in_b[8]) ^ ((fiEnable && (2 == GlobalFiNumber)) ? GlobalFiSignal[0] : {1{1'b0}});
#HDFIT_LAST_TARGET_ID = 3419  # 32/32-bit inputs/outputs (unsigned mac): Last one is assign _0184_[0] =( _0182_[0]) ^ ((fiEnable && (3419 == GlobalFiNumber)) ? GlobalFiSignal[0] : {1{1'b0}});
#HDFIT_LAST_TARGET_ID  = 591  # 8/32-bit inputs/outputs (unsigned mac) 
HDFIT_LAST_TARGET_ID  = 824   # 8/32-bit inputs/outputs (signed mac)


# the status of each fault in the fault list
@dataclass
class FaultStatus:
    msk_gemm:  bool = False  # if the fault was masked by Gemmini (note: masked in the **fi layer** )
    msk_scale: bool = False  # ... masked by scaling
    msk_round: bool = False  # ... masked by rounding
    msk_clamp: bool = False  # ... masked by clamping
    critical:  bool = False  # if the fault caused a top-1 misclassification
    corrupted_elements: int=0

# the tile positions gemmini will perform the matmul
# the fault position depends on which of the tiles (A or B) is selected for fault injection
# the output tile position is (a_row, b_col) -> sweeps all A cols of row a_row, and all B rows of col b_col 
@dataclass
class TilePos:
    a_row: int
    a_col: int
    b_row: int
    b_col: int

# the gemmini positions
@dataclass
class GemminiPos:
    target: int # e.g., in_a, in_b, etc...
    pe_row: int 
    pe_col: int
    bit:    int
    cell:   int=-1 # the cell to inject. only valid if using GL MAC modules

@dataclass
class Fault:
    tag: int             # the fault id
    layer_id: int        # target layer
    tile: TilePos        # target tile 
    gemm: GemminiPos     # target Gemmini position
    status: FaultStatus  # the fault status in the injection layer (**not the status in the model classification)

    def __str__(self):
        return f"Tag {self.tag} - Layer: {self.layer_id} - {self.tile} {self.gemm}"


# returns a new fault position (a Fault object) generated at runtime
def gen_random_fault(shape_A: Tuple[int, int], 
                     shape_B:Tuple[int, int], 
                     layer_id:int, 
                     possible_targets: List[int],
                     DIM:int) -> Fault:

    rows_A, cols_A = shape_A
    rows_B, cols_B = shape_B

    # the tile positions
    tile_a_rand_row, tile_a_rand_col = tile_ops.get_random_tile_position (rows_A, cols_A, DIM)
    tile_b_rand_row, tile_b_rand_col = tile_ops.get_random_tile_position (rows_B, cols_B, DIM)
    tile_b_rand_row = tile_a_rand_col  # overwrite this. tile_b row must be equal tile_a col

    n  = len(possible_targets)
    rd = random.randint(0, n-1)
    target = possible_targets[rd][0]
    bits   = possible_targets[rd][1]

    # the pe positions
    pe_row, pe_col = random.randint(0, DIM-1), random.randint(0, DIM-1)
    
    # the target bit
    bit = random.randint(0, bits-1)  # randint(min, max) -> min and max are both included

    cell = random.randint(HDFIT_FIRST_TARGET_ID, HDFIT_LAST_TARGET_ID)

    vals = [
        tile_a_rand_row, 
        tile_a_rand_col, 
        tile_b_rand_row, 
        tile_b_rand_col,
        target, 
        pe_row, 
        pe_col, 
        bit, 
        cell
    ]

    return Fault(
        42, 
        layer_id, 
        TilePos(*vals[0:4]), 
        GemminiPos(*vals[4:9]), 
        FaultStatus(False, False, False)
        )


# receives the shape of the two unfolded tensors (each of shape [rows][cols])
# creates a fault list in the form:
# layer_no, tile_a_row, tile_a_col, tile_b_col, target, pe_row, pe_col, bit
def generate_fault_list(
    layer_id, 
    shape_A, 
    shape_B, 
    start_tag, 
    possible_targets: List[int], 
    max_faults=10000,
    DIM=8) -> List[Fault]:

    rows_A, cols_A = shape_A
    rows_B, cols_B = shape_B

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions do not match for multiplication.")

    new_fault_list = []
    # use a fixed random if required
    random.seed(42)

    # start_tag: when generating the fault list for all layers generate_fault_list() is called once per layer. start_tag allows to keep the tag counter accross all layers
    l = 0

    while l < max_faults:
        random_fault = gen_random_fault(shape_A, shape_B, layer_id, possible_targets, DIM)
        random_fault.tag = start_tag + l
        new_fault_list.append(random_fault)
        l += 1

    return new_fault_list


# dumps the fault list to a file
def dump_fault_list (flist: List[Fault], filename="default.csv"):
    header = [
        'tag', 
        'layer', 
        'tile_a_row', 
        'tile_a_col', 
        'tile_b_row', 
        'tile_b_col', 
        'target', 
        'pe_row', 
        'pe_col', 
        'bit', 
        'cell'
    ]

    #print(f"Dumping fault list to file {filename} - size {len(flist)}")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header)

        for fault in flist:
            csv_row = [
                fault.tag, 
                fault.layer_id,
                fault.tile.a_row, 
                fault.tile.a_col,
                fault.tile.b_row, 
                fault.tile.b_col,
                fault.gemm.target, 
                fault.gemm.pe_row,
                fault.gemm.pe_col,
                fault.gemm.bit, 
                fault.gemm.cell
                ]

            writer.writerow(csv_row)

    print(f"Saved fault list: {filename}  -  Size: {len(flist)} rows")


# loads the fault list in a given line range (line zero is always skipped)
# the list is read from the file created by the generate_fault_list() function
# the fault list is returned in form of a deque
def load_fault_list(
    file_path: str, 
    itv_tuple: Union[int, Tuple[int, int]], # itvl tuple = (row_start, row_end)
    filters: Optional[dict]=None, 
    shuffle_list=False) -> deque[Fault]: 

    if not os.path.exists(file_path):
        raise ValueError(f"Missing fault list: {file_path}")

    if filters:
        df = u.filter_csv(in_fn=file_path, filters=filters)
    else:
        df = pd.read_csv(file_path, comment='#', sep='\t')   

    row_start, row_end = (itv_tuple, itv_tuple) if isinstance(itv_tuple, int) else itv_tuple

    filtered_rows = df.iloc[row_start:row_end+1]

    # Convert each row into a list and store in a list
    fault_list_rows = filtered_rows.values.tolist()

    if shuffle_list:
        random.shuffle(fault_list_rows)
                
    new_fault_list = deque()

    for vals in fault_list_rows: 
        new_fault_list.append(
            Fault(vals[0], 
            vals[1], 
            TilePos(*vals[2:6]),
            GemminiPos(*vals[6:11]), 
            FaultStatus(False, False, False)
            )
        )

    #print(f"Returning fault list of size {len(new_fault_list)}")
    return new_fault_list