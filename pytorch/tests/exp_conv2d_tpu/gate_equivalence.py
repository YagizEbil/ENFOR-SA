
"""
    Gate equivalents (GE_USED): (min, max) GE_USED equivalence w.r.t to a NAND2 cell

    The GL MAC unit report from yosys:
        983 wires
        2633 wire bits
        210 public wires
        1860 public wire bits
        4 ports
        80 port bits
        773 cells
        369   $and
        8   $not
        152   $or
        244   $xor

    The number of gates for the GL mac unit

    Procedure to count it is as follows:
        export PATH="/home/rafaelbt/Downloads/oss-cad-suite-linux-x64-20251124/oss-cad-suite/bin/:$PATH"
        yosys -p "read_verilog MacUnit_netlist.v; write_json netlist.json"
        python count.py

    Result:
        Counter({'AND2': 369, 'XOR2': 244, 'OR2': 152, 'NOT1': 8})
"""

MAC_GATES = {
    "AND2": 369,
    "XOR2": 244,
    "NOT1": 8,
    "OR2":  152,
}


GATE_EQUIVALENT = {
    "standard": {
        "NAND2": 4/4, # this is the reference cell : 4 transistors
        "AND2": 6/4,
        "XOR2": 10/4, # can be up to 12
        "NOT1": 2/4,
        "OR2":  6/4,
        "SRAM_CELL": 1.5,
        "DFF": 22/4, # 16-24  transistors
    
        # not actually used
        #"full_adder_1b": 20,   # Ripple-carry adder (RCA) -> n-bit is (20n, 30n) 
        #"multiplier_co_1b": 5, # Basic combinational multiplier -> n-bit is (n*n*5, n*n*10),
        #"multiplier_bw_1b": 3,  # Booth/Wallace. -> n-bit is (n*n*3, n*n*6),
    }
}

used_key = "standard"

GE_USED = GATE_EQUIVALENT[used_key]

use_max=False

def get_ge_buffer(nbits=1024*1024):
    # the input image is 1024 * 1024. each item is a single one-channel pixel, i.e., 1 byte
    return GE_USED["SRAM_CELL"]*nbits


def get_ge_dff(nbits):
    return GE_USED["DFF"]*nbits


def get_ge_adder(nbits, use_max=True):
    return GE_USED["full_adder_1b"]*nbits


def get_ge_mult_co(nbits, use_max=True):
    return GE_USED["multiplier_co_1b"]*nbits*nbits


def get_ge_mult_bw(nbits, use_max=True):
    return GE_USED["multiplier_bw_1b"]*nbits*nbits


def get_ge_mac_unit():
    return MAC_GATES["AND2"] * GE_USED["AND2"] +\
           MAC_GATES["XOR2"] * GE_USED["XOR2"] +\
           MAC_GATES["NOT1"] * GE_USED["NOT1"] +\
           MAC_GATES["OR2"]  * GE_USED["OR2"]


