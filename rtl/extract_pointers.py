import re
import sys

"""
This tool parses an input Mesh.sv input file generated for the Gemmini systolica array, 
and outputs the C++ pointers to the equivalent verilated C++ model. The pointers point to the 
input and output registers of each PE. These registers can be then accessed directly in the C++ code.


Example, for tile_0_0 of a given Mesh:

  Tile mesh_0_0 (   // @[Mesh.scala:39:71]
    .io_in_a_0                  (r_0),  // @[Reg.scala:19:16]
    .io_in_b_0                  (b_0),  // @[Reg.scala:19:16]
    .io_in_d_0                  (b_16_0),   // @[Reg.scala:19:16]
     ...
     ...
    .io_out_a_0                 (_mesh_0_0_io_out_a_0),
    .io_out_c_0                 (_mesh_0_0_io_out_c_0),
    .io_out_b_0                 (_mesh_0_0_io_out_b_0),
  );

  Tile mesh_0_1 (   // @[Mesh.scala:39:71]
    .io_in_a_0                  (r_1_0),    // @[Reg.scala:19:16]
    .io_in_b_0                  (b_4_0),    // @[Reg.scala:19:16]
    .io_in_d_0                  (b_20_0),   // @[Reg.scala:19:16]
    ...
    ...
    .io_out_a_0                 (_mesh_0_1_io_out_a_0),
    .io_out_c_0                 (_mesh_0_1_io_out_c_0),
    .io_out_b_0                 (_mesh_0_1_io_out_b_0),

  );

  Tile mesh_3_3 (   // @[Mesh.scala:39:71]
    .io_in_a_0                  (r_15_0),   // @[Reg.scala:19:16]
    .io_in_b_0                  (b_15_0),   // @[Reg.scala:19:16]
    .io_in_d_0                  (b_31_0),   // @[Reg.scala:19:16]
    ...
    ...
    .io_out_a_0                 (_mesh_3_3_io_out_a_0),
    .io_out_c_0                 (_mesh_3_3_io_out_c_0),
    .io_out_b_0                 (_mesh_3_3_io_out_b_0),

  );

 r_1_0 <= _mesh_0_0_io_out_a_0;   (1)
 r_2_0 <= _mesh_0_1_io_out_a_0;   (2)
 b_1_0 <= _mesh_0_0_io_out_b_0;   (3)

The parser output will be:

void * const in_a[DIM][DIM] = {
  { &dut->rootp->Mesh__DOT__r_0, &dut->rootp->Mesh__DOT__r_1_0, &dut->rootp->Mesh__DOT__r_2_0, &dut->rootp->Mesh__DOT__r_3_0 },
  { &dut->rootp->Mesh__DOT__r_4_0, &dut->rootp->Mesh__DOT__r_5_0, &dut->rootp->Mesh__DOT__r_6_0, &dut->rootp->Mesh__DOT__r_7_0 },
  { &dut->rootp->Mesh__DOT__r_8_0, &dut->rootp->Mesh__DOT__r_9_0, &dut->rootp->Mesh__DOT__r_10_0, &dut->rootp->Mesh__DOT__r_11_0 },
  { &dut->rootp->Mesh__DOT__r_12_0, &dut->rootp->Mesh__DOT__r_13_0, &dut->rootp->Mesh__DOT__r_14_0, &dut->rootp->Mesh__DOT__r_15_0 }
};

void * const in_b[DIM][DIM] = {
  { &dut->rootp->Mesh__DOT__b_0, &dut->rootp->Mesh__DOT__b_4_0, &dut->rootp->Mesh__DOT__b_8_0, &dut->rootp->Mesh__DOT__b_12_0 },
  { &dut->rootp->Mesh__DOT__b_1_0, &dut->rootp->Mesh__DOT__b_5_0, &dut->rootp->Mesh__DOT__b_9_0, &dut->rootp->Mesh__DOT__b_13_0 },
  { &dut->rootp->Mesh__DOT__b_2_0, &dut->rootp->Mesh__DOT__b_6_0, &dut->rootp->Mesh__DOT__b_10_0, &dut->rootp->Mesh__DOT__b_14_0 },
  { &dut->rootp->Mesh__DOT__b_3_0, &dut->rootp->Mesh__DOT__b_7_0, &dut->rootp->Mesh__DOT__b_11_0, &dut->rootp->Mesh__DOT__b_15_0 }
};

void * const out_a[DIM][DIM] = {
  { &dut->rootp->Mesh__DOT__r_1_0, &dut->rootp->Mesh__DOT__r_2_0, &dut->rootp->Mesh__DOT__r_3_0, NULL },
  { &dut->rootp->Mesh__DOT__r_5_0, &dut->rootp->Mesh__DOT__r_6_0, &dut->rootp->Mesh__DOT__r_7_0, NULL },
  { &dut->rootp->Mesh__DOT__r_9_0, &dut->rootp->Mesh__DOT__r_10_0, &dut->rootp->Mesh__DOT__r_11_0, NULL },
  { &dut->rootp->Mesh__DOT__r_13_0, &dut->rootp->Mesh__DOT__r_14_0, &dut->rootp->Mesh__DOT__r_15_0, NULL }
};

void * const out_b[DIM][DIM] = {
  { &dut->rootp->Mesh__DOT__b_1_0, &dut->rootp->Mesh__DOT__b_5_0, &dut->rootp->Mesh__DOT__b_9_0, &dut->rootp->Mesh__DOT__b_13_0 },
  { &dut->rootp->Mesh__DOT__b_2_0, &dut->rootp->Mesh__DOT__b_6_0, &dut->rootp->Mesh__DOT__b_10_0, &dut->rootp->Mesh__DOT__b_14_0 },
  { &dut->rootp->Mesh__DOT__b_3_0, &dut->rootp->Mesh__DOT__b_7_0, &dut->rootp->Mesh__DOT__b_11_0, &dut->rootp->Mesh__DOT__b_15_0 },
  { &dut->rootp->Mesh__DOT__r_64_0, &dut->rootp->Mesh__DOT__r_70_0, &dut->rootp->Mesh__DOT__r_76_0, &dut->rootp->Mesh__DOT__r_82_0 }
};


Here we have:

in_a:
    PE00: Mesh__DOT__r_0
    PE01: Mesh__DOT__r_1_0
    PE33: Mesh__DOT__r_15_0

in_b:
    PE00: Mesh__DOT__b_0
    PE01: Mesh__DOT__b_4_0
    PE33: Mesh__DOT__b_15_0

out_a:
    the output signals in the verilog code are wires that are then mapped to registers, as shown in the examples.
    the pointers will then point to these registers

    PE00: Mesh__DOT__r_1_0 (1)
    PE01: Mesh__DOT__r_2_0 (2)
    PE33: NULL, because this is the last column of the array, and is not passed to any register
"""


if len(sys.argv) < 3:
    print(f"Syntax: {sys.argv[0]}  <input Mesh.sv file> <output C file>")
    exit(0)

VERILOG_FILE = sys.argv[1]
OUT_C_FILE   = sys.argv[2]


# --- Regexes ---

# Tile mesh_i_j ( ... );
tile_re = re.compile(
    r"Tile\s+mesh_(\d+)_(\d+)\s*\((.*?)\);\s*",
    re.S
)

# .port(signal)
port_re = re.compile(
    r"\.(\w+)\s*\(\s*([^)]+)\s*\)"
)

# reg <= signal;
assign_re = re.compile(
    r"(\w+)\s*<=\s*(\w+)\s*;"
)


# --- Parse Verilog ---

with open(VERILOG_FILE) as f:
    verilog = f.read()

# Build map: output_wire -> register
out_to_reg = {}
for lhs, rhs in assign_re.findall(verilog):
    out_to_reg[rhs] = lhs

tiles = {}

for m in tile_re.finditer(verilog):
    row = int(m.group(1))
    col = int(m.group(2))
    body = m.group(3)

    ports = dict(port_re.findall(body))
    tiles[(row, col)] = ports

# Infer DIM
rows = max(r for r, c in tiles.keys()) + 1
cols = max(c for r, c in tiles.keys()) + 1
DIM = max(rows, cols)

# --- Emit C ---



# if the input is passed uses concatenation, e.g, something like ".io_in_b_0 ({{12{pipe_b_0[7]}}, pipe_b_0})"
# we then extract only the signal name, e.g, io_in_b_0
                
def extract_pipe_signal(s):
    # Match signal names like pipe_b_12_0 optionally followed by [number]
    match = re.search(r'([a-zA-Z_]\w*)\[\d+\]|([a-zA-Z_]\w*)', s)
    
    if match:
        # Return whichever group matched
        return match.group(1) if match.group(1) else match.group(2)

    return None


def emit_array(f, c_type, name, port, use_reg=False):
    f.write(f"{c_type} * const {name}[DIM][DIM] = {{\n")
    for i in range(DIM):
        f.write("  { ")
        row_elems = []
        for j in range(DIM):
            sig = tiles.get((i, j), {}).get(port)

            if sig is None:
                row_elems.append("NULL")
                continue

            if use_reg: # for output signals, we export the register assigned by the output wire
                reg = out_to_reg.get(sig)
                
                if reg is None:
                    row_elems.append("NULL")
                else:
                    row_elems.append(f"&dut->rootp->Mesh__DOT__{reg}")
            else:
                if "{" in sig:
                    sig = extract_pipe_signal(sig)

                row_elems.append(f"&dut->rootp->Mesh__DOT__{sig}")

        f.write(", ".join(row_elems))
        f.write(" }")
        if i != DIM - 1:
            f.write(",")
        f.write("\n")
    f.write("};\n\n")


OBJ_PTR = "&dut->rootp"


def gen_pe_io_out_d_pointers():

    var_decl = "void * const pe_io_out[DIM][DIM] = \n"

    print(f"write to file {OUT_C_FILE}")
    with open(OUT_C_FILE, mode='a', newline='') as f:

        f.write(var_decl)
        f.write("{\n")

        for i in range(DIM):
            f.write("\t{\n")

            for j in range(DIM):
                f.write(f"\t\t{OBJ_PTR}->Mesh__DOT__mesh_{i}_{j}__DOT__tile_0_0__DOT___mac_unit_io_out_d,\n")

            f.write("\t},\n")
        f.write("};\n\n")


# generates the PE C2 pointers

"""
IData/*30:0*/ Mesh__DOT__mesh_0_1__DOT__tile_0_0__DOT__c1_tvu__DOT___vandorOut_rVan2_m_q;
IData/*30:0*/ Mesh__DOT__mesh_0_1__DOT__tile_0_0__DOT__c1_tvu__DOT___vandorOut_rVan1_m_q;
Mesh__DOT__mesh_0_2__DOT__tile_0_0__DOT___c1_tvu_io_out


IData/*31:0*/ Mesh__DOT__mesh_0_1__DOT__tile_0_0__DOT__c2_tvu__DOT__unprotOut;
"""

def gen_pe_c1_c2_pointers_ft():

    var_decl = "void * const pe_c1[DIM][DIM] = \n"

    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")

        for i in range(DIM):
            f.write("\t{\n")

            for j in range(DIM):
                f.write(f"\t\t{OBJ_PTR}->Mesh__DOT__mesh_{i}_{j}__DOT__tile_0_0__DOT___c1_tvu_io_out,\n")
                                       # Mesh__DOT__mesh_0_2    __DOT__tile_0_0__DOT___c1_tvu_io_out
            f.write("\t},\n")
        f.write("};\n\n")


    var_decl = "void * const pe_c2[DIM][DIM] = \n"

    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")

        for i in range(DIM):
            f.write("\t{\n")

            for j in range(DIM):
                f.write(f"\t\t{OBJ_PTR}->Mesh__DOT__mesh_{i}_{j}__DOT__tile_0_0__DOT__c2_tvu__DOT__unprotOut,\n")
                                       # Mesh__DOT__mesh_0_1    __DOT__tile_0_0__DOT__c2_tvu__DOT__unprotOut
            f.write("\t},\n")
        f.write("};\n\n")

def gen_pe_c1_c2_pointers():

    var_decl = "void * const pe_c1[DIM][DIM] = \n"

    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")

        for i in range(DIM):
            f.write("\t{\n")

            for j in range(DIM):
                f.write(f"\t\t{OBJ_PTR}->Mesh__DOT__mesh_{i}_{j}__DOT__tile_0_0__DOT__c1,\n")

            f.write("\t},\n")
        f.write("};\n\n")


    var_decl = "void * const pe_c2[DIM][DIM] = \n"

    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")

        for i in range(DIM):
            f.write("\t{\n")

            for j in range(DIM):
                f.write(f"\t\t{OBJ_PTR}->Mesh__DOT__mesh_{i}_{j}__DOT__tile_0_0__DOT__c2,\n")

            f.write("\t},\n")
        f.write("};\n\n")



# Mesh-level pointers 
def gen_mesh_pointers():
    # mesh in_a
    var_decl = "void * const mesh_in_a[DIM] = \n"
   
    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")
        for i in range(DIM):
            f.write(f"\t\t{OBJ_PTR}->io_in_a_{i}_0,\n")
        f.write("};\n\n")

    # mesh in_b
    var_decl = "void * const mesh_in_b[DIM] = \n"
   
    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")
        for i in range(DIM):
            f.write(f"\t\t{OBJ_PTR}->io_in_b_{i}_0,\n")
        f.write("};\n\n")


    # mesh mesh_in_d
    var_decl = "void * const mesh_in_d[DIM] = \n"
   
    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")
        for i in range(DIM):
            f.write(f"\t\t{OBJ_PTR}->io_in_d_{i}_0,\n")
        f.write("};\n\n")


    # mesh mesh_out_b
    var_decl = "void * const mesh_out_b[DIM] = \n"
   
    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")
        for i in range(DIM):
            f.write(f"\t\t{OBJ_PTR}->io_out_b_{i}_0,\n")
        f.write("};\n\n")


    # mesh mesh_out_c
    var_decl = "void * const mesh_out_c[DIM] = \n"
   
    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")
        for i in range(DIM):
            f.write(f"\t\t{OBJ_PTR}->io_out_c_{i}_0,\n")
        f.write("};\n\n")


    # mesh valid
    var_decl = "void * const mesh_valid[DIM] = \n"
   
    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")
        for i in range(DIM):
            f.write(f"\t\t{OBJ_PTR}->io_in_valid_{i}_0,\n")
        f.write("};\n\n")


    # mesh mesh propagate
    var_decl = "void * const mesh_propagate[DIM] = \n"
   
    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")
        for i in range(DIM):
            f.write(f"\t\t{OBJ_PTR}->io_in_control_{i}_0_propagate,\n")
        f.write("};\n\n")

 
    # shift signals (OS only)
    """
    var_decl = "void * const mesh_shift[DIM] = \n"
   
    with open(OUT_C_FILE, mode='a', newline='') as f:
        f.write(var_decl)
        f.write("{\n")
        for i in range(DIM):
            f.write(f"\t\t{OBJ_PTR}->io_in_control_{i}_0_shift,\n")
        f.write("};\n\n")
    """
    
def parse_verilog():
    with open(OUT_C_FILE, "w") as f:
        f.write("// Auto-generated from Verilog mesh\n\n")
        
        # this is already defined earlier. no need to put this again in the pointers file
        # f.write(f"#define DIM {DIM}\n\n")

        emit_array(f, "void", "in_a",  "io_in_a_0")
        emit_array(f, "void", "in_b",  "io_in_b_0")
        emit_array(f, "void", "in_d",  "io_in_d_0")

        emit_array(f, "void", "in_valid",  "io_in_valid_0")
        emit_array(f, "void", "in_propag", "io_in_control_0_propagate")
        
        # OS only
        #emit_array(f, "void", "in_shift",  "io_in_control_0_shift")

        # outputs use registers
        emit_array(f, "void", "out_a", "io_out_a_0", use_reg=True)  # for output signals, we export the register assigned by the output wire
        emit_array(f, "void", "out_c", "io_out_c_0", use_reg=True)
        emit_array(f, "void", "out_b", "io_out_b_0", use_reg=True)


parse_verilog()

gen_pe_io_out_d_pointers() # OS only
#gen_pe_c1_c2_pointers()
gen_pe_c1_c2_pointers_ft()

gen_mesh_pointers()


print(f"Wrote {OUT_C_FILE} with DIM={DIM}")
