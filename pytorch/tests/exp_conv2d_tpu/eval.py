import sys
import os
import plot
import numpy as np
import pandas as pd
import gate_equivalence as ge
from collections import defaultdict
from tabulate import tabulate

DIM = 64
PES = DIM*DIM


# signal identifiers
IN_A=0
IN_B=1

OUT_B=3

C1=8
C2=9

SIG_PROPAG=5
SIG_VALID=6

K=1024
M=K*K

SIG_NAMES = {
    IN_A: "IN_A",
    IN_B: "IN_B",
    OUT_B: "OUT_B",
    C1: "C1",
    C2: "C2",
    SIG_PROPAG: "PROPAG",
    SIG_VALID: "VALID",
}

GE_PER_SIGNAL = {
    IN_A: ge.get_ge_dff(8),
    IN_B: ge.get_ge_dff(8),
    OUT_B: ge.get_ge_dff(32),
    C1: ge.get_ge_dff(8),
    C2: ge.get_ge_dff(8),
    SIG_PROPAG: ge.get_ge_dff(1),
    SIG_VALID: ge.get_ge_dff(1),
}


def process_fault_trace():
    # Standard 2D Convolution - radiation pattern distribution
    tab_rad_prop = [
        ["Single", 63.4],
        ["Line", 2.5],
        ["Box", 34.2]
        ]

    hvf_global, hvf_sing_total, hvf_line_total, hvf_box_total = estimate_pattern_area_derated_hvf()
    
    hvfs = [hvf_sing_total, hvf_line_total, hvf_box_total]
    total = sum(hvfs)
    hvf_norm = [100*h/total for h in hvfs]
    
    tab_inj_prop = [
        ["Single", hvf_norm[0]],
        ["Line", hvf_norm[1]],
        ["Box", hvf_norm[2]]
        ] 

    print("Pattern distribution estimated with radiation (%):")
    print(tabulate(tab_rad_prop))

    print("Pattern distribution estimated with fault injection (%):")
    print(tabulate(tab_inj_prop))
    print()


# evaluates the complete fault trace and computes the per-hw-resource HVF 
# global HVF is estimated by taking into acount the gate equivalent of each hw resource type
def estimate_pattern_area_derated_hvf(genplots=True):
    fn = sys.argv[1]

    # Reads the trace file
    df_full = pd.read_csv(fn, comment='#', sep='\t')

    hvf_sing_component = defaultdict(int)
    hvf_line_component = defaultdict(int)
    hvf_boxe_component = defaultdict(int)
    hvf_total_component = defaultdict(int)
    area_frac_comp = defaultdict(int)
    inj_component = defaultdict(int)

    hvf_sing_total, hvf_line_total, hvf_box_total = 0, 0, 0

    injected_sig_names = []
    input_size = 1*M
    total_buff_size = 4*M

    # computes the gate equivalent hw parts
    ge_in_buff = ge.get_ge_buffer(nbits=total_buff_size)
    ge_mac_units = ge.get_ge_mac_unit()*PES
    
    ge_in_a = ge.get_ge_dff(8)*PES
    ge_psum = ge.get_ge_dff(32)*PES

    ge_propag = ge.get_ge_dff(1)*PES
    ge_valid  = ge.get_ge_dff(1)*PES
    ge_c2 = ge.get_ge_dff(8)*PES

    ge_array = ge_in_buff + ge_mac_units + ge_in_a + ge_psum + ge_propag + ge_valid + ge_c2

    total_area_frac_causing_single, total_area_frac_causing_line = 0, 0

    for target, df_comp in df_full.groupby('target'):
        # The total number of injections
        injected_sig_names.append(SIG_NAMES[target])

        inj_component[target] = len(df_comp)

        # df for single pattern critical faults 
        df_critical_comp_sing = df_comp[(df_comp['corrupted_elements'] == 1)]

        # df for line pattern critical faults
        df_critical_comp_line = df_comp[(df_comp['corrupted_elements'] > 1)]

        sdcs_comp_sing = len(df_critical_comp_sing)
        sdcs_comp_line = len(df_critical_comp_line)

        hvf_comp_sing = sdcs_comp_sing/inj_component[target]
        hvf_comp_line = sdcs_comp_line/inj_component[target]
        hvf_comp_tota = (sdcs_comp_sing + sdcs_comp_line)/inj_component[target]

        # the fraction of area os a given signal is approximated by: area_frac_comp = signal GE * PES / total array GE
        area_frac_comp[target] = PES*GE_PER_SIGNAL[target]/ge_array

        # stores the HVF for each pattern/signal type
        hvf_sing_component[target] = hvf_comp_sing
        hvf_line_component[target] = hvf_comp_line
        hvf_boxe_component[target] = 0  # Mesh signals do not generate any squares
        hvf_total_component[target] = hvf_comp_tota

        # the global HVFs is estimatd as described in the paper's equations
        hvf_sing_total += hvf_comp_sing*area_frac_comp[target]
        hvf_line_total += hvf_comp_line*area_frac_comp[target]

        if hvf_sing_component != 0:
            total_area_frac_causing_single += area_frac_comp[target]

        if hvf_comp_line != 0:
            total_area_frac_causing_line += area_frac_comp[target]

    pvf_box = 0.9338*0.80
    tgt_io_buff = 42 # any id
    area_frac_comp[tgt_io_buff] = ge_in_buff/ge_array

    hvf_sing_component[tgt_io_buff] = 0
    hvf_line_component[tgt_io_buff] = 0
    hvf_boxe_component[tgt_io_buff] = pvf_box*input_size/total_buff_size
    hvf_box_total = hvf_boxe_component[tgt_io_buff]*area_frac_comp[tgt_io_buff]

    # MAC units
    tgt_mac = 43 # any id
    hvf_mac_total = 0.23416385135135134 # from the GL experiments (gl_col_0_50k)
    
    injected_sig_names.append("io_buff")
    injected_sig_names.append("mac_unit")

    area_frac_comp[tgt_mac] = ge_mac_units/ge_array
    hvf_sing_component[tgt_mac] = hvf_mac_total
    hvf_line_component[tgt_mac] = 0
    hvf_boxe_component[tgt_mac] = 0

    # add the extra HVF_mac * mac area fracton to the global HVF_sing
    hvf_sing_total += hvf_sing_component[tgt_mac]*area_frac_comp[tgt_mac]

    hvf_global = hvf_sing_total + hvf_line_total + hvf_box_total

    """ HVF plots """
    if genplots:
        # signal order until here:   IN_A,            OUT_B,       PROPAG,         VALID,               C2,           io_buff,         mac_unit,
        injected_sig_names = ["Activation\nreg", "Weight\nreg", "Psum\nreg", "MAC logic\n(gates)", "Valid\nctrl", "Propag\nctrl", "Input\nbuffers"] # permuted order
        # i want it to be in the order above, so we permute as
        perm = [0, 4, 1, 6, 3, 2, 5]

        hvf_sing_as_list = list(hvf_sing_component.values())
        hvf_line_as_list = list(hvf_line_component.values())
        hvf_boxe_as_list = list(hvf_boxe_component.values())

        hvf1_perm = [100*hvf_sing_as_list[i] for i in perm]
        hvf2_perm = [100*hvf_line_as_list[i] for i in perm]
        hvf3_perm = [100*hvf_boxe_as_list[i] for i in perm]
        
        masked = [100 - (hvf1_perm[i] + hvf2_perm[i] + hvf3_perm[i]) for i in range(len(perm))]

        plot.plot_stacked_bars( 
            list1=hvf1_perm, 
            list2=hvf2_perm,
            list3=hvf3_perm,
            list4=masked,
            labels=injected_sig_names,
            category_names=["Single", "Line", "Box", "Masked"],
            )

        # SW injection PVF results
        df = pd.read_csv("./SWFI_Line_Faults/fault_summary.csv") 
        df["AVF"] = df['Critical SDC Rate'] * hvf_global * 100 

        plot.plot_grouped_bars(
            df, 
            col_groups="Benchmark",
            col_types="Type",
            col_values="AVF")

    """ END PLOT """


    tab_area_frac_causing_pattern = [
        ["Single: ", 100*(total_area_frac_causing_single + area_frac_comp[tgt_mac])],
        ["Line: ", 100*total_area_frac_causing_line],
        ["Box: ", 100*area_frac_comp[tgt_io_buff]]
    ]

    print("GE area causing a pattern (%):")
    print(tabulate(tab_area_frac_causing_pattern))


    return hvf_global, hvf_sing_total, hvf_line_total, hvf_box_total



if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print(f"Syntax: {sys.argv[0]} <log file>")
        exit(0)

    process_fault_trace()
