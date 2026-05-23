import sys
import pandas as pd
from tabulate import tabulate

if len(sys.argv) < 2:
    print(f"Syntax: {sys.argv[0]} <input file>")
    exit(0)

fn = sys.argv[1]

# Reads the trace file
df_full = pd.read_csv(fn, comment='#', sep='\t')

# The dataframe of the critical faults only
df_critical = df_full[(df_full['sdc1'] == 1)]

# The total number of injections
injections = len(df_full)

# The total number of critical runs
sdc1 = len(df_critical)

# AVF
avf = sdc1/injections

# The list of (unique) critical faults
critical_list = df_critical["fault_tag"].unique().tolist()


# RTL/GL injections only
"""
# The number of injections exposed to the the output layer, in the SW level
exposed_count = (
    (df_full['gemm_msk']  == 0) &
    (df_full['scale_msk'] == 0) &
    (df_full['round_msk'] == 0) &
    (df_full['clamp_msk'] == 0)
).sum()


# The fraction of injections masked during the Gemmini matmul
gemm_msk_count  = (df_full['gemm_msk']  == 1).sum()
scale_msk_count = (df_full['scale_msk'] == 1).sum()
round_msk_count = (df_full['round_msk'] == 1).sum()
clamp_msk_count = (df_full['clamp_msk'] == 1).sum()

total_msk = gemm_msk_count + scale_msk_count + round_msk_count + clamp_msk_count
exposed_count = injections - total_msk
exposed_frac = exposed_count/injections

tab_msk = [
    ["gemm_msk",  gemm_msk_count/injections],
    ["scale_msk", scale_msk_count/injections],
    ["round_msk", round_msk_count/injections],
    ["clamp_msk", clamp_msk_count/injections]
]

print(tabulate(tab_msk))

print(f"Exposed:  {100*exposed_frac:.2f}%")
"""

print(f"Mean AVF: {100*avf:.2f}%")
