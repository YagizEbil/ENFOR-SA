import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

"""
python scripts/data_proc/eval_criticality.py reports/ResNet18/exp-batches-rtl/sequential/trace/sim-xyz-s12-OSDIM8.csv
"""


def compute_fault_crit(df):
    grouped = df.groupby('fault_tag').agg(
        injections=('fault_tag', 'count'),
        sdc1_sum=('sdc1', 'sum'),
        #sdc5_sum=('sdc5', 'sum')
    )

    grouped['sdc1_ratio'] = grouped['sdc1_sum'] / grouped['injections']
    #grouped['sdc5_ratio'] = grouped['sdc5_sum'] / grouped['injections']

    # the total number of critical faults 
    sdc1_total_tag = grouped['sdc1_sum'].to_dict()

    # computes the sdc avf only for the faults that are critical
    sdc1_avf_tag = grouped['sdc1_ratio'].to_dict()    

    injections_tag = grouped['injections'].to_dict()

    return sdc1_avf_tag

def main():
    if len(sys.argv) < 2:
        print(f"Syntax: {sys.argv[0]} <input file>")
        exit(0)

    fn1 = sys.argv[1]
    df1 = pd.read_csv(fn1, comment='#', sep='\t')
    
    sdc1_avf_tag = compute_fault_crit(df1)

    dc2_avf_tag = None

    if len(sys.argv) > 2:
        fn2 = sys.argv[2]
        df2 = pd.read_csv(fn2, comment='#', sep='\t')
        dc2_avf_tag = compute_fault_crit(df2)

    plot_histogram(sdc1_avf_tag, dc2_avf_tag, fn1)



def plot_histogram(data1, data2, fn):
    # Sample data
    #data1 = np.random.normal(loc=0, scale=1, size=1000)
    #data2 = np.random.normal(loc=2, scale=1.5, size=1000)

    values_1 = list(data1.values())

    values_2=None

    if data2 is not None:
        values_2 = list(data2.values())
        values_2 = [100*v for v in values_2]

    values_1 = [100*v for v in values_1]

    nbins_1 = int(get_nbins_method_3(values_1))

    plt.clf()

    # Plot
    plt.hist(values_1, bins=nbins_1, alpha=0.5, color='blue')
    
    if values_2 is not None:
        plt.hist(values_2, bins=nbins_1, alpha=0.5, color='red')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12, label='Sequential'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Parallel')
        ]

        plt.legend(handles=legend_elements)

    plt.title(f'Dataset: {fn}')
    plt.xlabel('Criticality (%)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()


# Constant bin size
def get_nbins_method_1(data):
    return 100

# Square Root Choice
def get_nbins_method_2(data):
    return int(np.sqrt(len(data)))

# Rice Rule
def get_nbins_method_3(data):
    return int(2*len(data)**(1/3))

# Sturges' Formula
def get_nbins_method_4(data):
    return int(np.ceil(np.log2(len(data)) + 1))

# Scott's Rule
def get_nbins_method_5(data):
    std_dev = np.std(data)
    bin_width = 3.5 * std_dev / len(data)**(1/3)
    num_bins = int(np.ceil((max(data) - min(data)) / bin_width))

# Freedman-Diaconis' Rule
def get_nbins_method_6(data):
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr / len(data)**(1/3)
    num_bins = int(np.ceil((max(data) - min(data)) / bin_width))
    return num_bins

if __name__ == "__main__":
    main()