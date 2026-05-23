import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


"""
python scripts/data_proc/eval_pe.py reports/ResNet18/exp-batches-rtl/sequential/trace/sim-xyz-s12-OSDIM8.csv
"""


SIGNAL = {
    "IN_A":  0,
    "IN_B":  1,
    "OUT_C": 4,
    "SIG_PROPAG": 5,
    "SIG_VALID":  6,
}

IN_A, IN_B, OUT_C = SIGNAL["IN_A"], SIGNAL["IN_B"], SIGNAL["OUT_C"]
PROPAG, VALID = SIGNAL["SIG_PROPAG"], SIGNAL["SIG_VALID"]


def main():
    if len(sys.argv) < 2:
        print(f"Syntax: {sys.argv[0]} <input file>")
        exit(0)

    fn = sys.argv[1]

    df = pd.read_csv(fn, comment='#', sep='\t')

    DIM=8

    avf = [[0 for _ in range(DIM)] for _ in range(DIM)]
    inj = [[0 for _ in range(DIM)] for _ in range(DIM)]
    sdc = [[0 for _ in range(DIM)] for _ in range(DIM)]
    
    gemm_msk = [[0 for _ in range(DIM)] for _ in range(DIM)]
    scale_msk = [[0 for _ in range(DIM)] for _ in range(DIM)]
    round_msk = [[0 for _ in range(DIM)] for _ in range(DIM)]
    clamp_msk = [[0 for _ in range(DIM)] for _ in range(DIM)]
    round_clamp_msk = [[0 for _ in range(DIM)] for _ in range(DIM)]

    for i in range(DIM):
        for j in range(DIM):
            pe_df = df[(df["pe_row"] == i) & (df["pe_col"] == j)]

            pe_df = pe_df[(pe_df['target'] == IN_A)]

            inj[i][j] = len(pe_df)

            if inj[i][j] == 0:
                pass
                
            sdc[i][j] = pe_df["sdc1"].sum()

            gemm_msk_pe = pe_df["gemm_msk"].sum()
            scale_msk_pe = pe_df["scale_msk"].sum()
            round_msk_pe = pe_df["round_msk"].sum()
            clamp_msk_pe = pe_df["clamp_msk"].sum()

            avf[i][j] = sdc[i][j]/inj[i][j] if inj[i][j] else 0
            gemm_msk[i][j] = 100*gemm_msk_pe/inj[i][j] if inj[i][j] else -1
            scale_msk[i][j] = 100*scale_msk_pe/inj[i][j] if inj[i][j] else -1
            round_msk[i][j] = 100*round_msk_pe/inj[i][j] if inj[i][j] else -1
            clamp_msk[i][j] = 100*clamp_msk_pe/inj[i][j] if inj[i][j] else -1
            round_clamp_msk[i][j] = 100*(pe_df["clamp_msk"].sum() + pe_df["round_msk"].sum())/inj[i][j]  if inj[i][j] else -1

    plot_heatmap(avf, bar_label="AVF")
    #plot_heatmap(inj, bar_label="Injections")


def plot_heatmap(data, title='title', bar_label="Mean AVF"):
    #data = np.random.rand(5, 5)
    # data has to be 2d data!

    ax = sns.heatmap(data, annot=True, annot_kws={"size": 14, "weight": "bold"}, cmap='coolwarm')#, vmin=0, vmax=4)
   
    for text in ax.texts:
        text.set_fontsize(14) 

    ax.set_xticklabels(range(1, 9))
    ax.set_yticklabels(range(1, 9), rotation=0)
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20) # color bar number sizes

    cbar.set_label(bar_label, rotation=90, fontsize=24)
    #cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}%'))
    
    plt.show()


if __name__ == "__main__":
    main()
