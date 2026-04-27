import math
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm


"""
Computes the number of fault injection samples required (paper in https://ieeexplore.ieee.org/document/5090716)
(Eq. 1)

Parameters:
    e (float): Desired error margin (e.g., 0.05 for 5%)
    confidence_level (float): Confidence level (e.g., 0.95 for 95%)
    N (int): Total number of possible injection points (N)
    p (float): Expected probability of error (default=0.5 is the worst case. in reality, this is much smaller)

Returns the required number of samples
"""
def compute_required_samples(
    error_margin, 
    confidence_level, 
    num_points, 
    p=0.5):
    
    # Z-score from confidence level
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    # Initial sample size (infinite population)
    n = (z**2 * p * (1 - p)) / (error_margin**2)

    # Finite population correction
    if num_points is not None and num_points > 0:
        n = n / (1 + (n - 1) / num_points)

    return math.ceil(n)


def plot_data(df, mode="RTL"):
    # em    cl  n
    for em_value in df["em"].unique():
        subset = df[df["em"] == em_value]
        plt.plot(subset["cl"], subset["n"], label=f"{em_value}")

    plt.title(f"Required FI samples assuming error rate=0.50 - {mode}", fontsize=14)
    plt.xlabel("Confidence Level", fontsize=12)
    plt.ylabel("Sample Size (n)", fontsize=12)

    x_min = df["cl"].min()
    x_max = df["cl"].max()
    plt.grid(axis="y", linestyle="--", alpha=1)
    plt.xticks(np.arange(x_min, x_max + 0.01, 0.01))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Error Margin", fontsize=12)
    plt.show()


def main():
    # RTL simul:  IN_A, PSUM (Out_B), C1, C2, PROPAG, VALID
    bits_per_pe = 8 +   32 +          8 + 8 + 1 +     1

    # gate-level simul: the number of cells per PE = 823
    cells_per_mac_unit = 823

    dim=64
    npes = dim*dim
    cycles=2*64

    N_rtl = bits_per_pe*npes*cycles  # RTL injection
    N_gl = cells_per_mac_unit*npes*cycles # GL injection

    error_margins = np.linspace(0.01, 0.05, 5)
    confidence_levels = np.linspace(0.95, 0.99, 5)

    em_list = []
    cl_list = []
    n_rtl_list = []
    n_gl_list = []

    for em in error_margins:
        for cl in confidence_levels:
            samples_rtl = compute_required_samples(
                error_margin=em,
                confidence_level=cl,
                num_points=N_rtl
            )

            samples_gl = compute_required_samples(
                error_margin=em,
                confidence_level=cl,
                num_points=N_gl
            )

            em_list.append(em)
            cl_list.append(cl)

            n_rtl_list.append(samples_rtl)
            n_gl_list.append(samples_gl)

    df_rtl = pd.DataFrame({
        "em": em_list,
        "cl": cl_list,
        "n": n_rtl_list
    })

    df_gl = pd.DataFrame({
        "em": em_list,
        "cl": cl_list,
        "n": n_gl_list
    })

    print("RTL sample size")
    print(df_rtl)
    plot_data(df_rtl, mode="RTL")

    print("\nGL sample size")
    print(df_gl)
    plot_data(df_gl, mode="Gate level")

if __name__ == '__main__':
    main()