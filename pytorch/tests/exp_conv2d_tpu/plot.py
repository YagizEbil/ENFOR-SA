import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


color_single = "#b2df8a"
color_line = "#a6cee3"
color_box = "#ffd5a8"
color_masked = "#b3b3a4"

color_neutrons = "#88b2cd"
color_protons = "#f1948a"

color_rad_line = "#aec5d1" # Short-Line
color_rtl_line = "#c8b7d1" # Long-Line

def plot_stacked_bars(
    list1, 
    list2, 
    list3, 
    list4, 
    ylabel="HVF",
    labels=None, 
    category_names=["Single", "Line", "Box", "Masked"]):

    """
    Plots a 100% stacked bar chart from four lists.

    Parameters:
        list1, list2, list3, list4: Lists of percentages (same length, each column sums to 100)
        labels: Labels for x-axis (optional)
        category_names: Names for the 4 categories (optional)
    """

    data = np.array([list1, list2, list3, list4])
    n_points = len(list1)

    # X positions
    x = np.arange(n_points)

    # Default labels
    if labels is None:
        labels = [f"Point {i}" for i in range(n_points)]
    if category_names is None:
        category_names = ["Cat 1", "Cat 2", "Cat 3", "Cat 4"]

    # Plot
    fig, ax = plt.subplots()

    bottom = np.zeros(n_points)

    colors = [color_single, color_line, color_box, color_masked]

    for i in range(4):
        ax.bar(x, data[i], bottom=bottom, color=colors[i], label=category_names[i])
        bottom += data[i]

    tick_size=37

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    #plt.grid(True)
    ax.grid(axis="y", linestyle="--", alpha=1)

    ax.set_ylabel("HVF", fontsize=tick_size)
    #ax.set_title(ylabel, fontsize=tick_size)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}%'))

    legend = ax.legend(fontsize=tick_size)

    bbox = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    x, y, w, h = bbox.bounds
    legend.set_bbox_to_anchor((x, y - 0.05, w, h), transform=ax.transAxes)

    plt.xticks(fontsize=30)#, rotation=25)
    plt.yticks(fontsize=tick_size)

    plt.show()


def plot_grouped_bars(
    df, 
    col_groups="Benchmark", # what defines a bar group (e.g., the benchmarks)
    col_types="Type",       # defines each bar type inside a group (e.g., Single, Line, Box)
    col_values="AVF"           # the column/values to plot (e.g., AVF)
    ):

    # Define the desired order
    #type_order = ["Single", "Line (avg)", "Box"]
    #type_order = ["Single", "Rad-Line", "RTL-Line", "Box"]
    type_order = ["Single", "Short-Line", "Long-Line", "Box"]

    groups = df[col_groups].unique()
    x = np.arange(len(groups))  # group positions
    #width = 0.25  # bar width
    width = 0.2

    fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=(7, 5))

    colors = [color_single, color_rad_line, color_rtl_line, color_box]

    for i, t in enumerate(type_order):
        values = []
        for b in groups:
            val = df[(df[col_groups] == b) & (df[col_types] == t)][col_values]
            values.append(val.values[0] if len(val) > 0 else 0)
        
        ax.bar(x + i * width, values, width, label=t, color=colors[i])

    tick_size=34

    #plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}%'))
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}%'))

    # Labels and formatting
    #ax.set_xlabel("Benchmark")
    ax.set_ylabel("AVF", fontsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)

    #ax.set_title("Grouped Bar Plot by Benchmark and Type")
    ax.set_xticks(x + width * (len(type_order) - 1) / 2)
    ax.set_xticklabels(groups,  fontsize=30)
    ax.legend(fontsize=32)
    
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
