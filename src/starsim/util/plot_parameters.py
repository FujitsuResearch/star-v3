import matplotlib.pyplot as plt
import numpy as np


def set_parameter_for_plot(N_plot: int = 8):
    """Apply the plotting defaults used in the paper notebooks."""

    # Figure settings
    plt.rcParams["figure.dpi"] = 400  # Image resolution
    plt.rcParams["figure.figsize"] = [6.4, 4.0]  # [width (inch), height (inch)]
    plt.rcParams["savefig.dpi"] = 500  # Resolution for saved figures

    # Tick and axis settings
    plt.rcParams["xtick.direction"] = "in"  # Tick direction: in, out, or inout
    plt.rcParams["ytick.direction"] = "in"  # Tick direction: in, out, or inout
    plt.rcParams["xtick.major.width"] = 1.7  # Major tick width on x-axis
    plt.rcParams["ytick.major.width"] = 1.7  # Major tick width on y-axis
    plt.rcParams["xtick.major.size"] = 4  # Major tick size on x-axis
    plt.rcParams["ytick.major.size"] = 4  # Major tick size on y-axis
    plt.rcParams["xtick.minor.width"] = 1.6  # Minor tick width on x-axis
    plt.rcParams["ytick.minor.width"] = 1.6  # Minor tick width on y-axis
    plt.rcParams["xtick.minor.size"] = 0  # Minor tick size on x-axis
    plt.rcParams["ytick.minor.size"] = 3  # Minor tick size on y-axis
    plt.rcParams["axes.linewidth"] = 1.5  # Axes frame line width

    # Default color cycle
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.get_cmap("cool")(np.linspace(0, 1, N_plot))
    )

    # Layout adjustment
    plt.rcParams["axes.titley"] = 1.03  # Title y-position: 1.0 is the top edge

    # Font-related settings
    # plt.rcParams["font.family"] = "sans-serif"  # Font family
    plt.rcParams["font.size"] = 14  # Base font size
    plt.rcParams["xtick.labelsize"] = 14  # Font size of x-axis ticks
    plt.rcParams["ytick.labelsize"] = 14  # Font size of y-axis ticks
    plt.rcParams["figure.titlesize"] = 11  # Subtitle font size
    plt.rcParams["axes.titlesize"] = 13  # Axis title font size
    plt.rcParams["axes.labelsize"] = 16  # Axis label font size
    plt.rcParams["legend.fontsize"] = 11  # Legend font size

    # Legend settings
    plt.rcParams["legend.loc"] = "best"  # Automatic legend placement
    plt.rcParams["legend.frameon"] = True  # Draw a legend frame
    plt.rcParams["legend.framealpha"] = 0.8  # Frame opacity
    plt.rcParams["legend.facecolor"] = "white"  # Background color
    plt.rcParams["legend.edgecolor"] = "lightgray"  # Frame color
    plt.rcParams["legend.fancybox"] = True  # Rounded legend box
