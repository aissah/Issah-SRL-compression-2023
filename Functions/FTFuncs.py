"""
Created on Fri Aug 12 08:51:28 2022

@author: issah
"""
import matplotlib.pyplot as plt


def plotsaveimshow(
    data,
    xextent,
    yextent,
    mini,
    maxi,
    xlabel,
    ylabel,
    title,
    color_scheme,
    label_size,
    filename,
    dateaxis=None,
):
    """
    Plotting function: very specialized and definitely not optimal way of doing
    this.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Data.
    xextent : list/tuple
        2-value list indicating the values at the beginning and end of x-axis.
    yextent : list/tuple
        2-value list indicating the values at the beginning and end of y-axis.
    mini : float
        minimum value to clip colorbar to.
    maxi : int/float
        maximum value to clip colorbar to.
    xlabel : str
        x-axis label.
    ylabel : str
        label of y-axis.
    title : str
        titile of plot.
    color_scheme : cmap options
        color scheme to use.
    label_size : TYPE
        size of plot lavels.
    filename : str
        name of saved image of plot.
    dateaxis : int, optional
        Indicate if either axis is a datetime axis. The default is None.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(11, 7))
    if mini is None:
        plt.imshow(
            data,
            cmap=color_scheme,
            aspect="auto",
            extent=(xextent[0], xextent[1], yextent[0], yextent[1]),
        )
    else:
        plt.imshow(
            data,
            vmin=mini,
            vmax=maxi,
            cmap=color_scheme,
            aspect="auto",
            extent=(xextent[0], xextent[1], yextent[0], yextent[1]),
        )
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.title(title, fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.xticks(fontsize=label_size)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=label_size)
    if dateaxis == "x":
        ax = plt.gca()
        ax.xaxis_date()
    elif dateaxis == "y":
        ax = plt.gca()
        ax.yaxis_date()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

