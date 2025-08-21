import matplotlib.pyplot as plt
import numpy as np

def plot_lorenz_attractor(data, verbose=True):
    """Plot Lorenz attractor in 3D and optionally show time series"""
    fig = plt.figure(figsize=(10, 8))
    x, y, z = data.T
    
    # 3D plot
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    
    if verbose:
        # Time series plots
        fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(10, 12))
        ax_x.plot(x, lw=0.5)
        ax_x.set_ylabel("x")
        ax_y.plot(y, lw=0.5)
        ax_y.set_ylabel("y")
        ax_z.plot(z, lw=0.5)
        ax_z.set_ylabel("z")
        ax_z.set_xlabel("Time")
    
    plt.show()

def plot_prediction_comparison(actual, predicted, title="Prediction vs Actual"):
    """Plot comparison between actual and predicted values"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    components = ['X', 'Y', 'Z']
    
    for i, (ax, comp) in enumerate(zip(axes, components)):
        ax.plot(actual[:, i], label='Actual', alpha=0.7)
        ax.plot(predicted[:, i], label='Predicted', alpha=0.7)
        ax.set_ylabel(comp)
        ax.legend()
    
    axes[-1].set_xlabel("Time")
    plt.suptitle(title)
    plt.tight_layout()
    # plt.show()
    
    return fig, axes  # Return the figure and axes objects 