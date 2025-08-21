import numpy as np
import matplotlib.pyplot as plt
import os

def basin_map_iterate(x0, y0, lam_x, lam_y, n_iter=100, y_thresh=1e6):
    x, y = x0, y0
    for _ in range(n_iter):
        x = (lam_x * x) % 1
        y = lam_y * y + np.cos(2 * np.pi * x)
        if y > y_thresh:
            return 1  # +infty attractor
        if y < -y_thresh:
            return -1  # -infty attractor
    # If not diverged, classify by sign
    return 1 if y > 0 else -1

def save_basin_map_classification(lam_x, lam_y, x_min, x_max, x_count, y_min, y_max, y_count, n_iter, y_thresh, outpath, filename):
    xs = np.linspace(x_min, x_max, x_count)
    ys = np.linspace(y_min, y_max, y_count)
    data = []
    for i, x0 in enumerate(xs):
        for j, y0 in enumerate(ys):
            label = basin_map_iterate(x0, y0, lam_x, lam_y, n_iter=n_iter, y_thresh=y_thresh)
            data.append([x0, y0, label])
    arr = np.array(data)
    np.save(os.path.join(outpath, filename + '.npy'), arr)

def plot_basin_map_from_data(npy_path, show=True, savepath=None, title='Basin Map'):
    data = np.load(npy_path)
    for row in data:
        color = 'black' if row[2] == 1 else 'white'
        plt.plot(row[0], row[1], marker='o', markersize=1, color=color)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.gca().set_facecolor('gray')
    plt.grid(False)
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()

if __name__ == '__main__':
    lam_x = 3
    lam_y = 1.5
    x_min, x_max, x_count = 1/3, 2/3, 500
    y_min, y_max, y_count = -2, 2, 500
    n_iter = 100
    y_thresh = 1e6
    outpath = './data'
    filename = f'basinmap_lx{lam_x}_ly{lam_y}_{x_count}x{y_count} second period'
    os.makedirs(outpath, exist_ok=True)
    save_basin_map_classification(lam_x, lam_y, x_min, x_max, x_count, y_min, y_max, y_count, n_iter, y_thresh, outpath, filename)
    plot_basin_map_from_data(os.path.join(outpath, filename + '.npy'), show=False, savepath=f'./graphs/basins_of_attraction/{filename}.png', title=filename) 