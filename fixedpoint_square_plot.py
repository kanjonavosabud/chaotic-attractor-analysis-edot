import numpy as np
import matplotlib.pyplot as plt
from helper.lorenz import my_lorenz
from reservoir_train_predict import generate_uv_for_rectangular_yz
import os

# ##################################################################################
# original function returning initial points in x=0, y=0 and fixed point plane
# plotting only on the y z axis with a diamond for the fixed point plane
# ##################################################################################
def get_initial_points(sigma=10, rho=22, beta=8/3, var1min=-20, var1max=20, var1count=10, var2min=-10, var2max=70, var2count=10):

    u_vals = np.linspace(var1min,var1max,var1count)
    v_vals = np.linspace(var2min,var2max,var2count)
   
    # Initial conditions on the x=0 plane

    initial_conditions_x = [[0, y, z] for y in u_vals for z in v_vals]

    # Initial Conditions on the y=0 plane
    initial_conditions_y = [[x, 0, z] for x in u_vals for z in v_vals]

    # Initial Conditions on the fixed points
    fixed_points = [(np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
                    (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)]

    initial_conditions_f = [[u_vals[i]*fixed_points[0][0] + v_vals[j]*fixed_points[1][0],
            u_vals[i]*fixed_points[0][1] + v_vals[j]*fixed_points[1][1],
            u_vals[i]*fixed_points[0][2] + v_vals[j]*fixed_points[1][2]] for i in range(len(u_vals)) for j in range(len(v_vals))]
        
    return initial_conditions_x, initial_conditions_y, initial_conditions_f

def fixed_point_analysis(rho=22, t_end=50, plane=1, var1min=-20, var1max=20, var1count=10, var2min=-10, var2max=70, var2count=10):

    sigma, beta  = 10, 8/3

    initial_conditions = get_initial_points(sigma, rho, beta, 
                                            var1min, var1max, var1count, 
                                            var2min, var2max, var2count)[plane-1]

    # Initial Conditions on the fixed points
    fixed_points = [(np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
                    (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)]

    # Solve for all initial conditions and classify trajectories
    trajectories = []
    classifications = []
    threshold = 1  # Distance threshold for classification

    for init in initial_conditions:
        # Solve the Lorenz system for the initial condition
        sol = my_lorenz(sigma=sigma, beta=beta, rho=rho, x0=init, t_end=t_end)
        final_state = sol[-1]

        # Classify the final state based on proximity to fixed points
        if np.linalg.norm(final_state - np.array(fixed_points[0])) < threshold:
            classification = "Fixed Point #1"
        elif np.linalg.norm(final_state - np.array(fixed_points[1])) < threshold:
            classification = "Fixed Point #2"
        else:
            classification = "Butterfly Attractor"

        trajectories.append(sol)
        classifications.append(classification)

    # Plot the initial conditions with classifications
    for init, classification in zip(initial_conditions, classifications):
        # Assign colors based on classification
        if classification == "Fixed Point #1": color = 'r'
        elif classification == "Fixed Point #2": color = 'g'
        else:
            print(init)
            color = 'black'

        plt.plot(init[1], init[2], marker='o', markersize=5, color=color)

    # Set plot labels and title
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('Initial Conditions')
    plt.grid(True)
    plt.show()

    # Print legend and range information
    print("LEGEND \n RED -- Fixed Point #1 \n GREEN -- Fixed Point #2 \n BLUE -- Butterfly")
    print()
    print(f"VAR1 MIN: {var1min}, VAR1 MAX: {var1max}, VAR1 COUNT: {var1count}")
    print(f"VAR2 MIN: {var2min}, VAR2 MAX: {var2max}, VAR2 COUNT: {var2count}")

# ##################################################################################
# secondary initial point generating function only for fixed point plane
# used to get a filled in basins of attraction graph in y z plane and u v plane
# ##################################################################################
def get_initial_points_uv(sigma=10, rho=22, beta=8/3, umin=-1, umax=1, ucount=20, vmin=-1, vmax=1, vcount=20):
    u_vals = np.linspace(umin, umax, ucount)
    v_vals = np.linspace(vmin, vmax, vcount)
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]
    initial_conditions = []
    uv_coords = []
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            x = u * fixed_points[0][0] + v * fixed_points[1][0]
            y = u * fixed_points[0][1] + v * fixed_points[1][1]
            z = u * fixed_points[0][2] + v * fixed_points[1][2]
            initial_conditions.append([x, y, z])
            uv_coords.append([u, v])
    return np.array(initial_conditions), np.array(uv_coords)

def plot_diamond_only(rho=22, t_end=50, umin=-1, umax=1, ucount=20, vmin=-1, vmax=1, vcount=20, diamond_thresh=1, show=True):
    sigma, beta = 10, 8/3
    initial_conditions, uv_coords = get_initial_points_uv(sigma, rho, beta, umin, umax, ucount, vmin, vmax, vcount)
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]
    classifications = []
    threshold = 1
    for idx, init in enumerate(initial_conditions):
        u, v = uv_coords[idx]
        # Only plot if inside the diamond
        if abs(u) + abs(v) <= diamond_thresh:
            sol = my_lorenz(sigma=sigma, beta=beta, rho=rho, x0=init, t_end=t_end)
            final_state = sol[-1]
            if np.linalg.norm(final_state - np.array(fixed_points[0])) < threshold:
                color = 'r'
            elif np.linalg.norm(final_state - np.array(fixed_points[1])) < threshold:
                color = 'g'
            else:
                color = 'black'
            plt.plot(init[1], init[2], marker='o', markersize=5, color=color)
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.suptitle('Basins of Attraction')
    plt.title(f"params: {umin}, {umax}, {ucount}")
    plt.grid(True)
    if show:
        plt.show()
    print("LEGEND \n RED -- Fixed Point #1 \n BLUE -- Fixed Point #2 \n BLACK -- Butterfly")
    print(f"Diamond threshold: |u| + |v| <= {diamond_thresh}")

def plot_uv_axes(rho=22, t_end=50, umin=-1, umax=1, ucount=20, vmin=-1, vmax=1, vcount=20):
    sigma, beta = 10, 8/3
    initial_conditions, uv_coords = get_initial_points_uv(sigma, rho, beta, umin, umax, ucount, vmin, vmax, vcount)
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]
    threshold = 1
    for idx, init in enumerate(initial_conditions):
        u, v = uv_coords[idx]
        sol = my_lorenz(sigma=sigma, beta=beta, rho=rho, x0=init, t_end=t_end)
        final_state = sol[-1]
        if np.linalg.norm(final_state - np.array(fixed_points[0])) < threshold:
            color = 'r'
        elif np.linalg.norm(final_state - np.array(fixed_points[1])) < threshold:
            color = 'g'
        else:
            color = 'black'
        plt.plot(u, v, marker='o', markersize=5, color=color)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('Initial Conditions (u-v axes)')
    plt.grid(True)
    plt.show()
    print("LEGEND \n RED -- Fixed Point #1 \n BLUE -- Fixed Point #2 \n BLACK -- Butterfly")
    print(f"u: {umin} to {umax}, v: {vmin} to {vmax}")

def save_classification_data(rho, t_end, umin, umax, ucount, vmin, vmax, vcount, diamond_thresh, 
                             outpath, filename):
    sigma, beta = 10, 8/3
    initial_conditions, uv_coords = get_initial_points_uv(sigma, rho, beta, umin, umax, ucount, vmin, vmax, vcount)
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]
    threshold = 1
    data = []
    for idx, init in enumerate(initial_conditions):
        u, v = uv_coords[idx]
        if abs(u) + abs(v) <= diamond_thresh:
            sol = my_lorenz(sigma=sigma, beta=beta, rho=rho, x0=init, t_end=t_end)
            final_state = sol[-1]
            if np.linalg.norm(final_state - np.array(fixed_points[0])) < threshold:
                label = 1
            elif np.linalg.norm(final_state - np.array(fixed_points[1])) < threshold:
                label = 2
            else:
                label = 0
            # Save u, v, x, y, z, label
            data.append([u, v, init[0], init[1], init[2], label])
    
    outpath += "/"+filename+".npy"
    np.save(outpath, np.array(data))

def plot_from_classification_data(npy_path, plane='uv', show=True, savepath=None, 
                                  title="Basins of Attraction Plot"):
    data = np.load(npy_path)
    # data: [u, v, x, y, z, label]
    color_map = {0: 'xkcd:sky blue', 1: 'r', 2: 'yellow'}
    if plane == 'uv':
        x_idx, y_idx = 0, 1  # u, v
        xlabel, ylabel = 'u', 'v'
    elif plane == 'yz':
        x_idx, y_idx = 3, 4  # y, z
        xlabel, ylabel = 'Y', 'Z'
    elif plane == 'xz':
        x_idx, y_idx = 2, 4  # x, z
        xlabel, ylabel = 'X', 'Z'
    elif plane == 'xy':
        x_idx, y_idx = 2, 3
        xlabel, ylabel = 'X', "Y"
    else:
        raise ValueError("plane must be one of 'uv', 'yz', or 'xz'")
    for row in data:
        color = color_map[row[5]]
        plt.plot(row[x_idx], row[y_idx], marker='o', markersize=5, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle('Basins of Attraction')
    plt.title(title)
    plt.grid(True)
    # Set Z axis limits if plotting a Z axis
    if plane in ['yz', 'xz']:
        plt.ylim(-60, 75)
    if savepath:
        filename = title+".png"
        filepath = os.path.join(savepath, filename)
        plt.savefig(filepath)
    if show:
        plt.show()
    plt.close()

def initial_points_plot(rho, t_end, initial_conditions, show=True, title="Basins of Attraction Plot"):

    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]

    threshold = 1
    for idx, init in enumerate(initial_conditions):
        sol = my_lorenz(sigma=sigma, beta=beta, rho=rho, x0=init, t_end=t_end)
        final_state = sol[-1]
        if np.linalg.norm(final_state - np.array(fixed_points[0])) < threshold:
            color = 'r'
        elif np.linalg.norm(final_state - np.array(fixed_points[1])) < threshold:
            color = 'yellow'
        else:
            color = 'xkcd:sky blue'
        plt.plot(init[1], init[2], marker='o', markersize=5, color=color)

    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.suptitle('Basins of Attraction')
    plt.title(f"Time: {t_end}")
    plt.grid(True)
    plt.show()
    print("LEGEND \n RED -- Fixed Point #1 \n BLUE -- Fixed Point #2 \n BLACK -- Butterfly")

if __name__ == "__main__":

    # Parameters
    rho = 24
    sigma = 10
    beta = 8/3
    predict_time = 70
    train_time = 50

    # Define rectangular region in Y,Z
    y_bounds = (-60, 60)
    z_bounds = (-60, 60)
    train_bounds = 5
    test_bounds = 7
    train_count = 100
    test_count = 170

    # Generate training points (smaller region)
    train_points, train_uv = generate_uv_for_rectangular_yz(
        rho=rho,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        umin=-train_bounds, umax=train_bounds, ucount=train_count,
        vmin=-train_bounds, vmax=train_bounds, vcount=train_count
    )

    test_points, test_uv = generate_uv_for_rectangular_yz(
        rho=rho,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        umin=-test_bounds, umax=test_bounds, ucount=test_count,
        vmin=-test_bounds, vmax=test_bounds, vcount=test_count
    )

    print(f"Generated {len(train_points)} training points in Y ∈ {y_bounds}, Z ∈ {z_bounds}")
    print(f"Generated {len(test_points)} testing points in Y ∈ {y_bounds}, Z ∈ {z_bounds}")

    initial_points_plot(rho, train_time, train_points, title="Basins of Attraction")
    # initial_points_plot(rho, predict_time, test_points, title="Basins of Attraction")
    
    rho = 24  # Given rho value
    sigma = 10
    beta = 8/3

    t_end = 70
    t_span = (0, t_end)  # Time span for evolution
    t_eval = np.linspace(*t_span, t_end*100)  # Time points
    
    # varmin, varmax, varcount = -2, 2, 100
    
    # # var1min, var1max, var1count, var2min, var2max, var2count = -1,1,20,-1,1,20
    # var1min, var1max, var1count, var2min, var2max, var2count = varmin,varmax,varcount,varmin,varmax,varcount
    
    # diamond_thresh = varmax
    # plot_diamond_only(rho = rho, t_end=t_end,
    #          umin=var1min, umax=var1max, ucount=var1count,
    #          vmin=var2min, vmax=var2max, vcount=var2count, diamond_thresh=diamond_thresh)

    # plot_uv_axes(rho = rho, t_end=t_end,
    #          umin=var1min, umax=var1max, ucount=var1count,
    #          vmin=var2min, vmax=var2max, vcount=var2count)

    # condition_list = [(-1,1,20), (-1,1,100), (-2,2,40), (-2,2,120), (-3,3,80), (-3,3,160)]
    # condition_list = [(-4,4,160), (-5,5,200), (-7,7,250)]
    # output_dir = "./graphs/basins_of_attraction"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # for varmin, varmax, varcount in condition_list:
    #     # ... [set up params] ...
    #     var1min, var1max, var1count = varmin, varmax, varcount
    #     var2min, var2max, var2count = varmin, varmax, varcount
    #     diamond_thresh = varmax

    #     plot_diamond_only(
    #         rho=rho, t_end=t_end,
    #         umin=var1min, umax=var1max, ucount=var1count,
    #         vmin=var2min, vmax=var2max, vcount=var2count,
    #         diamond_thresh=diamond_thresh,
    #         show=False  # <--- Don't show the plot
    #     )

        
    #     filename = f"{varmin}_{varmax}_{varcount}.png"
    #     filepath = os.path.join(output_dir, filename)
    #     plt.savefig(filepath)
    #     plt.close()

    # condition_list = [(-1,1,20), (-1,1,100), (-2,2,40), (-2,2,120), (-3,3,80), (-3,3,160)]
    # condition_list = [(-4,4,160), (-5,5,200), (-7,7,250)]

    # condition_list = [(-7,7,250)]
    # data_path = "./data"
    # graph_output_path = "./graphs/basins_of_attraction"

    # for varmin, varmax, varcount in condition_list:
        
    #     print(f"Running {varmin} {varmax} {varcount}")

    #     var1min, var1max, var1count = varmin, varmax, varcount
    #     var2min, var2max, var2count = varmin, varmax, varcount
    #     diamond_thresh = varmax

    #     tag_name = f"{var1min}_{var1max}_{var1count}"

    #     save_classification_data(
    #         rho=rho, t_end=t_end,
    #         umin=var1min, umax=var1max, ucount=var1count,
    #         vmin=var2min, vmax=var2max, vcount=var2count,
    #         diamond_thresh=diamond_thresh,
    #         outpath=data_path,
    #         filename=tag_name
    #     )

    #     plot_from_classification_data(npy_path="./data/"+tag_name+".npy", plane='yz', 
    #                                   show=False, savepath=graph_output_path, title=tag_name+' yz')

    # plot_from_classification_data(npy_path="./data/-7_7_250.npy", plane='yz', 
    #                              show=False, savepath='./graphs/basins_of_attraction/', title='-7_7_250 restricted z limit')