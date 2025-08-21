import numpy as np
from helper.lorenz import my_lorenz
from helper.reservoir import run_reservoir
from helper.visualization import plot_prediction_comparison
import matplotlib.pyplot as plt
import os
from helper.lorenz import solve_lorenz

def generate_uv_for_rectangular_yz(rho, y_bounds, z_bounds, umin, umax, ucount, vmin, vmax, vcount):
    """
    Generate (u, v) points such that their mapped (Y, Z) fall within a rectangular region.
    Returns initial_points (in xyz) and uv_coords.
    """
    sigma, beta = 10, 8/3
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]
    u_vals = np.linspace(umin, umax, ucount)
    v_vals = np.linspace(vmin, vmax, vcount)
    initial_points = []
    uv_coords = []
    for u in u_vals:
        for v in v_vals:
            x = u * fixed_points[0][0] + v * fixed_points[1][0]
            y = u * fixed_points[0][1] + v * fixed_points[1][1]
            z = u * fixed_points[0][2] + v * fixed_points[1][2]
            if y_bounds[0] <= y <= y_bounds[1] and z_bounds[0] <= z <= z_bounds[1]:
                initial_points.append([x, y, z])
                uv_coords.append([u, v])
    return np.array(initial_points), np.array(uv_coords)

def generate_fixed_point_plane_points(rho=24, train_bounds=3, test_bounds=6, train_count=20, test_count=40):
    """
    Generate training and testing initial points on the fixed point plane (u,v plane).
    
    Parameters:
    -----------
    rho : float
        Lorenz parameter
    train_bounds : float
        Bounds for training points (u and v will range from -train_bounds to train_bounds)
    test_bounds : float
        Bounds for testing points (u and v will range from -test_bounds to test_bounds)
    train_count : int
        Number of points per dimension for training grid
    test_count : int
        Number of points per dimension for testing grid
    
    Returns:
    --------
    tuple
        (train_initial_points, test_initial_points, train_uv_coords, test_uv_coords)
    """
    sigma, beta = 10, 8/3
    
    # Calculate fixed points
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]
    
    # Generate training points
    train_u_vals = np.linspace(-train_bounds, train_bounds, train_count)
    train_v_vals = np.linspace(-train_bounds, train_bounds, train_count)
    train_initial_points = []
    train_uv_coords = []
    
    for u in train_u_vals:
        for v in train_v_vals:
            x = u * fixed_points[0][0] + v * fixed_points[1][0]
            y = u * fixed_points[0][1] + v * fixed_points[1][1]
            z = u * fixed_points[0][2] + v * fixed_points[1][2]
            train_initial_points.append([x, y, z])
            train_uv_coords.append([u, v])
    
    # Generate testing points
    test_u_vals = np.linspace(-test_bounds, test_bounds, test_count)
    test_v_vals = np.linspace(-test_bounds, test_bounds, test_count)
    test_initial_points = []
    test_uv_coords = []
    
    for u in test_u_vals:
        for v in test_v_vals:
            x = u * fixed_points[0][0] + v * fixed_points[1][0]
            y = u * fixed_points[0][1] + v * fixed_points[1][1]
            z = u * fixed_points[0][2] + v * fixed_points[1][2]
            test_initial_points.append([x, y, z])
            test_uv_coords.append([u, v])
    
    return np.array(train_initial_points), np.array(test_initial_points), np.array(train_uv_coords), np.array(test_uv_coords)

def train_and_predict_reservoir(
    train_initial_points, train_time, rho,
    predict_initial_point, sync_time, predict_time,
    t_eval=0.1, transient_time=20, reservoir_params=None
):
    """
    Trains a reservoir on Lorenz time series, then synchronizes and predicts on a new initial point.
    The sync segment is appended to the training data, so the reservoir is in sync at prediction start.
    Returns all model components, predictions, and ground truth.
    """
    sigma, beta = 10, 8/3
    # --- Training data from all initial points ---
    all_train_data = []
    for x0 in train_initial_points:
        data = my_lorenz(sigma=sigma, beta=beta, rho=rho, t_start=0, t_end=train_time, t_trans=transient_time, t_eval=t_eval, x0=x0)
        all_train_data.append(data)

    # --- Sync and prediction data for the prediction initial point ---
    total_pred_time = transient_time + sync_time + predict_time
    full_pred_data = my_lorenz(sigma=sigma, beta=beta, rho=rho, t_start=0, t_end=total_pred_time, t_trans=0, t_eval=t_eval, x0=predict_initial_point)
    sync_steps = int(sync_time / t_eval)
    pred_steps = int(predict_time / t_eval)
    sync_data = full_pred_data[:sync_steps]
    pred_data = full_pred_data[sync_steps:sync_steps+pred_steps]

    # --- Append sync segment to training data ---
    all_train_data.append(sync_data)
    train_data = np.concatenate(all_train_data, axis=0)
    test_data = train_data.copy()

    # --- Train reservoir ---
    result = run_reservoir(train_data, test_data, reservoir_params=reservoir_params, fit_mode='case3')
    adj_matrix = result['adj_matrix']
    reservoir = result['reservoir']
    reservoir_input = result['res_input']
    reservoir_output = result['res_output']
    C = result['C']
    output_dim = train_data.shape[1]

    # --- Predict autonomously from last state after training ---
    res_state = reservoir.copy()
    predictions = np.zeros_like(pred_data)
    current_state = res_state.copy()
    for t in range(pred_data.shape[0]):
        predictions[t] = np.dot(reservoir_output, current_state)
        x = np.dot(adj_matrix, current_state) + np.dot(reservoir_input, predictions[t]) + C
        current_state = np.tanh(x)

    # --- Plot comparison ---
    fig, axes = plot_prediction_comparison(pred_data, predictions)

    # Add horizontal lines for fixed points
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]
    for fp in fixed_points:
        for i, ax in enumerate(axes):
            ax.axhline(y=fp[i], color='gray', linestyle='--', linewidth=1)  # Add to Y-axis plot

    plt.show()  # Show the plot after adding lines

    # --- Return all model components and predictions ---
    return {
        'adj_matrix': adj_matrix,
        'reservoir': reservoir,
        'reservoir_input': reservoir_input,
        'reservoir_output': reservoir_output,
        'C': C,
        'predictions': predictions,
        'ground_truth': pred_data
    }

def reservoir_basins_of_attraction(
    train_initial_points, train_time, rho,
    test_initial_points, sync_time, predict_time,
    t_eval=0.1, transient_time=20, reservoir_params=None,
    classification_threshold=1.0, show=True, savepath=None, title="Reservoir Basins of Attraction"
):
    """
    Trains a reservoir on multiple initial points, then tests multiple prediction points
    and classifies their end behavior to create a basins of attraction plot.
    """
    sigma, beta = 10, 8/3
    # --- Training data from all initial points ---
    all_train_data = []
    for x0 in train_initial_points:
        data = my_lorenz(sigma=sigma, beta=beta, rho=rho, t_start=0, t_end=train_time, t_trans=transient_time, t_eval=t_eval, x0=x0)
        all_train_data.append(data)
    train_data = np.concatenate(all_train_data, axis=0)
    test_data = train_data.copy()
    # --- Train reservoir ---
    result = run_reservoir(train_data, test_data, reservoir_params=reservoir_params)
    adj_matrix = result['adj_matrix']
    reservoir = result['reservoir']
    reservoir_input = result['res_input']
    reservoir_output = result['res_output']
    C = result['C']
    # --- Calculate fixed points for classification ---
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]
    # --- Test each initial point and classify ---
    classifications = []
    predictions_list = []
    for test_point in test_initial_points:
        total_pred_time = transient_time + sync_time + predict_time
        full_pred_data = my_lorenz(sigma=sigma, beta=beta, rho=rho, t_start=0, t_end=total_pred_time, t_trans=0, t_eval=t_eval, x0=test_point)
        sync_steps = int(sync_time / t_eval)
        pred_steps = int(predict_time / t_eval)
        sync_data = full_pred_data[:sync_steps]
        # sync_data = np.zeros((sync_steps, 3))
        # Synchronize reservoir with this dummy sequence
        res_state = reservoir.copy()
        for t in range(sync_data.shape[0]):
            x = np.dot(adj_matrix, res_state) + np.dot(reservoir_input, sync_data[t]) + C
            res_state = np.tanh(x)
        # Predict autonomously
        predictions = np.zeros((pred_steps, 3))
        current_state = res_state.copy()
        for t in range(pred_steps):
            predictions[t] = np.dot(reservoir_output, current_state)
            x = np.dot(adj_matrix, current_state) + np.dot(reservoir_input, predictions[t]) + C
            current_state = np.tanh(x)
        # Classify final state
        final_state = predictions[-1]
        if np.linalg.norm(final_state - np.array(fixed_points[0])) < classification_threshold:
            classification = 1  # Fixed Point #1
        elif np.linalg.norm(final_state - np.array(fixed_points[1])) < classification_threshold:
            classification = 2  # Fixed Point #2
        else:
            classification = 0  # Butterfly Attractor
        classifications.append(classification)
        predictions_list.append(predictions)
    # --- Plot basins of attraction ---
    color_map = {0: 'xkcd:sky blue', 1: 'r', 2: 'yellow'}
    for i, test_point in enumerate(test_initial_points):
        color = color_map[classifications[i]]
        plt.plot(test_point[1], test_point[2], marker='o', markersize=5, color=color)
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.suptitle('Reservoir Basins of Attraction')
    plt.title(title)
    plt.grid(True)
    plt.ylim(-75, 75)
    if savepath:
        filename = title + ".png"
        filepath = os.path.join(savepath, filename)
        plt.savefig(filepath)
    if show:
        plt.show()
    plt.close()
    print("LEGEND \n RED -- Fixed Point #1 \n YELLOW -- Fixed Point #2 \n BLUE -- Butterfly Attractor")
    return {
        'adj_matrix': adj_matrix,
        'reservoir': reservoir,
        'reservoir_input': reservoir_input,
        'reservoir_output': reservoir_output,
        'C': C,
        'classifications': classifications,
        'predictions_list': predictions_list,
        'test_points': test_initial_points
    }

def initial_points_plot(rho, t_end, initial_conditions,beta=8/3, sigma=10, show=True, title="Basins of Attraction Plot"):

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

def run_reservoir_basin_analysis():
    # Parameters
    rho = 24
    train_time = 150
    predict_time = 70
    sync_time = 1
    transient_time = 0

    print("Generating training and testing points for rectangular Y,Z region...")
    
    # Define rectangular region in Y,Z
    y_bounds = (-60, 60)
    z_bounds = (-60, 60)
    train_bounds = 5
    test_bounds = 7
    train_count = 15
    test_count = 25

    # Generate training points (smaller region)
    train_points, train_uv = generate_uv_for_rectangular_yz(
        rho=rho,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        umin=-train_bounds, umax=train_bounds, ucount=train_count,
        vmin=-train_bounds, vmax=train_bounds, vcount=train_count
    )
    # Generate testing points (larger region)
    test_points, test_uv = generate_uv_for_rectangular_yz(
        rho=rho,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        umin=-test_bounds, umax=test_bounds, ucount=test_count,
        vmin=-test_bounds, vmax=test_bounds, vcount=test_count
    )

    # train_points = [[-5,-5,-20]]

    print(f"Generated {len(train_points)} training points and {len(test_points)} testing points in Y ∈ {y_bounds}, Z ∈ {z_bounds}")
    
    # initial_points_plot(rho, train_time, train_points, title="Basins of  for Training Points")
    # initial_points_plot(rho, predict_time, test_points, title=" Basins of Attraction for Testing Points")


    # Run basins of attraction analysis
    print("\nRunning reservoir basins of attraction analysis...")
    result = reservoir_basins_of_attraction(
        train_initial_points=train_points,
        train_time=train_time,
        rho=rho,
        test_initial_points=test_points,
        sync_time=sync_time,
        predict_time=predict_time,
        transient_time=transient_time,
        t_eval=0.01,
        title=f"Reservoir Basins: Rectangular YZ, Train[-{train_bounds},{train_bounds}], Test[-{test_bounds},{test_bounds}], ρ={rho}"
    )
    
    print("Analysis complete!")
    print(f"Model trained on {len(train_points)} points")
    print(f"Tested on {len(test_points)} points")
    print(f"Classification results: {np.bincount(result['classifications'])}")

def run_single_point_analysis():
    rho = 24
    beta = 8/3
    sigma = 10

    train_time = 50
    sync_time = 0
    predict_time = 70
    t_eval = 0.01
    transient_time = 0
    reservoir_params = None


    # Define rectangular region in Y,Z
    y_bounds = (-60, 60)
    z_bounds = (-60, 60)
    train_bounds = 20
    train_count = 15

    # Generate training points (smaller region)
    train_points, train_uv = generate_uv_for_rectangular_yz(
        rho=rho,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        umin=-train_bounds, umax=train_bounds, ucount=train_count,
        vmin=-train_bounds, vmax=train_bounds, vcount=train_count
    )
    perturb_factor = 10
    fixed_points_with_perturbation = [
        (np.sqrt(beta * (rho - 1)) + perturb_factor, np.sqrt(beta * (rho - 1)) + perturb_factor, rho - 1 + perturb_factor),
        (-np.sqrt(beta * (rho - 1)) + perturb_factor, -np.sqrt(beta * (rho - 1)) + perturb_factor, rho - 1 + perturb_factor)
    ]
    test_points = [fixed_points_with_perturbation[0], fixed_points_with_perturbation[1]]
    idx = 1

    print("Running single point analysis...")
    print(f"transient time: {transient_time} and sync time: {sync_time}")
    print(f"train time: {train_time} and predict time: {predict_time}")
    print(f"Training on {len(train_points)} points")
    print(f"Training points: {train_points}")
    print(f"Testing points: {test_points[idx]}")

    print("Training reservoir using correct lorenz data function...")
    res = train_and_predict_reservoir(
        train_initial_points=train_points,
        train_time=train_time,
        rho=rho,
        predict_initial_point=test_points[idx],
        sync_time=sync_time,
        predict_time=predict_time,
        t_eval=t_eval,
        transient_time=transient_time,
        reservoir_params=reservoir_params
    )

    print("DONE")

if __name__ == "__main__":

    #run_single_point_analysis()
    run_reservoir_basin_analysis()