import numpy as np
import matplotlib.pyplot as plt
from helper.lorenz import generate_lorenz_data, solve_lorenz
from helper.reservoir import create_reservoir, fit_reservoir, predict, run_reservoir
from helper.visualization import plot_lorenz_attractor, plot_prediction_comparison
from helper.analysis import compare_predictions, plot_mse, analyze_prediction
from helper.hyperparameter_tuning import *
from helper.dampcalc import *

def zero_vs_normal_exp():

    # Generate data
    x0 = [1,1,1]
    
    t_end = 500
    t_eval = .1
    t_span = (0,t_end)

    train_test_split = .8

    data = solve_lorenz(t_end=t_end, t_eval=t_eval, initial_state=x0, rho=28)
    plot_lorenz_attractor(data, verbose=False)

    # Split data
    split_idx = int((t_end/t_eval)*train_test_split)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Use the function for both cases
    # Original reservoir
    result_original = run_reservoir(train_data, test_data)
    predictions = result_original['predictions']

    # Zero adjacency matrix reservoir
    reservoir_params = {
            'output_dim': 3,
            'reservoir_dim': 300,
            'spectral_radius': 1.2,
            'edge_prob': 0.1,
            'graph_type': 'zero',
            'beta': 1.0
        }
    result_zero = run_reservoir(train_data, test_data, reservoir_params=reservoir_params)
    predictions_2 = result_zero['predictions']

    # Compare the predictions
    slope1, slope2, divs1, divs2, times = compare_predictions(
        actual_data=test_data,
        predicted_data1=predictions,
        predicted_data2=predictions_2,
        t_span=(0, len(test_data))
    )

    print("\nComparison of Reservoirs:")
    print(f"Original Reservoir divergence rate: {slope1:.4f}")
    print(f"Zero-Matrix Reservoir divergence rate: {slope2:.4f}")

    # Calculate MAE for both models
    mae1 = np.mean(np.exp(divs1))
    mae2 = np.mean(np.exp(divs2))

    print(f"\nMean Absolute Error:")
    print(f"Original Reservoir: {mae1:.4f}")
    print(f"Zero-Matrix Reservoir: {mae2:.4f}")

    # # Analyze original reservoir predictions
    # metrics_orig = analyze_prediction(
    #     test_data,
    #     predictions,
    #     model_name="Original Reservoir"
    # )

    # # Analyze simple reservoir predictions
    # metrics_simple = analyze_prediction(
    #     test_data,
    #     predictions_2,
    #     model_name="Zero Reservoir"
    # )

    # # Print comparison summary
    # print("\nComparison Summary:")
    # print(f"Original Reservoir MAE: {metrics_orig['mae']:.4f}")
    # print(f"Simple Reservoir MAE: {metrics_simple['mae']:.4f}")
    # print(f"Original Reservoir Horizon: {metrics_orig['horizon']:.2f} time units")
    # print(f"Simple Reservoir Horizon: {metrics_simple['horizon']:.2f} time units")

    # Visualize results
    plot_prediction_comparison(test_data, predictions) 
    plot_prediction_comparison(test_data, predictions_2)

    # Plotting MSE over time
    plot_mse(test_data, predictions, "Original", t_span)
    plot_mse(test_data, predictions_2, "Zero", t_span)

def param_tuning():
    
    param_grid = {
        'edge_prob': [.1, .2, .3],  # Adjusted to include a more varied set of parameters
        'graph_type': ["default","zero"],
        # 'reservoir_dim': [300]
    }

    initial_points = [[1, 1, 1]]#, [0, 0, 0], [2, 2, 2]]  # List of different initial points
    results = {}  # Dictionary to store results for each initial point

    best_mse = float('inf')  # Initialize best MSE to infinity
    worst_mse = float('-inf')  # Initialize worst MSE to negative infinity
    best_initial_point = None  # To store the best initial point
    worst_initial_point = None  # To store the worst initial point

    for x0 in initial_points:
        data = solve_lorenz(t_end=500, t_eval=.1, initial_state=x0, rho=22)
        train_data = data[:4400]
        test_data = data[4400:]

        model_func = run_model_func
        scoring_func = mse_scoring_func

        best_params, best_score, final_result = grid_search_tuning(
            train_data=train_data,
            test_data=test_data,
            param_grid=param_grid,
            model_func=model_func,
            scoring_func=scoring_func
        )

        # Store the results for the current initial point
        results[tuple(x0)] = {
            'best_params': best_params,
            'best_score': best_score
        }

        # Check for best and worst MSE
        if best_score < best_mse:
            best_mse = best_score
            best_initial_point = x0

        if best_score > worst_mse:
            worst_mse = best_score
            worst_initial_point = x0

    print(results, best_initial_point, best_mse, worst_initial_point, worst_mse)  # Return the results and best/worst info



def get_initial_points(sigma=10, rho=22, beta=8/3):

    y_min, y_max, y_count = -20, 20, 10
    z_min, z_max, z_count = -10, 70, 10

    # Initial conditions on the x=0 plane
    y_vals = np.linspace(y_min, y_max, y_count)  # Varying y
    z_vals = np.linspace(z_min, z_max, z_count)  # Varying z
    initial_conditions_x = [[0, y, z] for y in y_vals for z in z_vals]

    x_min, x_max, x_count = -20, 20, 10
    z_min, z_max, z_count = 0, 40, 10

    # Initial Conditions on the y=0 plane
    x_vals = np.linspace(x_min, x_max, x_count)  # Varying x
    z_vals = np.linspace(z_min, z_max, z_count)  # Varying z
    initial_conditions_y = [[x, 0, z] for x in x_vals for z in z_vals]

    # Initial Conditions on the fixed points
    fixed_points = [(np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
                    (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)]

    u_min, u_max, u_step = 0, 2, .01
    v_min, v_max, v_step = 0, 2, .01

    u = np.arange(u_min,u_max,u_step)
    v = np.arange(v_min,v_max,v_step)

    initial_conditions_f = [[u[i]*fixed_points[0][0] + v[j]*fixed_points[1][0],
            u[i]*fixed_points[0][1] + v[j]*fixed_points[1][1],
            u[i]*fixed_points[0][2] + v[j]*fixed_points[1][2]] for i in range(len(u)) for j in range(len(v))]
        
    return initial_conditions_x, initial_conditions_y, initial_conditions_f

def fixed_point_analysis():

    sigma, rho, beta  = 10, 22, 8/3

    initial_conditions = get_initial_points(sigma, rho, beta)[2]

    # Initial Conditions on the fixed points
    fixed_points = [(np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
                    (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)]

    # Solve for all initial conditions and classify trajectories
    trajectories = []
    classifications = []
    threshold = 1  # Distance threshold for classification

    for init in initial_conditions:
        # Solve the Lorenz system for the initial condition
        sol = solve_lorenz(initial_state=init, sigma=sigma, beta=beta, rho=rho)
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

        print(init)

    # Plot the initial conditions with classifications
    for init, classification in zip(initial_conditions, classifications):
        # Assign colors based on classification
        if classification == "Fixed Point #1": color = 'r'
        elif classification == "Fixed Point #2": color = 'g'
        else: color = 'black'

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
    # print(f"X MIN: {u_min}, X MAX: {u_max}, X STEP: {u_step}")
    # print(f"Z MIN: {v_min}, Z MAX: {v_max}, Z STEP: {v_step}")

def dampness_calc():
    # Generate data

    beta, rho, sigma = 8/3, 15 , 10
    
    fixed_points = [
        (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1),
        (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    ]

    dx = 0.0001
    x0 = [fixed_points[0][0]+dx,fixed_points[0][1]+dx,fixed_points[0][2]+dx]
    # x0 = [fixed_points[0][0],fixed_points[0][1],fixed_points[0][2]]


    t_end = 500
    t_eval = .01
    t_span = (0,t_end)

    train_test_split =  .9

    data2 = generate_lorenz_data(50000,rho=rho,x0=x0, h = 0.001)
    print(len(data2))
    plot_lorenz_attractor(data2, verbose=False)
    plot_prediction_comparison(data2,data2,"Just Actual")

      # Split data
    split_idx = int(500000*train_test_split)
    train_data = data2[:split_idx]
    test_data = data2[split_idx:]

    #plot_prediction_comparison(train_data,train_data,"Just train data")
    plot_prediction_comparison(test_data,test_data,"Just test data")


    # data = solve_lorenz(t_end=t_end, t_eval=t_eval, initial_state=x0, rho=22)
    # print(len(data))
    # plot_lorenz_attractor(data, verbose=False)
    # # plot_prediction_comparison(data,data,"Just Actual")

    # # Split data
    # split_idx = int((t_end/t_eval)*train_test_split)
    # train_data = data[:split_idx]
    # test_data = data[split_idx:]

    # # plot_prediction_comparison(train_data,train_data,"Just train data")
    # plot_prediction_comparison(test_data,test_data,"Just test data")
    
    # ts = np.arange(0,t_end,t_eval)
    # px = data.T[0]
    # # calculate_damping_rate(ts,px)

    # ts = np.arange(0,t_end,t_eval)[split_idx:]

    # DAMPING RATE ANALYSIS
    # px, py, pz = test_data.T[0], test_data.T[1], test_data.T[2]
    # zetaOx = calculate_damping_rate(ts,px)
    # zetaOy = calculate_damping_rate(ts,py)
    # zetaOz = calculate_damping_rate(ts,pz)
    # print(f"Normal Adjacency Matric Zetas: {zetaOx, zetaOy, zetaOz}")

    # # Use the function for both cases
    # # Original reservoir
    # result_original = run_reservoir(train_data, test_data)
    # predictions = result_original['predictions']

    # px, py, pz = predictions.T[0], predictions.T[1], predictions.T[2]
    # zetax = calculate_damping_rate(ts,px)
    # zetay = calculate_damping_rate(ts,py)
    # zetaz = calculate_damping_rate(ts,pz)

    # print(f"Normal Adjacency Matric Zetas: {zetax, zetay, zetaz}")

    # # Zero adjacency matrix reservoir
    # reservoir_params = {
    #         'output_dim': 3,
    #         'reservoir_dim': 300,
    #         'spectral_radius': 1.2,
    #         'edge_prob': 0.1,
    #         'graph_type': 'zero',
    #         'beta': 1.0
    #     }
    # result_zero = run_reservoir(train_data, test_data, reservoir_params=reservoir_params)
    # predictions_2 = result_zero['predictions']

    # px, py, pz = predictions_2.T[0], predictions_2.T[1], predictions_2.T[2]
    # zeta2x = calculate_damping_rate(ts,px)
    # zeta2y = calculate_damping_rate(ts,py)
    # zeta2z = calculate_damping_rate(ts,pz)

    # print(f"Normal Adjacency Matric Zetas: {zeta2x, zeta2y, zeta2z}")

# zero_vs_normal_exp()
# param_tuning()
fixed_point_analysis()
# dampness_calc()