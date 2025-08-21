import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from helper.reservoir import run_reservoir
from helper.lorenz import generate_lorenz_data
from itertools import product
from helper.one_step_error import compute_one_step_error

def run_model_func(train_data, test_data, output_dim=3, reservoir_dim=300, spectral_radius=1.2, edge_prob=0.1, beta=1.0, graph_type="default"):
    """
    Create, train, and predict using a reservoir computing model.
    
    Parameters:
    -----------
    train_data : numpy.ndarray
        Training data for the model.
    test_data : numpy.ndarray
        Test data for prediction.
    output_dim : int, optional
        Number of output dimensions, default is 3.
    reservoir_dim : int, optional
        Number of neurons in the reservoir, default is 300.
    spectral_radius : float, optional
        Spectral radius of the reservoir, default is 1.2.
    edge_prob : float, optional
        Probability of edges in the reservoir, default is 0.1.
        
    Returns:
    --------
    dict
        Contains predictions and other relevant information.
    """
    
    # Create and train the reservoir
    reservoir_params={
        'output_dim': output_dim,
        'reservoir_dim': reservoir_dim,
        'spectral_radius': spectral_radius,
        'edge_prob': edge_prob,
        'graph_type': graph_type,
        'beta': beta
    }
    
    result = run_reservoir(train_data, test_data, reservoir_params=reservoir_params)
    
    return result

def calculate_prediction_horizon(predictions, actual_data, threshold=30):
    """
    Calculate the time until predictions diverge from actual data.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Predicted values from the model
    actual_data : numpy.ndarray
        Actual values for comparison
    threshold : float
        MSE threshold above which we consider the prediction to have diverged
        
    Returns:
    --------
    int
        Number of timesteps before divergence
    """
    # compute the full error‐series
    mse_per_timestep = (predictions - actual_data)**2

    # just for display, look at the first 10:
    print("first 10 MSE:", mse_per_timestep[:20])

    # now find when it crosses your threshold:
    divergence_points = np.where(mse_per_timestep > threshold)[0]
    if divergence_points.size == 0:
        return len(predictions)  # Never diverged
    
    return divergence_points[0]  # Return first divergence point

def mse_scoring_func(predictions, actual_data):
    """
    Calculate Mean Squared Error (MSE) between predictio ns and actual data.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Predicted values from the model.
    actual_data : numpy.ndarray
        Actual values for comparison.
        
    Returns:
    --------
    float
        The calculated MSE.
    """
    return np.mean((predictions - actual_data) ** 2)

def plot_mse_over_time(predictions, actual_data):
    """Plot MSE over time to help determine appropriate threshold"""
    mse_per_timestep = np.mean((predictions - actual_data) ** 2, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mse_per_timestep)
    plt.yscale('log')  # Use log scale to better see the divergence
    plt.xlabel('Time step')
    plt.ylabel('MSE (log scale)')
    plt.title('MSE over time')
    plt.grid(True)
    plt.show() 

def grid_search_OTSE(train_data, test_data, param_grid):
    """
    Perform grid search tuning optimizing for one-time-step error.
    
    Parameters:
    -----------
    train_data : numpy.ndarray
        Training data for the model.
    test_data : numpy.ndarray
        Test data for prediction.
    param_grid : dict
        Dictionary of parameters to try.
        
    Returns:
    --------
    best_params : dict
        Parameters giving the lowest average one-time-step error.
    best_score : float
        Average one-time-step error for the best parameters.
    all_results : dict
        Dictionary containing results for all parameter combinations.
    """

    best_score = np.inf
    best_params = None
    all_results = {}  # Store results for all parameter combinations

    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(product(*param_grid.values()))

    for values in param_values:
        params = dict(zip(param_names, values))
        print(f"Testing parameters: {params}")

        # Compute one-time-step errors for the current parameters
        preds, one_step_errors = compute_one_step_error(train_data, test_data, **params)
        avg_error = np.mean(one_step_errors)

        # Store results for this parameter combination
        all_results[str(params)] = {
            'params': params,
            'avg_one_step_error': avg_error
        }

        print(f"Average one-time-step error for parameters: {avg_error:.4f}")

        # Update best parameters if current score is better
        if avg_error < best_score:
            best_score = avg_error
            best_params = params

    print(f"\nBest parameters: {best_params}")
    print(f"Best average one-time-step error: {best_score:.4f}")

    return best_params, best_score, all_results

def grid_search_tuning(train_data, test_data, param_grid, model_func, scoring_func):
    """
    Perform grid search tuning optimizing for prediction horizon.
    
    Parameters:
    -----------
    train_data : numpy.ndarray
        Training data for the model
    test_data : numpy.ndarray
        Test data for final evaluation
    param_grid : dict
        Dictionary of parameters to try
    model_func : function
        Model function to evaluate
    scoring_func : function
        Scoring function (not used when optimizing for prediction horizon)
    
    Returns:
    --------
    best_params : dict
        Parameters giving longest prediction horizon
    best_score : float
        MSE score for best parameters
    final_result : dict
        Final model results
    best_horizon : int
        Longest prediction horizon achieved
    all_results : dict
        Dictionary containing results for all parameter combinations
    """

    best_horizon = 0
    best_params = None
    best_score = np.inf
    all_results = {}  # Store results for all parameter combinations

    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(product(*param_grid.values()))

    for values in param_values:
        params = dict(zip(param_names, values))
        print(f"Testing parameters: {params}")


        X_train, X_val = train_data, test_data
        
        # Run the model with current parameters
        result = model_func(X_train, X_val, **params)
        
        # Calculate both MSE score and prediction horizon
        score = scoring_func(result['predictions'], X_val)
        horizon = calculate_prediction_horizon(result['predictions'], X_val)


        
        # Store results for this parameter combination
        all_results[str(params)] = {
            'params': params,
            'horizon': horizon,
            'score': score
        }
        
        print(f"horizon for parameters: {horizon:.2f}")
        print(f"score for parameters: {score:.4f}")

        # # Update best parameters if current score is better
        # if score < best_score:
        #     best_horizon = horizon
        #     best_params = params
        #     best_score = score

        # Update best parameters if current horizon is better
        if horizon > best_horizon:
            best_horizon = horizon
            best_params = params
            best_score = score

    print(f"\nBest parameters: {best_params}")
    print(f"Best prediction horizon: {best_horizon:.2f}")
    print(f"MSE score for best parameters: {best_score:.4f}")

    # Final evaluation on test data with best parameters
    final_result = model_func(train_data, test_data, **best_params)
    
    # Calculate final prediction horizon
    final_horizon = calculate_prediction_horizon(
        final_result['predictions'], 
        test_data
    )
    
    return best_params, best_score, final_result, final_horizon, all_results 