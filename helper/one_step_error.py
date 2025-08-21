import numpy as np
from helper.reservoir import run_reservoir


def compute_one_step_error(train_data, test_data, output_dim=3, reservoir_dim=300, spectral_radius=1.2, edge_prob=0.1, beta=1.0, graph_type="default"):
    """
    Compute the one-time-step error for each time step in test_data.
    
    For each time step t in test_data, the model is trained on all past data
    (train_data plus test_data up to t-1) and then predicts only the t-th time step.
    The MSE for that single prediction is returned.
    
    Parameters:
    -----------
    train_data : numpy.ndarray
        Training data for the model.
    test_data : numpy.ndarray
        Test data for prediction.
    reservoir_params : dict
        Parameters for the reservoir model.
        
    Returns:
    --------
    numpy.ndarray
        Array of MSE values for each one-time-step prediction.
    """

    reservoir_params={
        'output_dim': output_dim,
        'reservoir_dim': reservoir_dim,
        'spectral_radius': spectral_radius,
        'edge_prob': edge_prob,
        'graph_type': graph_type,
        'beta': beta
    }

    n_test = test_data.shape[0]
    predictions = []
    one_step_errors = np.zeros(n_test)
    
    for t in range(n_test):
        # Concatenate train_data and test_data up to the current time step
        past_data = np.vstack((train_data, test_data[:t]))
        # The target is only the next time step
        target = test_data[t:t+1]
        
        # Train the model on past_data and predict the next time step
        result = run_reservoir(past_data, target, reservoir_params=reservoir_params)

        prediction = result['predictions']
        
        predictions.append(prediction)
        # Compute MSE for this single prediction
        one_step_errors[t] = np.mean((prediction - target) ** 2)
    
    return predictions, one_step_errors 