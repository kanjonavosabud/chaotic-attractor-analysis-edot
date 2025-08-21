import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def compare_predictions(actual_data, predicted_data1, predicted_data2, t_span=(0, 50), n_points=10000):
    """
    Compare two different model predictions with actual data using Lyapunov-like divergence
    
    Parameters:
    actual_data: numpy array of shape (n_steps, 3) for true trajectory
    predicted_data1: numpy array of shape (n_steps, 3) for first model's predictions
    predicted_data2: numpy array of shape (n_steps, 3) for second model's predictions
    """
    # Compute distances for both predictions
    distances1 = np.linalg.norm(actual_data - predicted_data1, axis=1)
    distances2 = np.linalg.norm(actual_data - predicted_data2, axis=1)
    
    # Avoid log(0)
    distances1 = np.maximum(distances1, 1e-10)
    distances2 = np.maximum(distances2, 1e-10)
    
    # Compute log distances
    log_distances1 = np.log(distances1)
    log_distances2 = np.log(distances2)
    
    # Compute average divergence rates (similar to Lyapunov exponents)
    t = np.linspace(t_span[0], t_span[1], len(actual_data))
    slope1, _ = np.polyfit(t, log_distances1, 1)
    slope2, _ = np.polyfit(t, log_distances2, 1)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    plt.plot(t, log_distances1, 'b-', label='Orginal Divergence')
    plt.plot(t, log_distances2, 'r-', label='Zero-Matrix Divergence')
    plt.xlabel('Time')
    plt.ylabel('Log Distance from True Trajectory')
    plt.title('Comparison of Model Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return slope1, slope2, log_distances1, log_distances2, t

def analyze_prediction(test_data, predictions, model_name="Model", threshold=np.log(1.0)):
    """
    Analyze predictions from one model against test data.
    
    Parameters:
    -----------
    test_data : numpy.ndarray
        Actual time series data, shape (n_steps, n_features)
    predictions : numpy.ndarray
        Predictions from model, shape (n_steps, n_features)
    model_name : str, optional
        Name of the model for display purposes
    threshold : float, optional
        Threshold for prediction horizon calculation, default=log(1.0)
        
    Returns:
    --------
    dict
        Dictionary containing all analysis metrics
    """
    # Compare trajectories
    distances = np.linalg.norm(test_data - predictions, axis=1)
    distances = np.maximum(distances, 1e-10)  # Avoid log(0)
    log_distances = np.log(distances)
    
    # Compute average divergence rate
    times = np.linspace(0, len(test_data), len(test_data))
    slope, _ = np.polyfit(times, log_distances, 1)
    
    # Calculate mean absolute error
    mae = np.mean(np.exp(log_distances))
    
    # Calculate prediction horizon
    horizon = times[np.where(log_distances > threshold)[0][0]] if any(log_distances > threshold) else times[-1]
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    plt.plot(times, log_distances, 'b-', label=f'{model_name} Divergence')
    plt.xlabel('Time')
    plt.ylabel('Log Distance from True Trajectory')
    plt.title(f'Prediction Analysis for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print results
    print(f"Average divergence rate ({model_name}): {slope:.4f}")
    print(f"Mean Absolute Error ({model_name}): {mae:.4f}")
    print(f"Prediction Horizon ({model_name}): {horizon:.2f} time units")
    
    # Return metrics in a dictionary
    return {
        'divergence_rate': slope,
        'mae': mae,
        'horizon': horizon,
        'raw_data': {
            'times': times,
            'log_distances': log_distances
        }
    }

def plot_mse(test_data, predictions, series_name="Model Predictions", t_span=(0, 50)):
    """
    Plot Mean Squared Error (MSE) between predictions and test data over time.
    
    Parameters:
    predictions : numpy.ndarray
        Predictions from the model, shape (n_steps, n_features)
    test_data : numpy.ndarray
        Actual time series data, shape (n_steps, n_features)
    series_name : str, optional
        Name of the series for display purposes
    t_span : tuple, optional
        Time span for the x-axis, default=(0, 50)
    """
    # Compute MSE
    mse = np.mean((predictions - test_data) ** 2, axis=1)
    
    # Time array
    times = np.linspace(t_span[0], t_span[1], len(mse))
    
    # Plot MSE
    plt.figure(figsize=(12, 8))
    plt.plot(times, mse, 'g-', label='Mean Squared Error')
    plt.xlabel('Time')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Mean Squared Error Over Time for {series_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# metrics = analyze_prediction(test_data, predictions, model_name="Complex Reservoir")
#
# # Access specific metrics
# mae = metrics['mae']
# horizon = metrics['horizon']
#
# # Plot custom visualization using raw data
# plt.plot(metrics['raw_data']['times'], metrics['raw_data']['log_distances'])