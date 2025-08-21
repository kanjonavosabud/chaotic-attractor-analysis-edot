import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def calculate_damping_rate(time_series, signal_series):
    """
    Calculate the damping rate of an oscillating signal.
    
    Parameters:
    -----------
    time_series : array-like
        Array of time points
    signal_series : array-like
        Array of signal values (the damped oscillation)
        
    Returns:
    --------
    zeta : float
        Damping coefficient
    """
    # Convert inputs to numpy arrays if they aren't already
    time = np.array(time_series)
    signal = np.array(signal_series)
    
    # Find peaks in the signal
    peak_indices, _ = find_peaks(signal, distance=10)
    peak_times = time[peak_indices]
    peak_amplitudes = signal[peak_indices]
    
    # Find troughs (peaks in the negative signal)
    trough_indices, _ = find_peaks(-signal, distance=10)
    trough_times = time[trough_indices]
    trough_amplitudes = signal[trough_indices]
    
    print(f"Found {len(peak_indices)} peaks and {len(trough_indices)} troughs")
    
    # Calculate damping rate from peaks
    # For a damped oscillation: A(t) = A0 * exp(-zeta * t)
    # Taking natural log: ln(A(t)) = ln(A0) - zeta * t
    
    # Get absolute amplitudes relative to the mean
    mean_signal = np.mean(signal)
    peak_deviations = np.abs(peak_amplitudes - mean_signal)
    
    # Take natural log of amplitudes
    log_amplitudes = np.log(peak_deviations)
    
    # Linear regression: ln(A) = ln(A0) - zeta * t
    slope, intercept = np.polyfit(peak_times, log_amplitudes, 1)
    
    # Damping coefficient is negative of slope
    zeta = -slope
    
    # Initial amplitude
    A0 = np.exp(intercept)
    
    # Plot the results
    plot_results(time, signal, peak_times, peak_amplitudes, 
                trough_times, trough_amplitudes, zeta, A0)
    
    return zeta

def plot_results(time, signal, peak_times, peak_amplitudes, 
                trough_times, trough_amplitudes, zeta, A0):
    """Plot the original signal with exponential envelope and amplitude decay."""
    plt.figure(figsize=(12, 8))
    
    # Plot original signal with peaks, troughs, and envelope
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, label='Signal')
    plt.plot(peak_times, peak_amplitudes, 'ro', label='Peaks')
    plt.plot(trough_times, trough_amplitudes, 'go', label='Troughs')
    
    # Create envelope curves
    mean_signal = np.mean(signal)
    envelope_upper = A0 * np.exp(-zeta * time) + mean_signal
    envelope_lower = -A0 * np.exp(-zeta * time) + mean_signal
    
    plt.plot(time, envelope_upper, 'k--', label=f'Decay Envelope: ζ = {zeta:.6f}')
    plt.plot(time, envelope_lower, 'k--')
    
    plt.title('Damped Oscillation Analysis')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Plot peak amplitudes on log scale
    plt.subplot(2, 1, 2)
    peak_deviations = np.abs(peak_amplitudes - mean_signal)
    trough_deviations = np.abs(trough_amplitudes - mean_signal)
    
    plt.semilogy(peak_times, peak_deviations, 'ro', label='Peak Deviations')
    plt.semilogy(trough_times, trough_deviations, 'go', label='Trough Deviations')
    plt.semilogy(time, A0 * np.exp(-zeta * time), 'k-', 
                label=f'Exp Decay: A(t) = {A0:.2f}·e^(-{zeta:.6f}·t)')
    
    plt.title('Log Scale: Amplitude Decay')
    plt.xlabel('Time')
    plt.ylabel('Log Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()