import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#import pdb

def design_fir_filter(fs, fpass, fstop, pass_ripple_db, stop_atten_db, plot=True):
    """
    Design an FIR filter using the Remez exchange algorithm.
    
    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    fpass : float
        Passband frequency in Hz (must be less than fs/2)
    fstop : float
        Stopband frequency in Hz (must be less than fs/2)
    pass_ripple_db : float
        Maximum passband ripple in dB
    stop_atten_db : float
        Minimum stopband attenuation in dB
    plot : bool
        Whether to plot the filter response
        
    Returns:
    --------
    b : ndarray
        FIR filter coefficients
        
    Raises:
    -------
    ValueError
        If fpass or fstop is greater than or equal to fs/2 (Nyquist frequency)
    """
    # Check if frequencies are valid
    if fpass >= fs/2 or fstop >= fs/2:
        raise ValueError("Passband and stopband frequencies must be less than fs/2 (Nyquist frequency)")
    # Convert frequencies to normalized frequency (0 to 0.5, where 0.5 is Nyquist frequency)
    nyq = fs / 2
    fpass_norm = fpass / fs  # Normalize to fs instead of nyq
    fstop_norm = fstop / fs  # Normalize to fs instead of nyq
    
    # Check if transition band is too narrow
    min_transition = 0.1  # Minimum transition width as fraction of passband frequency
    if (fstop - fpass) < (fpass * min_transition):
        raise ValueError(f"Transition band is too narrow. Minimum recommended width: {fpass * min_transition} Hz")
    
    # Convert ripple and attenuation specs from dB to linear units
    pass_ripple = 1 - 10**(-pass_ripple_db / 20)
    stop_ripple = 10**(-stop_atten_db / 20)
    
    # Estimate filter order using Kaiser's estimate
    transition_width = fstop_norm - fpass_norm
    A = -20 * np.log10(np.sqrt(pass_ripple * stop_ripple))  # Combined ripple spec
    N = int(np.ceil((A - 7.95) / (2.285 * 2 * np.pi * transition_width) + 1))
    N = N + 1 if N % 2 == 0 else N  # Make order odd
    
    # Limit maximum order to prevent excessive computation
    max_order = 1000
    if N > max_order:
        N = max_order
        print(f"Warning: Filter order limited to {max_order} for computational efficiency")
    
    # Define frequency bands and desired response
    bands = [0, fpass_norm, fstop_norm, 0.5]  # Changed 1 to 0.5 for proper normalization
    desired = [1, 0]
    
    # Define weight vector (how much to weight error in each band)
    weights = [1/pass_ripple, 1/stop_ripple]
    
    # Design filter using Remez algorithm
    b = signal.remez(N, bands, desired, weights)
    #pdb.set_trace()
    if plot:
        plot_filter_response(b, fs)
    
    return b

def plot_filter_response(b, fs):
    """
    Plot the frequency and impulse response of the designed filter.
    
    Parameters:
    -----------
    b : ndarray
        Filter coefficients
    fs : float
        Sampling frequency in Hz
    """
    # Create frequency vector for plotting
    w, h = signal.freqz(b, worN=8000)
    f = w * fs / (2 * np.pi)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot frequency response magnitude
    ax1.plot(f, 20 * np.log10(np.abs(h)))
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title('Frequency Response')
    ax1.grid(True)
    
    # Plot impulse response
    ax2.plot(np.arange(len(b)), b)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Impulse Response')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def generate_test_signal(fs, duration, seed=42):
    """
    Generate a test signal with controlled dynamic range variation in each frame.
    
    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    duration : float
        Signal duration in seconds
    seed : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    x : ndarray
        Test signal with controlled dynamic range:
        - Each 30ms frame contains mix of small (<1) and large (<8) amplitudes
        - Equal distribution of small and large values
        - Values are spread throughout the frame
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Calculate frame parameters
    frame_duration = 0.03  # 30ms
    samples_per_frame = int(frame_duration * fs)
    total_samples = int(duration * fs)
    num_frames = total_samples // samples_per_frame
    
    # Initialize signal array
    signal = np.zeros(total_samples)
    
    # Generate base frequencies for each amplitude range
    t = np.arange(total_samples) / fs
    
    # For each frame
    for frame in range(num_frames):
        start_idx = frame * samples_per_frame
        end_idx = start_idx + samples_per_frame
        frame_t = t[start_idx:end_idx]
        
        # Generate components for this frame
        # 1. Small amplitude components (< 1)
        small_amp = np.random.uniform(0.3, 0.8)
        small_freq = np.random.choice([50, 100, 150])
        small_phase = np.random.uniform(0, 2*np.pi)
        
        # 2. Large amplitude components (< 8)
        large_amp = np.random.uniform(5, 7)
        large_freq = np.random.choice([200, 250, 300])
        large_phase = np.random.uniform(0, 2*np.pi)
        
        # Create frame signal
        frame_signal = np.zeros_like(frame_t)
        
        # Add small amplitude component
        frame_signal += small_amp * np.sin(2 * np.pi * small_freq * frame_t + small_phase)
        
        # Add large amplitude component
        # Use a window to ensure smooth transitions
        window = np.hanning(len(frame_t))
        large_component = large_amp * np.sin(2 * np.pi * large_freq * frame_t + large_phase)
        
        # Randomly position the large amplitude component
        position = np.random.uniform(0.2, 0.8)  # Position in the frame (20% to 80%)
        center = int(len(frame_t) * position)
        window_width = int(len(frame_t) * 0.3)  # 30% of frame length
        start_pos = max(0, center - window_width//2)
        end_pos = min(len(frame_t), center + window_width//2)
        
        # Apply windowed large component
        local_window = np.hanning(end_pos - start_pos)
        frame_signal[start_pos:end_pos] += large_component[start_pos:end_pos] * local_window
        
        # Add some randomness while maintaining amplitude constraints
        noise_small = 0.2 * np.random.randn(len(frame_t))
        noise_large = np.zeros_like(frame_t)
        large_positions = np.random.choice(len(frame_t), size=int(len(frame_t)*0.2), replace=False)
        noise_large[large_positions] = 2 * np.random.randn(len(large_positions))
        
        frame_signal += noise_small + noise_large * window
        
        # Ensure amplitude constraints
        frame_signal = np.clip(frame_signal, -8, 8)
        
        # Add to main signal
        signal[start_idx:end_idx] = frame_signal
    
    # Apply smoothing between frames to avoid discontinuities
    frame_transitions = np.arange(1, num_frames) * samples_per_frame
    for trans in frame_transitions:
        window = np.hanning(20)
        signal[trans-10:trans+10] *= window
    
    # Reset random seed to avoid affecting other parts of the code
    np.random.seed(None)
    
    # Verify dynamic range
    print("\nSignal Statistics per frame:")
    for frame in range(min(5, num_frames)):  # Print first 5 frames
        start_idx = frame * samples_per_frame
        end_idx = start_idx + samples_per_frame
        frame_signal = signal[start_idx:end_idx]
        small_values = np.sum(np.abs(frame_signal) < 1)
        large_values = np.sum(np.abs(frame_signal) > 5)
        print(f"Frame {frame}: Min: {np.min(frame_signal):.2f}, Max: {np.max(frame_signal):.2f}, "
              f"Small values: {small_values}, Large values: {large_values}")
    
    return signal

def process_signal_in_frames(x, b, frame_size):
    """
    Process signal using FIR filter coefficients in frames.
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    b : ndarray
        FIR filter coefficients
    frame_size : int
        Number of samples per frame
        
    Returns:
    --------
    y : ndarray
        Filtered output signal
    """
    # Initialize output array
    y = np.zeros(len(x))
    
    # Calculate overlap size based on filter length
    overlap = len(b) - 1
    
    # Initialize buffer for overlap-add
    overlap_buffer = np.zeros(overlap)
    
    # Process frames
    for i in range(0, len(x), frame_size):
        # Get current frame
        frame = x[i:min(i + frame_size, len(x))]
        
        # Pad frame with overlap
        frame_padded = np.concatenate([overlap_buffer, frame])
        
        # Filter the frame
        frame_filtered = signal.lfilter(b, 1, frame_padded)
        
        # Save the overlap for next frame
        overlap_buffer = frame_padded[-(overlap):]
        
        # Add filtered frame to output (excluding the overlap)
        y[i:min(i + frame_size, len(x))] = frame_filtered[overlap:]
    
    return y

# Example usage
if __name__ == "__main__":
    # Example filter specifications
    fs = 1000        # Sampling frequency (Hz)
    fpass = 100      # Passband frequency (Hz)
    fstop = 250      # Stopband frequency (Hz) - Increased for wider transition band
    pass_ripple = 1  # Passband ripple (dB)
    stop_atten = 40  # Stopband attenuation (dB) - Reduced for better convergence
    
    # Check if frequencies are valid
    if fpass >= fs/2 or fstop >= fs/2:
        raise ValueError("Passband and stopband frequencies must be less than fs/2 (Nyquist frequency)")
    
    # Design and plot the filter
    filter_coeffs = design_fir_filter(
        fs=fs,
        fpass=fpass,
        fstop=fstop,
        pass_ripple_db=pass_ripple,
        stop_atten_db=stop_atten,
        plot=True  # Show filter response
    )

    np.random.seed(42)
    # Generate random noise in range [-8, 8]
    noise = np.random.uniform(-3, 3, size=filter_coeffs.shape)
    filter_coeffs += noise
    np.random.seed(None)

    print(f"Filter order: {len(filter_coeffs) - 1}")
    print(f"Number of coefficients: {len(filter_coeffs)}")
    print(f"filter coefficients data type: max {np.max(filter_coeffs)}, min {np.min(filter_coeffs)},sum {np.sum(np.abs(filter_coeffs))}, noise {np.max(noise)}, noise sum {np.sum(np.abs(noise))}")

    filter_coeffs_float32 = filter_coeffs.astype(np.float32)
    with open('filter_coeffs.bin', 'wb') as f:
        filter_coeffs_float32.tofile(f)
    
    # Generate test signal (3 seconds duration)
    duration = 3
    test_signal = generate_test_signal(fs, duration)
    
    # Convert to float32 and save to binary file
    test_signal_float32 = test_signal.astype(np.float32)
    with open('test_signal.bin', 'wb') as f:
        test_signal_float32.tofile(f)
    print(f"Test signal data type: max {np.max(test_signal)}, min {np.min(test_signal)}")
    
    print(f"\nTest signal info:")
    print(f"Number of samples: {len(test_signal)}")
    print(f"Duration: {duration} seconds")
    #print(f"Data type: {test_signal_float32.dtype}")
    #print(f"File size: {len(test_signal_float32) * 4} bytes")  # float32 = 4 bytes per sample
    
    # Calculate frame size for 30ms frames
    frame_size = int(0.03 * fs)  # 30ms * 1000Hz = 30 samples
    print(f"\nProcessing signal with {frame_size} samples per frame (30ms at {fs}Hz)")
    
    # Process signal in frames
    filtered_signal = process_signal_in_frames(test_signal, filter_coeffs, frame_size)
    filtered_signal_float32 = filtered_signal.astype(np.float32)
    with open('filtered_signal.bin', 'wb') as f:
        filtered_signal_float32.tofile(f)
    
    # Plot original and filtered signals
    t = np.arange(0, duration, 1/fs)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, test_signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_signal)
    plt.title('Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nSignal Statistics:")
    print(f"Original signal range: [{np.min(test_signal):.2f}, {np.max(test_signal):.2f}]")
    print(f"Filtered signal range: [{np.min(filtered_signal):.2f}, {np.max(filtered_signal):.2f}]")
    print(f"Original signal RMS: {np.sqrt(np.mean(test_signal**2)):.2f}")
    print(f"Filtered signal RMS: {np.sqrt(np.mean(filtered_signal**2)):.2f}")
