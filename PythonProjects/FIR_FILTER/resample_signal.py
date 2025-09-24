import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
#import pdb

def sinc_filter(cutoff, num_taps, up):
    """
    Design a low-pass FIR filter using windowed sinc.
    cutoff: cutoff frequency (0..0.5) relative to Nyquist
    num_taps: number of filter coefficients
    up: upsampling factor (needed for gain compensation)
    """
    M = num_taps // 2
    n = np.arange(-M, M+1)
    h = np.sinc(2 * cutoff * n)
    window = 0.54 - 0.46 * np.cos(2*np.pi*(n+M)/(num_taps))
    h = h * window
    h = h / np.sum(h) * up  # normalize for upsampling
    return h

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

def read_pcm_as_float(filename, dtype=np.int16, channels=1, normalize=True):
    """
    Read a raw PCM file and return samples as float32 values.

    Parameters:
        filename (str): Path to .pcm file
        dtype (np.dtype): Data type of PCM (e.g., np.int16, np.int32, np.uint8, np.float32)
        channels (int): Number of channels (1 = mono, 2 = stereo, etc.)
        normalize (bool): If True, scale integer PCM to [-1.0, 1.0]

    Returns:
        np.ndarray: Audio samples as float32, shape (num_samples,) or (num_frames, channels)
    """
    with open(filename, "rb") as f:
        raw_data = f.read()

    # Convert to NumPy array
    pcm_data = np.frombuffer(raw_data, dtype=dtype)

    # Handle multi-channel
    if channels > 1:
        pcm_data = pcm_data.reshape(-1, channels)

    # Normalize if integer type
    if normalize and np.issubdtype(dtype, np.integer):
        float_data = pcm_data.astype(np.float32) / np.iinfo(dtype).max
    else:
        float_data = pcm_data.astype(np.float32)

    return float_data

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
    fs_out = 8000
    test_signal, fs = sf.read("voice-sample.wav")
    duration = len(test_signal) / fs
    # Convert to float32 and save to binary file
    test_signal_float32 = test_signal.astype(np.float32)
    with open('test_signal.bin', 'wb') as f:
        test_signal_float32.tofile(f)
    print(f"Test signal data type: max {np.max(test_signal)}, min {np.min(test_signal)},first 10 samples {test_signal[:10]}")
    print(f"\nTest signal info:")
    print(f"Number of samples: {len(test_signal)}")
    print(f"Duration: {duration} seconds")

    # Rational factors
    up = 80
    down = 441
    # Filter design
    num_taps = 511   # length of FIR
    cutoff = 1.0 / (2*down)   # normalized cutoff (relative to upsampled fs)
    filter_coeffs = sinc_filter(cutoff, num_taps, up)
    print(f"Filter order: {len(filter_coeffs) - 1}")
    print(f"Number of coefficients: {len(filter_coeffs)}")
    print(f"filter coefficients data type: max {np.max(filter_coeffs)}, min {np.min(filter_coeffs)},sum {np.sum(np.abs(filter_coeffs))}")

    filter_coeffs_float32 = filter_coeffs.astype(np.float32)
    with open('filter_coeffs.bin', 'wb') as f:
        filter_coeffs_float32.tofile(f)
    
    # Calculate frame size for 30ms frames
    frame_size = int(0.02 * fs)  # 20ms * 44100Hz = 882 samples
    print(f"\nProcessing signal with {frame_size} samples per frame (10ms at {fs}Hz)")
    
    # Upsample: insert zeros
    upsampled = np.zeros(len(test_signal) * up)
    upsampled[::up] = test_signal

    # Convolution (FIR filter)
    filtered_signal = process_signal_in_frames(upsampled, filter_coeffs, frame_size)

    # Downsample
    filtered_signal = filtered_signal[::down]
    filtered_signal_float32 = filtered_signal.astype(np.float32)
    with open('filtered_signal.bin', 'wb') as f:
        filtered_signal_float32.tofile(f)
    
    # Plot original and filtered signals
    t1 = np.arange(0, duration, 1/fs)
    t2 = np.arange(0, duration, 1/fs_out)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(t1, test_signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t2, filtered_signal)
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
