import numpy as np
import argparse
import matplotlib.pyplot as plt

def read_float_file(filename):
    """
    Read a binary file containing float32 values.
    
    Parameters:
    -----------
    filename : str
        Path to the binary file
        
    Returns:
    --------
    data : ndarray
        Array of float32 values
    """
    try:
        data = np.fromfile(filename, dtype=np.float32)
        return data
    except Exception as e:
        raise IOError(f"Error reading file {filename}: {str(e)}")

import numpy as np

def calculate_rmse(data1, data2):
    """
    Calculate Root Mean Square Error between two arrays.
    If arrays have different lengths, truncate to the shorter one.

    Parameters:
    -----------
    data1, data2 : ndarray
        Input arrays to compare

    Returns:
    --------
    rmse_db : float
        Root Mean Square Error in dB
    error : ndarray
        Error signal (difference between truncated data1 and data2)
    """
    # Use minimum length
    min_len = min(len(data1), len(data2))
    d1 = data1[:min_len]
    d2 = data2[:min_len]

    # Error signal
    error = d1 - d2

    # RMSE calculation
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    return (20 * np.log10(rmse)), error, min_len


def plot_error_analysis(data1, data2, error,min_len, fs=16000):
    """
    Plot the original signals and their error.
    
    Parameters:
    -----------
    data1, data2 : ndarray
        Input arrays being compared
    error : ndarray
        Error signal (difference between data1 and data2)
    fs : float
        Sampling frequency in Hz (default: 1000)
    """
    # Create time vector
    t = np.arange(len(data1[:min_len])) / fs
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot first signal
    ax1.plot(t, data1[:min_len], label='Signal 1')
    ax1.set_title('Signal 1 (Python Implementation)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Plot second signal
    ax2.plot(t, data2[:min_len], label='Signal 2', color='orange')
    ax2.set_title('Signal 2 (C Implementation)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    # Plot error signal
    ax3.plot(t, error[:min_len], label='Error', color='red')
    ax3.set_title('Error Signal (Signal 1 - Signal 2)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True)
    
    # Add error statistics as text
    stats_text = f'Error Statistics:\n'
    stats_text += f'Mean: {np.mean(error):.6f}\n'
    stats_text += f'Std: {np.std(error):.6f}\n'
    stats_text += f'Max: {np.max(np.abs(error)):.6f}'
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Calculate RMSE between two files containing float32 values')
    parser.add_argument('file1', help='Path to first file')
    parser.add_argument('file2', help='Path to second file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed statistics')
    parser.add_argument('--fs', type=float, default=1000, help='Sampling frequency in Hz (default: 1000)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    try:
        # Read the files
        print(f"Reading file: {args.file1}")
        data1 = read_float_file(args.file1)
        print(f"Reading file: {args.file2}")
        data2 = read_float_file(args.file2)
        
        # Calculate RMSE and get error signal
        rmse, error, min_len = calculate_rmse(data1, data2)
        
        # Print results
        print(f"min_len: {min_len}")
        print("\nResults:")
        print(f"RMSE: {rmse:.6f} dB")
        
        if args.verbose:
            print("\nDetailed Statistics:")
            print(f"Number of samples: {len(data1)}")
            print(f"File 1 - Mean: {np.mean(data1):.6f}, Std: {np.std(data1):.6f}")
            print(f"File 2 - Mean: {np.mean(data2):.6f}, Std: {np.std(data2):.6f}")
            print(f"Maximum absolute difference: {np.max(np.abs(error)):.6f}")
            print(f"Error signal - Mean: {np.mean(error):.6f}, Std: {np.std(error):.6f}")
        
        # Plot signals and error if not disabled
        if not args.no_plot:
            plot_error_analysis(data1, data2, error, min_len, args.fs)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
