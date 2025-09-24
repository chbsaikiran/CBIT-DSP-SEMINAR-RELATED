import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

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
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pcm":
            data = read_pcm_as_float(filename, dtype=np.float32)
        else:
            data = np.fromfile(filename, dtype=np.float32)
        return data
    except Exception as e:
        raise IOError(f"Error reading file {filename}: {str(e)}")

def skip_samples(input_filename, output_filename, skip):
    """
    Reads float values from input file, skips samples, and writes the result as binary float32.

    Parameters:
    -----------
    input_filename : str
        Path to the input file containing float values (one per line).
    output_filename : str
        Path to the output file (binary floats).
    skip : int
        Skip value (e.g., 1 -> take every other sample: 0, 2, 4, ...).
    """

    data = read_float_file(input_filename)  # assumes this returns a list/array of floats

    # Select every (skip+1)-th element
    selected = data[::skip+1]

    selected_float32 = selected.astype(np.float32)
    with open(output_filename, 'wb') as f:
        selected_float32.tofile(f)

    print(f"Wrote {len(selected)} float32 samples to {output_filename}")

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Calculate RMSE between two files containing float32 values')
    parser.add_argument('file1', help='Path to first file')
    parser.add_argument('file2', help='Path to second file')
    parser.add_argument('skip', help='Skip value')
    
    args = parser.parse_args()
    
    try:
        skip_samples(args.file1, args.file2, int(args.skip))
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
