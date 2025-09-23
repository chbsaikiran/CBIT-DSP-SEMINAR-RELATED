import numpy as np
import argparse

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

def calculate_rmse(data1, data2):
    """
    Calculate Root Mean Square Error between two arrays.
    
    Parameters:
    -----------
    data1, data2 : ndarray
        Input arrays to compare
        
    Returns:
    --------
    rmse : float
        Root Mean Square Error
    """
    if len(data1) != len(data2):
        raise ValueError(f"Arrays have different lengths: {len(data1)} vs {len(data2)}")
    
    mse = np.mean((data1 - data2) ** 2)
    rmse = np.sqrt(mse)
    return (10*np.log10(rmse))

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Calculate RMSE between two files containing float32 values')
    parser.add_argument('file1', help='Path to first file')
    parser.add_argument('file2', help='Path to second file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed statistics')
    
    args = parser.parse_args()
    
    try:
        # Read the files
        print(f"Reading file: {args.file1}")
        data1 = read_float_file(args.file1)
        print(f"Reading file: {args.file2}")
        data2 = read_float_file(args.file2)
        
        # Calculate RMSE
        rmse = calculate_rmse(data1, data2)
        
        # Print results
        print("\nResults:")
        print(f"RMSE: {rmse:.6f}")
        
        if args.verbose:
            print("\nDetailed Statistics:")
            print(f"Number of samples: {len(data1)}")
            print(f"File 1 - Mean: {np.mean(data1):.6f}, Std: {np.std(data1):.6f}")
            print(f"File 2 - Mean: {np.mean(data2):.6f}, Std: {np.std(data2):.6f}")
            print(f"Maximum absolute difference: {np.max(np.abs(data1 - data2)):.6f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
