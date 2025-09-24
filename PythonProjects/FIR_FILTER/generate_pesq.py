import numpy as np
import soundfile as sf
from pesq import pesq
from scipy.signal import resample_poly
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
            data = read_pcm_as_float(filename)
        else:
            data = np.fromfile(filename, dtype=np.float32)
        return data
    except Exception as e:
        raise IOError(f"Error reading file {filename}: {str(e)}")


def calculate_pesq(clean_file, degraded_file, mode="wb"):
    """
    Calculate PESQ score between a clean and degraded speech signal.

    Parameters:
    -----------
    clean_file : str
        Path to clean reference speech file (.wav)
    degraded_file : str
        Path to degraded speech file (.wav)
    mode : str
        "nb" for narrow-band (8 kHz), "wb" for wide-band (16 kHz)

    Returns:
    --------
    pesq_score : float
        PESQ MOS score
    """

    # Load signals
    clean, sr_c = sf.read(clean_file)
    degraded, sr_d = sf.read(degraded_file)

    # Ensure mono
    if clean.ndim > 1:
        clean = clean[:, 0]
    if degraded.ndim > 1:
        degraded = degraded[:, 0]

    # Resample to match ITU required rates
    target_sr = 16000 if mode == "wb" else 8000
    if sr_c != target_sr:
        clean = resample_poly(clean, target_sr, sr_c)
    if sr_d != target_sr:
        degraded = resample_poly(degraded, target_sr, sr_d)

    # Compute PESQ
    score = pesq(target_sr, clean, degraded, mode)
    return score

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Calculate PESQ between two files containing float32 values')
    parser.add_argument('clean_file', help='Path to clean_file')
    parser.add_argument('sr1', help='sampling freq in Hz of clean file')
    parser.add_argument('degraded_file', help='Path to degraded_file')
    parser.add_argument('sr2', help='sampling freq in Hz of degraded file')
    
    args = parser.parse_args()
    
    try:
        clean_signal = read_float_file(args.clean_file)
        sf.write("clean.wav", clean_signal, int(args.sr1), subtype="FLOAT")
        degrade_signal = read_float_file(args.degraded_file)
        sf.write("degrade.wav", degrade_signal, int(args.sr2), subtype="FLOAT")
        score = calculate_pesq("clean.wav", "degrade.wav", mode="nb")
        print("PESQ score:", score)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
