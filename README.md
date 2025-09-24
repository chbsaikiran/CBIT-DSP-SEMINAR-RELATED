# FIR Filter Implementation and Analysis

This project provides a comprehensive implementation of FIR (Finite Impulse Response) filters in both Python and C, along with tools for analysis, comparison, and quality assessment. It's designed for educational purposes to demonstrate digital signal processing concepts and compare different implementation approaches.

## Project Structure

### Python Implementation (`PythonProjects/FIR_FILTER/`)

1. **`fir_filter_design.py`**
   - Core FIR filter implementation using the Remez exchange algorithm
   - Features:
     - Filter coefficient design with customizable parameters
     - Frame-based processing for real-time applications
     - Visualization of frequency and impulse responses
     - PCM audio file handling

2. **`calculate_rmse.py`**
   - Tool for comparing filter implementations
   - Calculates Root Mean Square Error (RMSE) between two signals
   - Provides visual analysis of signal differences
   - Useful for validating C vs Python implementations

3. **`downsample.py`**
   - Basic downsampling implementation
   - Reduces sample rate by skipping samples
   - Supports various input file formats

4. **`generate_pesq.py`**
   - PESQ (Perceptual Evaluation of Speech Quality) calculation
   - Supports both narrow-band (8kHz) and wide-band (16kHz)
   - Converts between different audio formats
   - Objective quality assessment of processed audio

5. **`resample_signal.py`**
   - Advanced signal resampling implementation
   - Proper interpolation methods for sample rate conversion

### C Implementation (`VSProjects/FIR_FILTER/`)

1. **`fir_filter_wo_circular_buffer.c`**
   - Direct FIR filter implementation in C
   - Demonstrates low-level signal processing
   - Useful for performance comparisons

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install Python dependencies:
   ```bash
   cd PythonProjects/FIR_FILTER
   pip install -r requirements.txt
   ```

3. For the C implementation:
   - Use Visual Studio to build the project in `VSProjects/FIR_FILTER/`
   - Or compile directly using your preferred C compiler

## Usage

### Python FIR Filter

```python
# Example usage of FIR filter design
from fir_filter_design import design_fir_filter

# Design a lowpass filter
filter_coeffs = design_fir_filter(
    fs=16000,        # Sampling frequency (Hz)
    fpass=4000,      # Passband frequency (Hz)
    fstop=5000,      # Stopband frequency (Hz)
    pass_ripple=1,   # Passband ripple (dB)
    stop_atten=40,   # Stopband attenuation (dB)
    plot=True        # Show filter response
)
```

### Comparing Implementations

```bash
# Compare Python and C implementations
python calculate_rmse.py python_output.bin c_output.bin --verbose
```

### Quality Assessment

```bash
# Calculate PESQ score
python generate_pesq.py original.pcm 16000 processed.pcm 16000
```

### Resampling

```bash
# Downsample a signal
python downsample.py input.pcm output.pcm 2  # Skip every other sample
```

## Features

- FIR filter design using Remez exchange algorithm
- Frame-based processing for real-time applications
- Multiple implementation comparisons (Python vs C)
- Signal quality assessment tools
- Comprehensive visualization capabilities
- Support for various audio formats
- Performance analysis tools

## Dependencies

- numpy==1.24.3
- scipy==1.10.1
- matplotlib==3.7.1
- soundfile==0.12.1
- pesq==0.0.4

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.