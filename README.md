# FIR Filter Implementation and Comparison

This project implements a Finite Impulse Response (FIR) filter in both Python and C, demonstrating different implementation approaches and comparing their performance. The project includes filter design, test signal generation, frame-based processing, and tools for comparing different implementations.

## Project Structure

```
├── PythonProjects/
│   └── FIR_FILTER/
│       ├── fir_filter_design.py    # Main Python implementation
│       ├── calculate_rmse.py       # Utility for comparing outputs
│       └── requirements.txt        # Python dependencies
│
└── VSProjects/
    └── FIR_FILTER/
        ├── fir_filter_with_circular_buffer.c    # C implementation with circular buffer
        └── fir_filter_wo_circular_bufffer.c     # C implementation without circular buffer
```

## Features

- FIR filter design using Parks-McClellan (Remez) algorithm
- Test signal generation with varying dynamic range
- Frame-based signal processing (30ms frames)
- Both floating-point and fixed-point implementations
- Performance comparison tools
- Visualization of filter responses and signals

## Requirements

### Python Implementation
- Python 3.10 or higher
- Required packages (install using `pip install -r requirements.txt`):
  - numpy
  - scipy
  - matplotlib

### C Implementation
- Visual Studio 2022 or compatible C compiler
- Windows environment (for VS projects)

## Running the Code

### 1. Python Implementation

1. Set up Python environment:
   ```bash
   cd PythonProjects/FIR_FILTER
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the filter design and processing:
   ```bash
   python fir_filter_design.py
   ```
   This will:
   - Design the FIR filter
   - Generate and process test signals
   - Save results in binary files
   - Display filter response and signal plots

### 2. C Implementation

1. Open the solution in Visual Studio:
   - Navigate to `VSProjects/FIR_FILTER`
   - Open `fir_filter_project.sln`

2. Build and run the project:
   - Select Release configuration
   - Build the solution (F7)
   - Run the program (F5)

### 3. Comparing Results

Use the RMSE calculator to compare outputs:
```bash
cd PythonProjects/FIR_FILTER
python calculate_rmse.py filtered_signal.bin ../../VSProjects/FIR_FILTER/out_msvc_wo_circ_buffer.bin
```

## Filter Specifications

The default filter parameters are:
- Sampling frequency: 1000 Hz
- Passband frequency: 100 Hz
- Stopband frequency: 250 Hz
- Passband ripple: 1 dB
- Stopband attenuation: 40 dB

## Test Signal Characteristics

The test signal includes:
- Low frequency (50 Hz) with increasing amplitude
- Medium frequency (150 Hz) with amplitude modulation
- High frequency (300 Hz) with decreasing amplitude
- Random bursts at fixed positions (controlled by seed)
- Duration: 3 seconds

## Implementation Details

### Python Version
- Uses SciPy's `signal.remez` for filter design
- Implements frame-based processing with proper overlap handling
- Generates test signals with controlled randomness
- Saves data in float32 binary format

### C Version
- Implements both circular and linear buffer approaches
- Supports fixed-point arithmetic for optimization
- Processes signals in frames
- Handles file I/O for coefficients and signals

## Verification

1. Check filter response plots:
   - Verify frequency response meets specifications
   - Examine impulse response characteristics

2. Compare signal plots:
   - Original vs filtered signal
   - Check for proper filtering effects
   - Verify no artifacts at frame boundaries

3. Use RMSE calculator:
   - Compare Python and C implementations
   - Check detailed statistics with `-v` flag

## Common Issues and Solutions

1. **Filter Design Fails to Converge**
   - Increase transition width between passband and stopband
   - Reduce filter order
   - Relax ripple/attenuation requirements

2. **Frame Processing Artifacts**
   - Verify overlap-add implementation
   - Check frame size and overlap calculations
   - Ensure proper buffer management

3. **Binary File Issues**
   - Verify file paths are correct
   - Check endianness if comparing across platforms
   - Ensure float32 format is consistent

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open-source and available under the MIT License.
