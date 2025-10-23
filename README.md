# FIR Filter Implementation and Analysis

This project provides a comprehensive implementation of FIR (Finite Impulse Response) filters in both Python and C, along with tools for analysis, comparison, and quality assessment. It's designed for educational purposes to demonstrate digital signal processing concepts and compare different implementation approaches.

## Project Structure

# FIR Filter Implementation and How to Run

This repository contains educational code for designing and testing FIR (Finite Impulse Response) filters. The implementation demonstrates:

- Filter design and visualization in Python (using scipy)
- Frame-based filtering in Python
- C implementations (float and fixed-point) to compare performance and numerical effects
- Utilities to resample audio and compute RMSE between outputs

The code is organized so you can generate coefficients and test signals in Python, process those signals in the C projects, and then compare results back in Python.

## Key locations

- `PythonProjects/FIR_FILTER/` — main Python tools:
   - `fir_filter_design.py` — design FIR filters (Remez), generate test signal, frame-based processing, saves `filter_coeffs.bin` and `test_signal.bin`.
   - `calculate_rmse.py` — read two float32 binary files and compute RMSE (dB) and plot signals + error.
   - `resample_signal.py` — resample audio using windowed-sinc FIR (uses `soundfile` to read WAVs).
   - `requirements.txt` — Python dependencies (numpy, scipy, matplotlib).

- `VSProjects/FIR_FILTER/` — C implementations and Visual Studio project files:
   - `fir_filter_with_circular_buffer.c` — float and fixed-point implementations using a circular approach (reads/writes binary files).
   - `fir_filter_wo_circular_bufffer.c` — frame-based FIR without circular buffer; uses fixed-point helpers; reads `filter_coeffs.bin` and `test_signal.bin` saved by Python.

## Quick setup (Windows PowerShell)

1. Install Python dependencies (prefer a virtual environment):

```powershell
cd .\PythonProjects\FIR_FILTER
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Generate filter coefficients and a test signal (run Python generator):

```powershell
# from the PythonProjects/FIR_FILTER folder
python .\fir_filter_design.py
```

This will produce at least the following files in that folder:
- `filter_coeffs.bin` (float32 binary, filter coefficients)
- `test_signal.bin` (float32 binary, test signal)
- `filtered_signal.bin` (Python filtered output)

3. Build and run the C implementation (Visual Studio):

- Open `VSProjects\FIR_FILTER\fir_filter_project.sln` in Visual Studio and build+run. The C project expects the Python-generated files in relative paths used by the C source (see notes below).

- Or compile from a command line using cl.exe (MSVC) or a cross-compiler, for example (PowerShell):

```powershell
# Example: compile the non-circular-buffer demo with MSVC (run from Developer Command Prompt for VS)
# cl /O2 /W3 /EHsc ..\..\PythonProjects\FIR_FILTER\fir_filter_wo_circular_bufffer.c
```

When the C executable runs it will read `filter_coeffs.bin` and `test_signal.bin` (paths are relative inside the C sources) and will write a binary output such as `out_msvc_wo_circ_buffer.bin`.

4. Compare results with Python RMSE tool:

```powershell
# From PythonProjects/FIR_FILTER folder, compare Python output to C output
python .\calculate_rmse.py .\filtered_signal.bin ..\..\VSProjects\FIR_FILTER\out_msvc_wo_circ_buffer.bin --verbose
```

Note: `calculate_rmse.py` expects float32 binary files. Adjust paths as needed.

## Important notes and troubleshooting

- Filenames in the C sources: some C files reference slightly different names (e.g. `fir_ceoffs_pygen.bin` or `input_pygen.bin`). If you run into "file not found" in C, either rename the Python output files to match the C expectation or edit the `fopen` lines in the C source to match the Python filenames (`filter_coeffs.bin`, `test_signal.bin`).

- Endianness / float format: Python writes IEEE-754 float32 using the host endianness. The C code assumes `sizeof(float)==4` and the same endianness — this is okay on typical Windows x86/x64 machines but be careful if moving files between different architectures.

- Fixed-point scaling: the C fixed-point code uses Q-format conversions and guard bits. If you compare Python (floating) output with C fixed-point output, small differences are expected due to quantization and rounding. Use `calculate_rmse.py` to quantify the difference.

- RMSE edge case: if two signals are identical, RMSE (in dB) becomes -inf with current code. You can modify `calculate_rmse.py` to special-case zero RMSE if desired.

## Running a full example (recommended)

1. From PowerShell (in `PythonProjects/FIR_FILTER`):

```powershell
# 1) design filter + generate test signal
python .\fir_filter_design.py

# 2) build/run C project (either from Visual Studio or command line) so it reads the .bin files and writes C output

# 3) compare
python .\calculate_rmse.py .\filtered_signal.bin ..\..\VSProjects\FIR_FILTER\out_msvc_wo_circ_buffer.bin --verbose
```

## Dependencies

- Python: numpy, scipy, matplotlib (see `PythonProjects/FIR_FILTER/requirements.txt`)
- Optional: `soundfile` for `resample_signal.py` if you want to use WAV I/O


## License

This repository uses the MIT license (see the LICENSE file if present).
- matplotlib==3.7.1
- soundfile==0.12.1
- pesq==0.0.4

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Youtube Video Link for Session 1

https://www.youtube.com/watch?v=I6rsuqhaX_U

## Youtube Video Link for Session 2

https://youtu.be/jFN6WyItc-I