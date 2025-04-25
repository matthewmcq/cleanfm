# cleanfm
Implementation of the Dirichlet Kernel Deconvolution algorithm (parallel and single-threaded) for precise FFT parameter estimation and spectral leakage elimination. Repository for code contained in Matthew McQuistion's Applied Mathematics Honors Thesis at Brown University

## CleanFM: A Windowless Approach to Spectral Leakage and the Frequency Modulation Transform

CleanFM is a novel approach to spectral analysis that directly addresses the fundamental limitations of traditional windowing techniques in digital signal processing. The primary contributions of this project are:

1. **Dirichlet Kernel Deconvolution (DKD)**: A method for extracting precise frequency components from discrete spectra by modeling and removing the intrinsic spectral leakage pattern, preserving both resolution and information content.

2. **Parallel DKD Implementation**: An efficient parallel version of DKD that maintains accuracy while significantly reducing computational requirements.

3. **Applications of DKD**: Demonstration of DKD's utility in tasks such as idealized signal resampling, creating a "cleaned" Short-Time Fourier Transform (STFT) with superior resolution, and source separation without machine learning.

4. **Frequency Modulation Transform (FMT)**: A novel mathematical operator that reveals harmonic relationships within complex signals by analyzing the "frequencies of frequencies." (C++ implementation in progress, not included in this version.)

## Dependencies

This project requires the FFTW3 library for Fast Fourier Transform computations.

### Installing FFTW3

- On macOS (using Homebrew): brew install fftw3
- On Linux (using apt): sudo apt-get install libfftw3-dev
- On Windows: Download the precompiled FFTW3 libraries from http://www.fftw.org/install/windows.html
  - Extract the downloaded file and add the `lib` directory to your system's PATH environment variable

## Repository Structure

- `CMakeLists.txt`: CMake configuration file for building the project.
- `constants.h`: Header file defining various inline constants used throughout the project.
- `main.cpp`: Main driver program for the CleanDFT (DKD) spectral analysis tool.
- `cleandft.cpp`, `cleandft.h`: Implementation and declaration of the DKD algorithm.
- `dirichletkernel.cpp`, `dirichletkernel.h`: Implementation and declaration of the Dirichlet kernel calculations.
- `threadpool.h`: Defines helper structures and a basic ThreadPool class for parallel processing.
- `waveprocessor.cpp`, `waveprocessor.h`: Implements functions for reading, processing (FFT/IFFT), and writing WAV audio files.
- `thesis.tex`: LaTeX source for the project's thesis write-up, providing detailed mathematical foundations and discussions of the methods and applications.
- `cstft.sh`: Bash script to run the Cleaned STFT (DKD + STFT) algorithm.

## Building and Running

1. Ensure you have CMake, a C++20 compatible compiler, and FFTW3 installed.

2. If the FFTW3 library is not in a standard system path, you might need to update the include path in `CMakeLists.txt` and the source files:
- In `CMakeLists.txt`, update the `INTERFACE_INCLUDE_DIRECTORIES` property for the `FFTW3::FFTW3` target.
- In `waveprocessor.h`, update the `#include` directive for the FFTW3 header.

3. Create a build directory: `mkdir build && cd build`

4. Configure the project: `cmake ..`

5. Build the project: `make`

6. Run the compiled executable: `./cleanfm <options>`

Use `./cleanfm --help` to see available command line options.

## Applications and Future Work

In addition to the core DKD and FMT algorithms, this project explores several promising applications:



- Idealized frequency domain resampling that preserves spectral characteristics
- A "Cleaned STFT" with dramatically improved frequency resolution
- *Preliminary work on source separation using the FMT representation

*Not included in the current repository, as the C++ implementation is in progress
If you would like the python + bash scripts to run FMT please either:
- Message me on GitHub or
- contact me at matthewdmcquistion_at_gmail_dot_com

Future work will focus on completing the C++ implementation of FMT, further optimizing computational efficiency, and exploring applications of these techniques to areas such as audio analysis, feature extraction, sound transformation, and unsupervised learning.

## License

This project is released under the Apache 2.0 License.
