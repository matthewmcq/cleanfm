/**
 * @file main.cpp
 * @author Matthew McQuistion
 * @date 04/25/25
 * @brief Command-line driver for the CleanDFT (DKD) spectral analysis tool.
 *
 * This program serves as the main executable for the Dirichlet Kernel Deconvolution
 * (DKD) implementation. It parses command-line arguments to configure the analysis,
 * loads audio data, utilizes the CleanDFT library functions for deconvolution
 * (component extraction) and decompression (signal reconstruction), handles file I/O
 * (WAV, CSV), performs timing benchmarks, and reports results including accuracy
 * metrics like Log Spectral Distance (LSD) if requested.
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <time.h>        // For clock_gettime, clock_t
#include <sys/resource.h> // For getrusage
#include <string>
#include <atomic>        // For atomic<int> in thread timing (if implemented)
#include <cmath>         // For log10, sqrt, abs

#include "fft/waveprocessor.h"
#include "deconvolve/cleandft.h" // Assumes cleandft.h defines MAX_COMPONENTS or similar

// Forward declaration if needed, or ensure it's defined elsewhere
// const int MAX_COMPONENTS = 10000; // Example default if not in cleandft.h

/**
 * @brief Writes extracted frequency components to a CSV file.
 *
 * Each row in the CSV file represents a component with its frequency in Hz,
 * phase in radians, and amplitude.
 *
 * @param components The vector of CleanDFT::Component objects to write.
 * @param N The size of the original FFT used to generate the components.
 * @param sample_rate The sample rate of the original audio signal (Hz).
 * @param filename The path to the output CSV file.
 */
void writeComponentsToCSV(const std::vector<CleanDFT::Component> &components,
                          const size_t N, // Original FFT size
                          const size_t sample_rate,
                          const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file " << filename << " for writing." << std::endl;
        return;
    }
    // Write CSV header
    file << "frequency_hz,phase,amplitude\n";

    // Iterate through each component
    for (const auto &comp: components) {
        // Convert component frequency from fractional bins (relative to N) to Hertz
        double freq_hz = comp.true_frequency * static_cast<double>(sample_rate) / static_cast<double>(N);
        // Write component data to the file
        file << freq_hz << ","
                << comp.true_phase << ","
                << comp.amplitude << "\n";
    }
    file.close(); // Close the file stream
    std::cout << "Components written to " << filename << std::endl;
}

/**
 * @brief Prints the command-line usage instructions for the program.
 *
 * @param programName The name of the executable (argv[0]).
 */
void printUsage(const char *programName) {
    std::cout << "Usage: " << programName << " [options]\n"
            << "Options:\n"
            << " -f, --file FILE            Input audio file path (required, .wav format)\n"
            << " -t, --threads NUM          Number of threads for parallel processing (default: hardware concurrency)\n"
            << " -s, --sequential           Use sequential processing (single thread, overrides -t)\n"
            << " -o, --output FILE          Output reconstructed audio file path (optional, .wav format)\n"
            << " -c, --csv FILE             Save extracted components to CSV file (optional)\n"
            << " -m, --metrics FILE         Save deconvolution accuracy metrics to CSV file (optional, uses metrics version)\n"
            << " -b, --benchmark            Output timing and component results in machine-readable format\n"
            << " -h, --help                 Show this help message and exit\n"
            << " -mc, --max-components NUM  Override maximum number of components to extract (default: set in code)\n"
            << " -ni, --non-iterative       Use non-iterative parallel deconvolution (faster, potentially less accurate phase)\n"
            << " -rs, --resample RATE       Resample output audio to RATE Hz (requires -o)\n";
}

/**
 * @brief Calculates the Log Spectral Distance (LSD) between two spectra.
 *
 * LSD measures the average squared difference between the log-magnitude spectra (in dB).
 * A lower value indicates greater similarity between the spectra.
 *
 * @param originalSpectrum The original complex spectrum.
 * @param reconstructedSpectrum The reconstructed complex spectrum.
 * @return The Log Spectral Distance in dB. Returns 0 if spectra are empty or sizes mismatch significantly.
 */
double calculateLSD(const std::vector<Complex> &originalSpectrum,
                    const std::vector<Complex> &reconstructedSpectrum) {
    // Use the minimum size to avoid out-of-bounds access, comparing only overlapping parts
    const size_t N = std::min(originalSpectrum.size(), reconstructedSpectrum.size());
    if (N <= 1) {
        // Need at least one bin beyond DC for meaningful comparison
        return 0.0;
    }

    double sum_squared_diff = 0.0;
    const double epsilon = 1e-10; // Small value to prevent log10(0)

    // Iterate through the bins (typically up to Nyquist, N/2, but here full spectrum for simplicity)
    // Start from k=0 (including DC) or k=1 (excluding DC). Excluding is common for LSD. Let's exclude.
    size_t count = 0;
    for (size_t k = 1; k < N / 2; k++) {
        // Iterate up to Nyquist bin
        // Calculate power (magnitude squared) and add epsilon
        double original_power = std::norm(originalSpectrum[k]) + epsilon;
        double reconstructed_power = std::norm(reconstructedSpectrum[k]) + epsilon;

        // Convert power to dB (10 * log10(power))
        double original_power_db = 10.0 * log10(original_power);
        double reconstructed_power_db = 10.0 * log10(reconstructed_power);

        // Calculate the squared difference in dB
        double diff = original_power_db - reconstructed_power_db;
        sum_squared_diff += diff * diff;
        count++;
    }

    if (count == 0) return 0.0; // Avoid division by zero if loop didn't run

    // Return the square root of the mean squared difference
    return sqrt(sum_squared_diff / count);
}

/**
 * @brief Main entry point for the CleanDFT deconvolution application.
 *
 * Parses command-line arguments, loads an audio file, performs FFT,
 * extracts components using CleanDFT (sequentially or in parallel),
 * optionally reconstructs the audio, saves components/metrics, and reports timing.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return 0 on success, 1 on error.
 */
int main(int argc, char *argv[]) {
    // --- Default Parameter Initialization ---
    size_t N = 0; // FFT size, determined after loading audio
    std::string input_file; // Path to input WAV file
    bool use_parallel = true; // Flag for parallel processing (default)
    uint32_t num_threads = std::thread::hardware_concurrency(); // Default thread count
    std::string output_file; // Path for reconstructed WAV file (optional)
    std::string csv_file; // Path for components CSV file (optional)
    bool benchmark_mode = false; // Flag for machine-readable output
    std::string metrics_file; // Path for metrics CSV file (optional)
    int maxcomp = MAX_COMPONENTS; // Max components to extract (use default from CleanDFT)
    bool iterative = true; // Use iterative refinement (default)
    bool resample = false; // Flag for resampling output
    size_t new_sample_rate = 0; // Target sample rate for resampling

    // --- Command Line Argument Parsing ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i]; // Current argument

        // Input file
        if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) {
                // Check if there is a value after the flag
                input_file = argv[++i]; // Assign value and increment index
            } else {
                std::cerr << "Error: Missing file path after " << arg << "\n";
                printUsage(argv[0]);
                return 1; // Exit with error
            }
        }
        // Number of threads
        else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                try {
                    num_threads = std::stoi(argv[++i]); // Convert string to int
                    if (num_threads <= 0) {
                        std::cerr << "Error: Number of threads must be positive\n";
                        return 1;
                    }
                    if (num_threads > std::thread::hardware_concurrency()) {
                        num_threads = std::thread::hardware_concurrency();
                        std::cout << "Input thread count exceeds available cores, using " << num_threads << " instead"
                                << "\n";
                    }
                    if (num_threads != 1) {
                        use_parallel = true;
                    }

                } catch (const std::invalid_argument &e) {
                    std::cerr << "Error: Invalid number provided for threads.\n";
                    return 1;
                } catch (const std::out_of_range &e) {
                    std::cerr << "Error: Number provided for threads is out of range.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: Missing number after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Sequential mode flag
        else if (arg == "-s" || arg == "--sequential") {
            use_parallel = false;
            num_threads = 1; // Force thread count to 1 for sequential
        }
        // Output file
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cerr << "Error: Missing output file path after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Components CSV file
        else if (arg == "-c" || arg == "--csv") {
            if (i + 1 < argc) {
                csv_file = argv[++i];
            } else {
                std::cerr << "Error: Missing CSV file path after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Benchmark mode flag
        else if (arg == "-b" || arg == "--benchmark") {
            benchmark_mode = true;
        }
        // Help flag
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0; // Exit successfully after showing help
        }
        // Metrics CSV file
        else if (arg == "-m" || arg == "--metrics") {
            if (i + 1 < argc) {
                metrics_file = argv[++i];
                // Ensure parallel metrics function is used even if user requested sequential
                if (!use_parallel) {
                    std::cout <<
                            "Warning: Metrics collection forces use of parallel function structure (with 1 thread if -s specified)."
                            << std::endl;
                    // The metrics function handles the single thread case internally
                    use_parallel = true; // Flag internally remains true to call the correct function
                    num_threads = 1; // But run it with only 1 thread if -s was set
                }
            } else {
                std::cerr << "Error: Missing metrics file path after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Max components override
        else if (arg == "-mc" || arg == "--max-components") {
            if (i + 1 < argc) {
                try {
                    maxcomp = std::stoi(argv[++i]);
                    if (maxcomp <= 0) {
                        std::cerr << "Error: Maximum components must be positive.\n";
                        return 1;
                    }
                    std::cout << "Max components overridden to: " << maxcomp << std::endl;
                } catch (const std::invalid_argument &e) {
                    std::cerr << "Error: Invalid number provided for max components.\n";
                    return 1;
                } catch (const std::out_of_range &e) {
                    std::cerr << "Error: Number provided for max components is out of range.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: Missing number after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Non-iterative flag
        else if (arg == "-ni" || arg == "--non-iterative") {
            iterative = false;
            if (!use_parallel && !benchmark_mode) {
                // Warn if trying sequential non-iterative (not implemented typically)
                std::cout <<
                        "Warning: Non-iterative mode usually requires parallel implementation. Forcing parallel structure."
                        << std::endl;
                use_parallel = true; // Force using the parallel non-iterative function
                num_threads = 1; // Run it with 1 thread if -s was also specified
            }
            if (!benchmark_mode) {
                std::cout << "Warning: Non-iterative mode selected. Phase accuracy might be lower." << std::endl;
            }
        }
        // Resample flag
        else if (arg == "-rs" || arg == "--resample") {
            if (i + 1 < argc) {
                try {
                    resample = true;
                    new_sample_rate = std::stoul(argv[++i]); // Use stoul for unsigned size_t
                    if (new_sample_rate <= 0) {
                        std::cerr << "Error: Resample rate must be positive.\n";
                        return 1;
                    }
                } catch (const std::invalid_argument &e) {
                    std::cerr << "Error: Invalid number provided for resample rate.\n";
                    return 1;
                } catch (const std::out_of_range &e) {
                    std::cerr << "Error: Number provided for resample rate is out of range.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: Missing sample rate after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Unknown option
        else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    } // End argument parsing loop

    // --- Input Validation ---
    // Check if input file was provided
    if (input_file.empty()) {
        std::cerr << "Error: Input file path is required (-f or --file).\n";
        printUsage(argv[0]);
        return 1;
    }
    // Check if resampling is requested without an output file
    if (resample && output_file.empty()) {
        std::cerr << "Error: Resampling (-rs) requires an output file (-o) to be specified.\n";
        printUsage(argv[0]);
        return 1;
    }
    // Check if metrics is requested with non-iterative (not compatible currently)
    if (!metrics_file.empty() && !iterative) {
        std::cerr <<
                "Error: Metrics collection (-m) is currently only implemented for the iterative deconvolution method.\n";
        return 1;
    }


    // --- Configuration Output (if not in benchmark mode) ---
    if (!benchmark_mode) {
        std::cout << "--- Configuration ---\n"
                << "Input file:           " << input_file << "\n"
                << "Processing mode:      " << (use_parallel ? "Parallel" : "Sequential") << "\n";
        if (use_parallel) {
            std::cout << "Threads:              " << num_threads << "\n";
        }
        std::cout << "Deconvolution mode:   " << (iterative ? "Iterative" : "Non-Iterative") << "\n";
        std::cout << "Max components:       " << maxcomp << "\n";
        if (!output_file.empty()) {
            std::cout << "Output file:          " << output_file << "\n";
            if (resample) {
                std::cout << "Resample output to:   " << new_sample_rate << " Hz\n";
            }
        }
        if (!csv_file.empty()) {
            std::cout << "Components CSV file:  " << csv_file << "\n";
        }
        if (!metrics_file.empty()) {
            std::cout << "Metrics CSV file:     " << metrics_file << "\n";
        }
        std::cout << "---------------------\n" << std::endl;
    }

    // --- Timing Setup ---
    auto wall_start = std::chrono::high_resolution_clock::now(); // Wall clock time start
    struct timespec cpu_start, cpu_end; // Structs for high-res CPU time
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start); // Get start CPU time for the process
    struct rusage start_usage, end_usage; // Structs for resource usage (user/system time)
    getrusage(RUSAGE_SELF, &start_usage); // Get start resource usage
    clock_t clock_start = clock(); // Start low-res user time clock

    // Per-thread timing data (optional, currently not populated in this example)
    // std::vector<double> thread_times;
    // std::atomic<int> thread_counter(0);
    // if (use_parallel) {
    //     thread_times.resize(num_threads, 0.0);
    // }


    try {
        // --- Load Audio File ---
        if (!benchmark_mode) std::cout << "Loading audio file..." << std::endl;
        // Use WaveProcessor to read the WAV file. Assuming it returns audio data and metadata.
        // Auto unpacks the returned tuple (C++17 structured binding)
        auto [leftChannel, rightChannel, sampleRate, channels] = WaveProcessor::readWav(input_file.c_str());

        if (leftChannel.empty()) {
            throw std::runtime_error("Failed to load audio data or audio file is empty.");
        }

        if (!benchmark_mode) {
            std::cout << "Loaded audio: "
                    << channels << " channels, "
                    << leftChannel.size() << " samples/channel at "
                    << sampleRate << " Hz\n";
            if (channels > 1) {
                std::cout << "Processing left channel only." << std::endl;
            }
        }

        // --- Compute FFT ---
        if (!benchmark_mode) std::cout << "Computing FFT..." << std::endl;
        // Compute the Fast Fourier Transform of the left channel
        std::vector<Complex> spectrum = WaveProcessor::computeFFT(leftChannel);
        N = spectrum.size(); // Store the FFT size
        if (N == 0) {
            throw std::runtime_error("FFT computation resulted in an empty spectrum.");
        }
        // Store the DC component (bin 0) separately, might be handled differently
        Complex DC = spectrum[0];


        // --- Initial Peak Counting (Optional Analysis) ---
        // Count local maxima in the magnitude spectrum up to Nyquist as a rough estimate
        int num_peaks = 0;
        if (N > 2) {
            // Need at least 3 points for peak definition (i-1, i, i+1)
            for (size_t i = 1; i < N / 2; ++i) {
                // Iterate up to Nyquist frequency
                // Check if magnitude at bin 'i' is greater than its neighbors
                if (std::abs(spectrum[i]) > std::abs(spectrum[i - 1]) &&
                    std::abs(spectrum[i]) > std::abs(spectrum[i + 1])) {
                    num_peaks++;
                }
            }
        }
        if (!benchmark_mode) {
            std::cout << "Initial FFT peaks found (up to Nyquist): " << num_peaks << std::endl;
        }


        // --- Deconvolve Spectrum into Components ---
        if (!benchmark_mode) std::cout << "Starting deconvolution..." << std::endl;
        std::vector<CleanDFT::Component> components; // Vector to store extracted components

        // Select the appropriate deconvolution function based on flags
        if (use_parallel) {
            if (!metrics_file.empty()) {
                // Use the parallel version that collects metrics
                if (!benchmark_mode) std::cout << "Using parallel deconvolution with metrics (" << num_threads <<
                                     " threads)..." << std::endl;
                components = CleanDFT::deconvolveParallelDirichletWithMetrics(
                    spectrum, sampleRate, num_threads, metrics_file, maxcomp);
            } else if (!iterative) {
                // Use the non-iterative parallel version
                if (!benchmark_mode) std::cout << "Using non-iterative parallel deconvolution (" << num_threads <<
                                     " threads)..." << std::endl;
                components = CleanDFT::deconvolveNonIterativeParallel(spectrum, sampleRate, num_threads, maxcomp);
                // Optional phase refinement step might be needed here
                // components = CleanDFT::refineComponentPhases(components, N);
            } else {
                // Use the standard iterative parallel version
                if (!benchmark_mode) std::cout << "Using iterative parallel deconvolution (" << num_threads <<
                                     " threads)..." << std::endl;
                components = CleanDFT::deconvolveParallelDirichlet(spectrum, sampleRate, num_threads, maxcomp);
            }
        } else {
            // Sequential processing
            if (!metrics_file.empty()) {
                // Metrics requested but sequential specified - run metrics version with 1 thread
                if (!benchmark_mode)
                    std::cout << "Using sequential deconvolution with metrics (via 1-thread parallel function)..." <<
                            std::endl;
                components = CleanDFT::deconvolveParallelDirichletWithMetrics(
                    spectrum, sampleRate, 1, metrics_file, maxcomp);
            } else if (!iterative) {
                // Non-iterative sequential (if implemented, otherwise falls back or errors)
                // Assuming non-iterative requires the parallel structure as noted before:
                if (!benchmark_mode)
                    std::cout << "Using non-iterative sequential deconvolution (via 1-thread parallel function)..." <<
                            std::endl;
                components = CleanDFT::deconvolveNonIterativeParallel(spectrum, sampleRate, 1, maxcomp);
            } else {
                // Use the standard sequential iterative version
                if (!benchmark_mode) std::cout << "Using sequential iterative deconvolution..." << std::endl;
                // Assuming a sequential function exists, e.g., deconvolveDirichletKernel
                components = CleanDFT::deconvolveDirichletKernel(spectrum, sampleRate, maxcomp);
                // Or run the parallel one with 1 thread if no dedicated sequential func:
                // components = CleanDFT::deconvolveParallelDirichlet(spectrum, sampleRate, 1, maxcomp);
            }
        }
        if (!benchmark_mode) {
            std::cout << "Deconvolution finished. Extracted " << components.size() << " components." << std::endl;
        }

        // --- Reconstruct Audio and Calculate LSD (if output file requested) ---
        // double lsd = 0.0; // Initialize Log Spectral Distance

        // The original DC component might be modified or recalculated during deconvolution/decompression.
        // Let's keep the original for comparison if needed, though reconstruction uses the potentially updated DC.
        // Complex oldDC = spectrum[0]; // Keep original DC if needed
        // DC might be recalculated based on components, or simply passed through.
        // DC = CleanDFT::computeDC(components, spectrum.size()); // Example if DC is recomputed
        DC = spectrum[0]; // Assume we pass the original DC for reconstruction for now

        if (!output_file.empty()) {
            if (!benchmark_mode) std::cout << "Reconstructing audio signal..." << std::endl;
            std::vector<double> reconstructed_signal; // Vector for the final time-domain signal
            size_t output_sample_rate = sampleRate; // Default to original sample rate

            // Perform resampling if requested
            if (resample) {
                if (!benchmark_mode) std::cout << "Resampling components to " << new_sample_rate << " Hz..." <<
                                     std::endl;
                reconstructed_signal = CleanDFT::resample(components, N, sampleRate, new_sample_rate, DC, 1.0,
                                                          use_parallel, num_threads);
                output_sample_rate = new_sample_rate; // Set the correct sample rate for writing the file
            }
            // Perform standard decompression (parallel or sequential)
            else if (use_parallel) {
                reconstructed_signal = CleanDFT::decompressComponentsParallel(components, N, DC, 1.0, num_threads);
            } else {
                reconstructed_signal = CleanDFT::decompressComponents(components, N, DC, 1.0);
            }

            if (reconstructed_signal.empty()) {
                throw std::runtime_error("Reconstruction resulted in an empty signal.");
            }

            // --- Calculate Log-Spectral Distance ---
            // if (!benchmark_mode) std::cout << "Calculating Log-Spectral Distance..." << std::endl;
            // Compute FFT of the reconstructed signal to compare spectra
            // Note: Reconstruction length might differ if resampling occurred. FFT must match.
            std::vector<Complex> reconstructed_spectrum = WaveProcessor::computeFFT(reconstructed_signal);
            // Ensure the original spectrum size matches the *intended* output length before resampling
            // If resampling, LSD comparison needs careful consideration (comparing different length/rate spectra?)
            // For now, we compare the original spectrum with the FFT of the potentially resampled reconstruction.
            // A more correct LSD might compare the original vs reconstruction *before* resampling,
            // or resample the *original* to the target rate for comparison.
            // Sticking to direct comparison for simplicity here:
            // lsd = calculateLSD(spectrum, reconstructed_spectrum);
            // if (!benchmark_mode) std::cout << "LSD calculated: " << lsd << " dB" << std::endl;


            // --- Write Output Audio File ---
            if (!benchmark_mode) std::cout << "Writing output audio file..." << std::endl;
            // Write the reconstructed signal to a WAV file with the appropriate sample rate
            WaveProcessor::writeWav(output_file.c_str(), reconstructed_signal, output_sample_rate);
            if (!benchmark_mode) std::cout << "Reconstructed audio written to " << output_file << std::endl;
        } // End if (!output_file.empty())

        // --- Save Components to CSV (if requested) ---
        if (!csv_file.empty()) {
            if (!benchmark_mode) std::cout << "Writing components to CSV..." << std::endl;
            // Write the extracted components (freq in Hz, phase, amp) to CSV
            writeComponentsToCSV(components, N, sampleRate, csv_file);
        }

        // --- Stop Timers ---
        clock_t clock_end = clock(); // Stop low-res clock
        getrusage(RUSAGE_SELF, &end_usage); // Get end resource usage
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end); // Get end CPU time
        auto wall_end = std::chrono::high_resolution_clock::now(); // Stop wall clock

        // --- Calculate Time Differences ---
        // High-resolution CPU time (includes time spent by all threads of the process)
        double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) +
                          (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e9;
        // Wall clock time (real-world time elapsed)
        double wall_time = std::chrono::duration<double>(wall_end - wall_start).count();
        // User CPU time (time spent executing user-level code)
        double user_time =
                (end_usage.ru_utime.tv_sec - start_usage.ru_utime.tv_sec) +
                (end_usage.ru_utime.tv_usec - start_usage.ru_utime.tv_usec) / 1e6;
        // System CPU time (time spent executing kernel-level code on behalf of the process)
        double system_time =
                (end_usage.ru_stime.tv_sec - start_usage.ru_stime.tv_sec) +
                (end_usage.ru_stime.tv_usec - start_usage.ru_stime.tv_usec) / 1e6;
        // Low-resolution user time (alternative measurement)
        double clock_time = (double) (clock_end - clock_start) / CLOCKS_PER_SEC;

        // Calculate total thread time (if measured, placeholder here)
        double total_thread_time = 0.0;
        // if (use_parallel) {
        //     for (double t : thread_times) { // Sum individual thread times if collected
        //         total_thread_time += t;
        //     }
        // }

        // --- Output Results ---
        if (benchmark_mode) {
            // Output results in a machine-readable format (key,value)
            std::cout << "INPUT_FILE," << input_file << std::endl;
            std::cout << "THREADS," << num_threads << std::endl;
            std::cout << "PARALLEL," << (use_parallel ? "1" : "0") << std::endl;
            std::cout << "ITERATIVE," << (iterative ? "1" : "0") << std::endl;
            std::cout << "MAX_COMPONENTS," << maxcomp << std::endl;
            std::cout << "WALL_TIME," << wall_time << std::endl;
            std::cout << "CPU_TIME," << cpu_time << std::endl; // Total CPU time for process
            std::cout << "USER_TIME," << user_time << std::endl; // User mode CPU time
            std::cout << "SYSTEM_TIME," << system_time << std::endl; // Kernel mode CPU time
            std::cout << "CLOCK_TIME," << clock_time << std::endl; // Low-res user time
            // std::cout << "TOTAL_THREAD_TIME," << total_thread_time << std::endl; // If measured
            std::cout << "NUM_PEAKS," << num_peaks << std::endl; // Initial FFT peaks
            std::cout << "NUM_COMPONENTS," << components.size() << std::endl; // Extracted components
            std::cout << "SAMPLE_RATE," << sampleRate << std::endl;
            std::cout << "NUM_SAMPLES," << leftChannel.size() << std::endl;
            std::cout << "DURATION," << static_cast<double>(leftChannel.size()) / sampleRate << std::endl;
            // if (!output_file.empty()) {
            //     std::cout << "LOG_SPECTRAL_DISTANCE," << lsd << std::endl;
            // }
            if (resample) {
                std::cout << "RESAMPLED_TO_HZ," << new_sample_rate << std::endl;
            }
        } else {
            // Output results in a user-friendly format
            std::cout << "\n--- Processing Summary ---\n";
            std::cout << "Wall clock time:      " << wall_time << " seconds\n";
            std::cout << "Total CPU time:       " << cpu_time << " seconds\n";
            std::cout << "User CPU time:        " << user_time << " seconds\n";
            std::cout << "System CPU time:      " << system_time << " seconds\n";
            // std::cout << "Clock() time:         " << clock_time << " seconds\n"; // Can be less precise
            if (use_parallel && wall_time > 1e-9) {
                // Avoid division by zero
                // Speedup: How much faster than sequential user time (ideally user_time / wall_time)
                // Efficiency: How well the parallel resources were used ((user_time / wall_time) / num_threads)
                double speedup = user_time / wall_time;
                double efficiency = (speedup / num_threads) * 100.0;
                std::cout << "Parallel Speedup:     " << speedup << "x (approx, vs sequential user time)\n";
                std::cout << "Parallel Efficiency:  " << efficiency << "% (approx)\n";
            }
            std::cout << "Initial spectral peaks: " << num_peaks << "\n";
            std::cout << "Extracted components: " << components.size() << "\n";
            if (num_peaks > 0 && !components.empty()) {
                // Compression ratio: How many original peaks are represented by one component
                double ratio = static_cast<double>(num_peaks) / components.size();
                // Or maybe FFT bins / components? N / 2 / components.size()
                double bin_ratio = static_cast<double>(N / 2) / components.size();
                std::cout << "Component Density:    1 component per ~" << bin_ratio << " Nyquist bins\n";
            }
            // if (!output_file.empty()) {
            //     std::cout << "Log-Spectral Distance: " << lsd << " dB\n";
            // }
            std::cout << "------------------------\n" << std::endl;
        }
    } catch (const std::exception &e) {
        // Catch standard exceptions and report the error
        std::cerr << "Error: " << e.what() << std::endl;
        // Consider ending timers here as well if needed
        return 1; // Return error code
    }

    // Return success code
    return 0;
}
