#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <time.h>
#include <sys/resource.h>
#include <string>
#include "fft/waveprocessor.h"
#include "deconvolve/cleandft.h"

void writeComponentsToCSV(const std::vector<CleanDFT::Component> &components,
                          const size_t N,
                          const size_t sample_rate,
                          const std::string &filename) {
    std::ofstream file(filename);
    file << "frequency_hz,phase,amplitude\n"; // CSV header

    for (const auto &comp: components) {
        // Convert frequency from bins to Hz
        double freq_hz = comp.true_frequency * sample_rate / N;
        file << freq_hz << ","
                << comp.true_phase << ","
                << comp.amplitude << "\n";
    }
    file.close();
}

void printUsage(const char *programName) {
    std::cout << "Usage: " << programName << " [options]\n"
            << "Options:\n"
            << " -f, --file FILE       Input audio file path (required)\n"
            << " -t, --threads NUM     Number of threads (default: auto)\n"
            << " -s, --sequential      Use sequential processing (default: parallel)\n"
            << " -o, --output FILE     Output reconstructed file (optional)\n"
            << " -c, --csv FILE        Save components to CSV file (optional)\n"
            << " -m, --metrics FILE    Save accuracy metrics to CSV file (optional)\n"
            << " -b, --benchmark       Output only benchmark results (for automation)\n"
            << " -h, --help            Show this help\n"
            << " -mc, --max-components Overrides maximum numer of components\n";
}

// Log Spectral Distance calculation
double calculateLSD(const std::vector<Complex> &originalSpectrum,
                    const std::vector<Complex> &reconstructedSpectrum) {
    const size_t N = std::min(originalSpectrum.size(), reconstructedSpectrum.size());
    double sum_squared_diff = 0.0;

    // Start from bin 1 to avoid DC component (can be tricky with log)
    for (size_t k = 0; k < N; k++) {
        double original_power_db = 10.0 * log10(std::norm(originalSpectrum[k]) + 1e-10);
        double reconstructed_power_db = 10.0 * log10(std::norm(reconstructedSpectrum[k]) + 1e-10);

        double diff = original_power_db - reconstructed_power_db;
        sum_squared_diff += diff * diff;
    }

    return sqrt(sum_squared_diff / (N - 1)); // N-1 because we skipped DC
}

int main(int argc, char *argv[]) {
    // Default parameters
    std::string input_file;
    bool use_parallel = true;
    int num_threads = std::thread::hardware_concurrency();
    std::string output_file;
    std::string csv_file;
    bool benchmark_mode = false;
    std::string metrics_file;
    int maxcomp = MAX_COMPONENTS;


    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) {
                input_file = argv[++i];
            } else {
                std::cerr << "Error: Missing file path after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
                if (num_threads <= 0) {
                    std::cerr << "Error: Number of threads must be positive\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: Missing number after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-s" || arg == "--sequential") {
            use_parallel = false;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cerr << "Error: Missing output file path after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-c" || arg == "--csv") {
            if (i + 1 < argc) {
                csv_file = argv[++i];
            } else {
                std::cerr << "Error: Missing CSV file path after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-b" || arg == "--benchmark") {
            benchmark_mode = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--metrics") {
            if (i + 1 < argc) {
                metrics_file = argv[++i];
            } else {
                std::cerr << "Error: Missing metrics file path after " << arg << "\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-mc" || arg == "--max-components") {
            maxcomp = std::stoi(argv[++i]);
            std::cout << MAX_COMPONENTS << "\n";
        } else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }


    // Check if input file is provided
    if (input_file.empty()) {
        std::cerr << "Error: Input file is required\n";
        printUsage(argv[0]);
        return 1;
    }

    // Output configuration if not in benchmark mode
    if (!benchmark_mode) {
        std::cout << "Configuration:\n"
                << "  Input file: " << input_file << "\n"
                << "  Processing mode: " << (use_parallel ? "Parallel" : "Sequential") << "\n"
                << "  Threads: " << num_threads << "\n";
        if (!output_file.empty()) {
            std::cout << "  Output file: " << output_file << "\n";
        }
        if (!csv_file.empty()) {
            std::cout << "  CSV file: " << csv_file << "\n";
        }
        std::cout << std::endl;
    }

    // Start wall clock timing
    auto wall_start = std::chrono::high_resolution_clock::now();

    // Start CPU timing
    struct timespec cpu_start, cpu_end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

    // Start user time measurement
    struct rusage start_usage, end_usage;
    getrusage(RUSAGE_SELF, &start_usage);

    // Start clock() timing (another way to measure user time)
    clock_t clock_start = clock();

    // Per-thread timing data
    std::vector<double> thread_times;
    std::atomic<int> thread_counter(0);
    if (use_parallel) {
        thread_times.resize(num_threads, 0.0);
    }

    try {
        // Load audio file
        auto [leftChannel, rightChannel, sampleRate, channels] = WaveProcessor::readWav(input_file.c_str());

        if (!benchmark_mode) {
            std::cout << "Loaded audio: "
                    << channels << " channels, "
                    << leftChannel.size() << " samples at "
                    << sampleRate << " Hz\n";
        }

        // Compute FFT for left channel
        std::vector<Complex> spectrum = WaveProcessor::computeFFT(leftChannel);;

        // Create normalized copy
        // std::vector<Complex> normalized_spectrum(spectrum.size());
        //
        // // Find maximum magnitude in the spectrum
        // double max_magnitude = 0.0;
        // for (const auto& val : spectrum) {
        //     max_magnitude = std::max(max_magnitude, std::abs(val));
        // }
        //
        // // Normalize by dividing each value by the maximum magnitude
        // if (max_magnitude > 0.0) {  // Avoid division by zero
        //     for (size_t i = 0; i < spectrum.size(); i++) {
        //         normalized_spectrum[i] = spectrum[i] / max_magnitude;
        //     }
        // } else {
        //     normalized_spectrum = spectrum; // Just copy if all zeros
        // }

        // spectrum = normalized_spectrum;


        Complex DC = spectrum[0];


        // Count local maxima (peaks) in spectrum
        int num_peaks = 0;
        for (int i = 1; i < spectrum.size() - 1; ++i) {
            if (std::abs(spectrum[i]) > std::abs(spectrum[i - 1]) &&
                std::abs(spectrum[i]) > std::abs(spectrum[i + 1])) {
                num_peaks++;
            }
        }

        // Adjust to count only up to Nyquist frequency
        num_peaks = num_peaks / 2;

        // Decompose spectrum into components
        std::vector<CleanDFT::Component> components;

        if (use_parallel) {
            if (!benchmark_mode) {
                std::cout << "Using parallel decomposition with " << num_threads << " threads" << std::endl;
            }
            if (!metrics_file.empty()) {
                // Use the version that collects metrics
                components = CleanDFT::deconvolveParallelDirichletWithMetrics(
                    spectrum, sampleRate, num_threads, metrics_file, maxcomp);
            } else {
                // Use the regular version
                // components = CleanDFT::deconvolveParallelDirichlet(spectrum, sampleRate, num_threads, maxcomp);
                components = CleanDFT::deconvolveNonIterativeParallel(spectrum, sampleRate, num_threads, maxcomp);
            }

            // filter out noisy components
            // std::vector<CleanDFT::Component> new_components;
            //
            // for (auto &component : components) {
            //     if (!(component.amplitude > 0.0f && component.amplitude < 1e-7f)) {
            //         // remove
            //         new_components.push_back(component);
            //     }
            // }
            // components = new_components;
        } else {
            if (!benchmark_mode) {
                std::cout << "Using sequential decomposition" << std::endl;
            }
            // if (!metrics_file.empty()) {
            //     // Use the version that collects metrics
            //     components = CleanDFT::deconvolveParallelDirichletWithMetrics(spectrum, sampleRate, 1, metrics_file, maxcomp);
            // } else {
                components = CleanDFT::deconvolveDirichletKernel(spectrum, sampleRate, maxcomp);
            // }
        }

        // Calculate LSD if output file is requested
        double lsd = 0.0;
        Complex oldDC = DC;




        DC = CleanDFT::computeDC(components, spectrum.size());
        if (!output_file.empty()) {
            // Generate reconstructed output
            std::vector<double> reconstructed;
            std::vector<Complex> reconstructed_spectrum;


            // reconstructed = CleanDFT::decompressComponentsEnhanced(components, spectrum.size(), DC, 1);
            if (use_parallel) {
                reconstructed = CleanDFT::decompressComponentsParallel(components, spectrum.size(), DC, 1);
            } else {
                reconstructed = CleanDFT::decompressComponents(components, spectrum.size(), DC, 1);
            }


            // Calculate reconstructed spectrum for LSD calculation

            // reassign DC


            reconstructed_spectrum = WaveProcessor::computeFFT(reconstructed);

            // Calculate Log Spectral Distance
            lsd = calculateLSD(spectrum, reconstructed_spectrum);

            // Write output file
            WaveProcessor::writeWav(output_file.c_str(), reconstructed, sampleRate);
        }

        // Save components to CSV if requested
        if (!csv_file.empty()) {
            writeComponentsToCSV(components, spectrum.size(), sampleRate, csv_file);
        }

        // End all timing measurements
        clock_t clock_end = clock();
        getrusage(RUSAGE_SELF, &end_usage);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
        auto wall_end = std::chrono::high_resolution_clock::now();

        // Calculate time differences
        double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) +
                          (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e9;

        double wall_time = std::chrono::duration<double>(wall_end - wall_start).count();

        double user_time =
                (end_usage.ru_utime.tv_sec - start_usage.ru_utime.tv_sec) +
                (end_usage.ru_utime.tv_usec - start_usage.ru_utime.tv_usec) / 1e6;

        double system_time =
                (end_usage.ru_stime.tv_sec - start_usage.ru_stime.tv_sec) +
                (end_usage.ru_stime.tv_usec - start_usage.ru_stime.tv_usec) / 1e6;

        double clock_time = (double) (clock_end - clock_start) / CLOCKS_PER_SEC;

        // Calculate total thread time (for parallel processing)
        double total_thread_time = 0.0;
        if (use_parallel) {
            for (double t: thread_times) {
                total_thread_time += t;
            }
        }

        std::cout << "Old DC: " << oldDC << " New DC: " << DC << std::endl;

        // Output benchmark results
        if (benchmark_mode) {
            // Output in a machine-readable format for the Python script
            std::cout << "INPUT_FILE," << input_file << std::endl;
            std::cout << "THREADS," << num_threads << std::endl;
            std::cout << "PARALLEL," << (use_parallel ? "1" : "0") << std::endl;
            std::cout << "WALL_TIME," << wall_time << std::endl;
            std::cout << "CPU_TIME," << cpu_time << std::endl;
            std::cout << "USER_TIME," << user_time << std::endl;
            std::cout << "SYSTEM_TIME," << system_time << std::endl;
            std::cout << "CLOCK_TIME," << clock_time << std::endl;
            std::cout << "TOTAL_THREAD_TIME," << total_thread_time << std::endl;
            std::cout << "NUM_PEAKS," << num_peaks << std::endl;
            std::cout << "NUM_COMPONENTS," << components.size() << std::endl;
            std::cout << "SAMPLE_RATE," << sampleRate << std::endl;
            std::cout << "DURATION," << static_cast<double>(leftChannel.size()) / sampleRate << std::endl;
            if (!output_file.empty()) {
                std::cout << "LOG_SPECTRAL_DISTANCE," << lsd << std::endl;
            }
        } else {
            // User-friendly output
            std::cout << "Processing completed:\n";
            std::cout << "  Wall clock time: " << wall_time << " seconds\n";
            std::cout << "  CPU time: " << cpu_time << " seconds\n";
            std::cout << "  User time: " << user_time << " seconds\n";
            std::cout << "  System time: " << system_time << " seconds\n";
            std::cout << "  Clock time: " << clock_time << " seconds\n";
            if (use_parallel) {
                std::cout << "  Total thread time: " << total_thread_time << " seconds\n";
                std::cout << "  Speedup: " << (user_time / wall_time) << "x\n";
                std::cout << "  Efficiency: " << (user_time / wall_time / num_threads * 100) << "%\n";
            }
            std::cout << "  Spectral peaks: " << num_peaks << "\n";
            std::cout << "  Extracted components: " << components.size() << "\n";
            std::cout << "  Compression ratio: " << static_cast<double>(num_peaks) / components.size() << ":1\n";
            if (!output_file.empty()) {
                std::cout << "  Log Spectral Distance: " << lsd << " dB\n";
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
