/**
* @file cleandft.cpp
* @author Matthew McQuistion
* @date 4/25/25
* @brief Implementation of the Dirichlet Kernel Deconvolution (DKD) algorithm
*
* This file contains the core implementation of the DKD algorithm, including both
* single-threaded and multi-threaded versions. The DKD algorithm extracts precise
* frequency components from discrete spectra by modeling and removing spectral leakage.
*/

#include "cleandft.h"
#include "threadpool.h"
#include <fstream>
#include <iostream>
#include <cmath>

CleanDFT::CleanDFT() = default;

/**
* @brief Computes the DC component from extracted components
*
* This function calculates how each extracted component contributes to the DC bin
* (zero frequency) of the spectrum, accounting for the Dirichlet kernel response.
*
* @param components Vector of extracted frequency components
* @param N FFT size
* @return Complex DC value with only real part (imaginary forced to 0)
*/
Complex CleanDFT::computeDC(const std::vector<Component> &components, const size_t N) {
    Complex new_dc = 0;

    // Iterate through all components to compute their DC contribution
    for (const auto &[true_frequency, true_phase, amplitude]: components) {
        // Direct contribution to DC (negative frequency due to bin 0)
        const double diff = -true_frequency;
        const double phase = true_phase + M_PI * diff / static_cast<double>(N);
        new_dc += amplitude * DirichletKernel::getValueAtBin(diff, phase, N);

        // Mirror frequency contribution (from conjugate symmetry)
        const double mirror_diff = -(N - true_frequency);
        const double mirror_phase = true_phase + M_PI * mirror_diff / static_cast<double>(N);

        // Add complex conjugate of mirror response
        new_dc += std::conj(amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
    }

    // Return only the real part (DC should be purely real)
    Complex ret = Complex(new_dc.real(), 0);
    return ret;
}

/**
* @brief Calculates spectral distance between two peaks
*
* Used to ensure adequate separation when selecting peaks for parallel processing
* to prevent interference between simultaneously processed components.
*
* @param peak1 First peak information
* @param peak2 Second peak information
* @param N FFT size for normalization
* @return Normalized spectral distance (0 to 1)
*/
double CleanDFT::spectralDistance(const PeakInfo &peak1, const PeakInfo &peak2, size_t N) {
    // Simple Euclidean distance in frequency space
    double bin_distance = std::abs(peak1.frequency - peak2.frequency);

    // Normalize by FFT size to get relative distance
    return bin_distance / N;
}

/**
* @brief Computes correlation between spectral data and Dirichlet kernel
*
* Core function used in optimization to measure how well a Dirichlet kernel
* with given parameters matches the observed spectrum. Used differently for
* frequency and phase optimization.
*
* @param data_slice Local spectrum around the peak
* @param kernel Dirichlet kernel values for comparison
* @param isPhase True for phase correlation, false for frequency correlation
* @return Normalized correlation coefficient
*/
double CleanDFT::computeCorrelation(const std::vector<Complex> &data_slice,
                                    const std::vector<Complex> &kernel,
                                    const bool isPhase) {
    Complex numerator = 0;

    // Compute dot product between data and kernel
    for (size_t i = 0; i < data_slice.size(); i++) {
        numerator += data_slice[i] * std::conj(kernel[i]);
    }

    // Compute norms for normalization
    double data_norm = 0;
    double kernel_norm = 0;
    for (size_t i = 0; i < data_slice.size(); i++) {
        data_norm += std::abs(data_slice[i]) * std::abs(data_slice[i]);
        kernel_norm += std::abs(kernel[i]) * std::abs(kernel[i]);
    }

    if (isPhase) {
        // For phase correlation, use real part to distinguish between phases
        return std::real(numerator) / (std::sqrt(data_norm) * std::sqrt(kernel_norm));
    }
    // For frequency correlation, use magnitude
    return std::abs(numerator) / (std::sqrt(data_norm) * std::sqrt(kernel_norm));
}

/**
* @brief Finds optimal phase for a frequency component
*
* Uses golden section search to maximize correlation between the Dirichlet
* kernel and the observed spectrum, determining the true phase.
*
* @param fft Full spectrum
* @param center_bin Bin index of the peak
* @param frequency True frequency (already optimized)
* @return Optimal phase in radians
*/
double CleanDFT::findOptimalPhase(const std::vector<Complex> &fft, const int center_bin, const double frequency) {
    // Create window of bins around the center
    std::vector<int> bins;
    for (int i = center_bin - PHASE_WINDOW_SIZE; i <= center_bin + PHASE_WINDOW_SIZE; i++) {
        if (i >= 0 && i < fft.size()) {
            bins.push_back(i);
        }
    }

    // Extract spectrum values in the window
    std::vector<Complex> data_slice;
    data_slice.reserve(bins.size());
    for (const int bin: bins) {
        data_slice.push_back(fft[bin]);
    }

    // Define objective function for optimization
    auto evaluate = [&](const double phase) {
        const std::vector<Complex> kernel = DirichletKernel::generateKernel(
            bins, frequency, phase, fft.size(), std::abs(fft[center_bin])
        );
        return computeCorrelation(data_slice, kernel, true);
    };

    // Search over the full phase range [-π, π]
    return goldenSectionSearch(evaluate, -1.0 * M_PI, 1.0 * M_PI);
}

/**
* @brief Finds optimal frequency for a peak
*
* Uses golden section search to find the precise frequency that maximizes
* correlation with the observed spectrum, achieving sub-bin precision.
*
* @param fft Full spectrum
* @param center_bin Initial bin estimate
* @param test_frequency Initial frequency guess
* @return Optimized frequency in fractional bins
*/
double CleanDFT::findOptimalFrequency(const std::vector<Complex> &fft, const int center_bin,
                                      const double test_frequency) {
    // Create window of bins around the center
    std::vector<int> bins;
    for (int i = center_bin - FREQUENCY_WINDOW_SIZE; i <= center_bin + FREQUENCY_WINDOW_SIZE; i++) {
        if (i >= 0 && i < fft.size()) {
            bins.push_back(i);
        }
    }

    // Extract spectrum values in the window
    std::vector<Complex> data_slice;
    data_slice.reserve(bins.size());
    for (const int bin: bins) {
        data_slice.push_back(fft[bin]);
    }

    // Use phase from peak bin as initial estimate
    const double center_phase = std::arg(fft[center_bin]);

    // Define objective function for optimization
    auto evaluate = [&](const double freq) {
        const std::vector<Complex> kernel = DirichletKernel::generateKernel(
            bins, freq, center_phase, fft.size(), std::abs(fft[center_bin])
        );
        return computeCorrelation(data_slice, kernel, false);
    };

    // Search ±0.5 bins around the test frequency
    return goldenSectionSearch(evaluate, test_frequency - 0.5, test_frequency + 0.5);
}

/**
* @brief Single-threaded DKD implementation
*
* Iteratively extracts frequency components from the spectrum by:
* 1. Finding peaks in the residual
* 2. Optimizing frequency and phase
* 3. Subtracting the component's Dirichlet kernel response
*
* @param fft Input spectrum
* @param sample_rate Sampling frequency
* @param maxcomp Maximum number of components to extract
* @return Vector of extracted components
*/
std::vector<CleanDFT::Component> CleanDFT::deconvolveDirichletKernel(const std::vector<Complex> &fft,
                                                                     const size_t sample_rate, const int maxcomp) {
    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;
    MAX_COMPONENTS = maxcomp;
    std::vector<Component> components;
    std::vector<Complex> residual(nyquist_bin);

    // Initialize residual with first half of spectrum
    std::copy_n(fft.begin(), nyquist_bin, residual.begin());
    residual[0] *= 0.0; // Remove DC component

    // Calculate total energy for convergence checking
    double total_energy = 0.0;
    for (const auto &val: residual) {
        total_energy += std::norm(val);
    }

    std::cout << "Total energy: " << total_energy << std::endl;

    // Main extraction loop
    while (components.size() < MAX_COMPONENTS) {
        // Find peak bin in residual
        int peak_bin = 0;
        double max_magnitude = 0.0;

        for (int i = 1; i < nyquist_bin; i++) {
            const double magnitude = std::abs(residual[i]);
            if (magnitude > max_magnitude && i < static_cast<double>(nyquist_bin) * 0.99) {
                max_magnitude = magnitude;
                peak_bin = i;
            }
        }

        // Check termination conditions
        if (static_cast<double>(peak_bin) >= static_cast<double>(nyquist_bin) * 0.99 || max_magnitude < 1e-6) {
            std::cout << "Peak bin too high or magnitude too low, stopping..." << std::endl;
            std::cout << peak_bin << " " << peak_bin * sample_rate / N << std::endl;
            break;
        }

        // Optimize frequency and phase
        const double true_freq = findOptimalFrequency(residual, peak_bin, peak_bin);
        const double true_phase = findOptimalPhase(residual, peak_bin, true_freq);

        // Calculate true amplitude
        const double amplitude = DirichletKernel::getAmplitudeAtBin(true_freq, std::abs(residual[peak_bin]), N,
                                                                    peak_bin);

        components.push_back({true_freq, true_phase, amplitude});

        // Update residual by subtracting the component's response
        for (size_t i = 1; i < N / 2; i++) {
            // Direct contribution
            const double diff = true_freq - static_cast<double>(i);
            const double corrected_phase = true_phase + M_PI * diff / static_cast<double>(N);
            residual[i] -= amplitude * DirichletKernel::getValueAtBin(diff, corrected_phase, N);

            // Mirror frequency contribution
            const size_t mirror_bin = N - i;
            const double mirror_diff = true_freq - static_cast<double>(mirror_bin);
            const double mirror_phase = true_phase + M_PI * mirror_diff / static_cast<double>(N);

            // Subtract complex conjugate for mirror response
            residual[i] -= std::conj(amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
        }

        // Calculate residual energy and check convergence
        double residual_energy = 0.0;
        for (const auto &val: residual) {
            residual_energy += std::norm(val);
        }

        // Progress reporting
        std::cout << "Component " << components.size() << ": true bin="
                << true_freq << ": old bin=" << peak_bin * sample_rate / N
                << ", phase=" << true_phase << std::endl;

        const double retention = residual_energy / total_energy;
        std::cout << "% L2 norm retained: " << retention << std::endl;

        // Check convergence threshold
        if (constexpr double THRESHOLD = 1e-6; residual_energy < THRESHOLD * total_energy) {
            break;
        }
    }

    return components;
}

/**
* @brief Writes spectrum data to CSV file for analysis
*
* Helper function for debugging that saves complex spectrum data
* to a CSV file with magnitude, phase, real, and imaginary components.
*
* @param spectrum Complex spectrum to save
* @param filename Output CSV filename
*/
void writeSpectrumToCSV(const std::vector<Complex> &spectrum, const std::string &filename) {
    std::ofstream file(filename);
    file << "bin,magnitude,phase,real,imag\n";

    for (size_t i = 0; i < spectrum.size(); i++) {
        file << i << ","
                << std::abs(spectrum[i]) << ","
                << std::arg(spectrum[i]) << ","
                << spectrum[i].real() << ","
                << spectrum[i].imag() << "\n";
    }
}

/**
* @brief Thread worker function for finding peaks in a frequency band
*
* Identifies local maxima within a specific frequency band and calculates
* spectral centroids for more accurate peak positioning.
*
* @param residual Current residual spectrum
* @param band_peaks Output vector for storing peaks by band
* @param band_index Index of this band
* @param band Frequency band boundaries
* @param N FFT size
*/
void findPeaksInBand(
    const std::vector<Complex> &residual,
    std::vector<std::vector<PeakInfo> > &band_peaks,
    size_t band_index,
    const FrequencyBand &band,
    size_t N
) {
    std::vector<PeakInfo> peaks;

    // Find local maxima in this band
    for (size_t i = band.start_bin + 1; i < band.end_bin - 1; i++) {
        double magnitude = std::abs(residual[i]);

        // Check if this is a local maximum
        if (magnitude > std::abs(residual[i - 1]) &&
            magnitude > std::abs(residual[i + 1]) &&
            magnitude > 1e-10) {
            // Calculate spectral centroid for better peak positioning
            double weighted_sum = 0;
            double sum_weights = 0;

            // Use ±5 bins around peak for centroid calculation
            for (int j = std::max(static_cast<int>(i) - 5, 0);
                 j <= std::min(static_cast<int>(i) + 5, static_cast<int>(band.end_bin));
                 j++) {
                double weight = std::abs(residual[j]);
                weighted_sum += j * weight;
                sum_weights += weight;
            }

            double centroid = sum_weights > 0 ? weighted_sum / sum_weights : i;

            // Create peak info structure
            PeakInfo peak;
            peak.bin = i;
            peak.magnitude = magnitude;
            peak.frequency = i; // Initial guess
            peak.phase = std::arg(residual[i]); // Initial guess
            peak.amplitude = magnitude; // Initial guess
            peak.spectral_centroid = centroid;
            peaks.push_back(peak);
        }
    }

    // Sort peaks by magnitude (descending) for processing priority
    std::sort(peaks.begin(), peaks.end(),
              [](const PeakInfo &a, const PeakInfo &b) {
                  return a.magnitude > b.magnitude;
              }
    );

    // Store results (thread-safe as each band has separate index)
    band_peaks[band_index] = std::move(peaks);
}

/**
* @brief Thread worker function for optimizing a peak's parameters
*
* Finds the optimal frequency, phase, and amplitude for a detected peak
* using golden section search optimization.
*
* @param residual Current residual spectrum
* @param batch_components Output vector for storing optimized components
* @param selected_peaks Vector of peaks to optimize
* @param peak_index Index of the peak to process
* @param N FFT size
*/
void optimizePeak(
    const std::vector<Complex> &residual,
    std::vector<CleanDFT::Component> &batch_components,
    const std::vector<PeakInfo> &selected_peaks,
    size_t peak_index,
    size_t N
) {
    const PeakInfo &peak = selected_peaks[peak_index];

    // Find optimal frequency using golden section search
    double true_freq = CleanDFT::findOptimalFrequency(residual, peak.bin, peak.bin);

    // Find optimal phase
    double true_phase = CleanDFT::findOptimalPhase(residual, peak.bin, true_freq);

    // Compute corrected amplitude
    double amplitude = DirichletKernel::getAmplitudeAtBin(
        true_freq, std::abs(residual[peak.bin]), N, peak.bin);

    // Store result (thread-safe as each peak has separate index)
    batch_components[peak_index] = CleanDFT::Component{true_freq, true_phase, amplitude};
}

/**
* @brief Updates a portion of the residual spectrum
*
* Subtracts a component's Dirichlet kernel response from a specific
* frequency range of the residual. Used for parallel residual updates.
*
* @param residual Residual spectrum to update
* @param component Component to subtract
* @param start_bin Starting bin index
* @param end_bin Ending bin index
* @param N FFT size
*/
void updateResidualRange(
    std::vector<Complex> &residual,
    const CleanDFT::Component &component,
    size_t start_bin,
    size_t end_bin,
    size_t N
) {
    for (size_t i = start_bin; i < end_bin; i++) {
        // Direct contribution to bin
        double diff = component.true_frequency - static_cast<double>(i);
        double corrected_phase = component.true_phase + M_PI * diff / static_cast<double>(N);
        residual[i] -= component.amplitude * DirichletKernel::getValueAtBin(diff, corrected_phase, N);

        // Mirror frequency contribution
        const size_t mirror_bin = N - i;
        const double mirror_diff = component.true_frequency - static_cast<double>(mirror_bin);
        const double mirror_phase = component.true_phase + M_PI * mirror_diff / static_cast<double>(N);

        // Subtract complex conjugate for mirror response
        residual[i] -= std::conj(component.amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
    }
}

/**
* @brief Validates frequency band division
*
* Checks that frequency bands cover the entire spectrum without gaps
* or excessive overlap. Used for debugging spectral division methods.
*
* @param bands Vector of frequency bands
* @param nyquist_bin Nyquist frequency bin index
* @return True if bands are valid, false otherwise
*/
bool validateBands(const std::vector<FrequencyBand> &bands, size_t nyquist_bin) {
    // Track bin coverage
    std::vector<bool> bin_covered(nyquist_bin, false);
    std::vector<bool> bin_overlap(nyquist_bin, false);

    // Check each band
    for (const auto &band: bands) {
        for (size_t bin = band.start_bin; bin < band.end_bin; bin++) {
            if (bin >= nyquist_bin) {
                std::cerr << "Error: Band exceeds Nyquist bin" << std::endl;
                return false;
            }

            if (bin_covered[bin]) {
                bin_overlap[bin] = true;
                std::cerr << "Warning: Bin " << bin << " is covered by multiple bands" << std::endl;
            }
            bin_covered[bin] = true;
        }
    }

    // Check for gaps
    for (size_t bin = 1; bin < nyquist_bin; bin++) {
        if (!bin_covered[bin]) {
            std::cerr << "Error: Bin " << bin << " is not covered by any band" << std::endl;
            return false;
        }
    }

    // Check for overlaps
    bool has_overlap = false;
    for (size_t bin = 1; bin < nyquist_bin; bin++) {
        if (bin_overlap[bin]) {
            has_overlap = true;
        }
    }

    if (has_overlap) {
        std::cerr << "Warning: Band overlap detected" << std::endl;
    }

    return true;
}

/**
* @brief Main parallel deconvolution function
*
* Multi-threaded implementation of DKD that:
* 1. Divides spectrum into frequency bands
* 2. Finds peaks in parallel across bands
* 3. Optimizes batches of well-separated peaks concurrently
* 4. Updates residual in parallel
*
* @param fft Input spectrum
* @param sample_rate Sampling frequency
* @param num_threads Number of threads to use
* @param maxcomp Maximum number of components to extract
* @return Vector of extracted components
*/
std::vector<CleanDFT::Component> CleanDFT::deconvolveParallelDirichlet(
    const std::vector<Complex> &fft, const size_t sample_rate, const size_t num_threads, const int maxcomp) {
    MAX_COMPONENTS = maxcomp;

    // Adjust batch size if needed
    if (maxcomp < BATCH_SIZE) {
        BATCH_SIZE = MAX_COMPONENTS;
    }

    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;

    // Set up frequency bands for parallel processing
    std::vector<FrequencyBand> bands;
    const size_t num_bands = num_threads;

    // Default to uniform spectral division
    bands.resize(num_threads);
    const size_t bins_per_band = nyquist_bin / num_threads;
    for (size_t i = 0; i < num_threads; i++) {
        bands[i].start_bin = i * bins_per_band + 1; // Skip DC
        bands[i].end_bin = (i == num_threads - 1) ? nyquist_bin : (i + 1) * bins_per_band;
    }
    std::cout << "Using default uniform spectral division" << std::endl;

    // Debug output for bands if verbose
    if (VERBOSE_OUTPUT) {
        std::cout << "Spectral bands division:" << std::endl;
        for (size_t i = 0; i < bands.size(); i++) {
            double min_freq = bands[i].start_bin * sample_rate / static_cast<double>(N);
            double max_freq = bands[i].end_bin * sample_rate / static_cast<double>(N);
            std::cout << "Band " << i << ": "
                    << bands[i].start_bin << "-" << bands[i].end_bin
                    << " (" << min_freq << "-" << max_freq << " Hz)" << std::endl;
        }
    }

    std::vector<Component> components;
    std::vector<Complex> residual(nyquist_bin);

    // Initialize residual
    std::copy_n(fft.begin(), nyquist_bin, residual.begin());
    residual[0] *= 0.0; // Remove DC

    // Calculate total energy
    double total_energy = 0.0;
    for (const auto &val: residual) {
        total_energy += std::norm(val);
    }

    std::cout << "Total energy: " << total_energy << std::endl;

    // Minimum spectral distance for peak separation
    double min_spectral_distance = 2 / N;
    if (fft.size() != sample_rate) {
        min_spectral_distance *= sample_rate;
    }

    // Main extraction loop
    int component_limit = MAX_COMPONENTS;
    while (components.size() < component_limit) {
        // Phase 1: Find peak candidates in each band (parallel)
        std::vector<std::vector<PeakInfo> > band_peaks(num_bands);
        std::vector<std::thread> find_threads;

        for (size_t b = 0; b < num_bands; b++) {
            find_threads.emplace_back(findPeaksInBand,
                                      std::ref(residual), std::ref(band_peaks), b, std::ref(bands[b]), N);
        }

        // Wait for all peak finding threads
        for (auto &t: find_threads) {
            if (t.joinable()) t.join();
        }

        // Collect all peak candidates
        std::vector<PeakInfo> all_peaks;
        for (const auto &peaks: band_peaks) {
            all_peaks.insert(all_peaks.end(), peaks.begin(), peaks.end());
        }

        // Sort all peaks by magnitude
        std::sort(all_peaks.begin(), all_peaks.end(),
                  [](const PeakInfo &a, const PeakInfo &b) {
                      return a.magnitude > b.magnitude;
                  }
        );

        // Check if any peaks found
        if (all_peaks.empty()) {
            break;
        }

        // Select batch of well-separated peaks
        std::vector<PeakInfo> selected_peaks;
        for (const auto &peak: all_peaks) {
            bool is_separated = true;
            for (const auto &selected: selected_peaks) {
                if (spectralDistance(peak, selected, N) < min_spectral_distance) {
                    is_separated = false;
                    break;
                }
            }

            if (is_separated) {
                selected_peaks.push_back(peak);
                if (selected_peaks.size() >= BATCH_SIZE) break;
            }
        }

        // Handle case where no well-separated peaks found
        if (selected_peaks.empty()) {
            if (!all_peaks.empty()) {
                selected_peaks.push_back(all_peaks[0]);
            } else {
                break;
            }
        }

        // Phase 2: Optimize selected peaks in parallel
        std::vector<Component> batch_components(selected_peaks.size());
        std::vector<std::thread> optimize_threads;

        optimize_threads.reserve(selected_peaks.size());
        for (size_t p = 0; p < selected_peaks.size(); p++) {
            optimize_threads.emplace_back(optimizePeak,
                                          std::ref(residual), std::ref(batch_components),
                                          std::ref(selected_peaks), p, N);
        }

        // Wait for all optimization threads
        for (auto &t: optimize_threads) {
            if (t.joinable()) t.join();
        }

        // Filter components by amplitude threshold
        std::vector<Component> filtered_components;
        for (const auto &comp: batch_components) {
            if (comp.amplitude > 1e-16) {
                filtered_components.push_back(comp);
            }
        }

        // Add to overall component list
        components.insert(components.end(), filtered_components.begin(), filtered_components.end());

        // Phase 3: Update residual with batch components (parallel)
        if (!filtered_components.empty()) {
            const size_t update_threads = std::min(num_threads, nyquist_bin / 1000 + 1);
            const size_t bins_per_thread = nyquist_bin / update_threads;

            for (const auto &component: filtered_components) {
                std::vector<std::thread> update_threads_vec;

                // Update different portions of residual in parallel
                for (size_t t = 0; t < update_threads; t++) {
                    size_t start_bin = t * bins_per_thread + 1; // Skip DC
                    size_t end_bin = (t == update_threads - 1) ? nyquist_bin : (t + 1) * bins_per_thread;

                    update_threads_vec.emplace_back(updateResidualRange,
                                                    std::ref(residual), std::ref(component), start_bin, end_bin, N);
                }

                // Wait for all update threads
                for (auto &t: update_threads_vec) {
                    if (t.joinable()) t.join();
                }
            }
        }

        // Calculate residual energy
        double residual_energy = 0.0;
        for (const auto &val: residual) {
            residual_energy += std::norm(val);
        }

        // Progress reporting
        std::cout << "Batch added " << filtered_components.size() << " components. ";
        std::cout << "Total components: " << components.size() << std::endl;
        const double retention = residual_energy / total_energy;
        std::cout << "% Energy left in residual: " << retention << std::endl;

        // Check convergence
        if (residual_energy < CONVERGENCE * total_energy || filtered_components.empty()) {
            break;
        }
    }

    return components;
}


/**
 * @brief Parallel deconvolution with accuracy metrics collection
 *
 * Same as deconvolveParallelDirichlet but also collects accuracy metrics
 * during extraction for analysis. Compares DKD performance with naive FFT
 * selection approach.
 *
 * @param fft Input spectrum (complex FFT coefficients)
 * @param sample_rate Sampling frequency of the original signal (Hz)
 * @param num_threads Number of threads to use for parallel processing
 * @param metrics_csv_file Path to save metrics CSV file. If empty, no metrics are collected.
 * @param maxcomp Maximum number of components to extract
 * @return Vector of extracted CleanDFT::Component objects
 */
std::vector<CleanDFT::Component> CleanDFT::deconvolveParallelDirichletWithMetrics(
    const std::vector<Complex> &fft, const size_t sample_rate, const size_t num_threads,
    const std::string &metrics_csv_file, const int maxcomp) {
    // Set the global maximum number of components
    MAX_COMPONENTS = maxcomp;

    const size_t N = fft.size(); // Total number of FFT points
    const size_t nyquist_bin = N / 2; // Index of the Nyquist frequency bin

    // Initialize CSV file for metrics if a path is provided
    std::ofstream metrics_file;
    if (!metrics_csv_file.empty()) {
        metrics_file.open(metrics_csv_file);
        // Write CSV header including cosine similarity for both methods
        metrics_file << "batch,components,lsd,rmse,cosine_sim,naive_lsd,naive_rmse,naive_cosine_sim" << std::endl;
    }

    // ======== SPECTRAL BAND DIVISION ========
    // Divide the spectrum (up to Nyquist) into bands for parallel processing
    std::vector<FrequencyBand> bands;
    const size_t num_bands = num_threads; // Use one band per thread

    // Select the method for dividing the spectrum

    // Default to uniform division if the method is not recognized
    bands.resize(num_threads); {
        const size_t bins_per_band = nyquist_bin / num_threads;
        for (size_t i = 0; i < num_threads; i++) {
            bands[i].start_bin = i * bins_per_band + 1;
            bands[i].end_bin = (i == num_threads - 1) ? nyquist_bin : (i + 1) * bins_per_band;
        }
    }
    std::cout << "Using default uniform spectral division" << std::endl;


    // Optional debug output for the frequency bands
    if (VERBOSE_OUTPUT) {
        std::cout << "Spectral bands division:" << std::endl;
        for (size_t i = 0; i < bands.size(); i++) {
            double min_freq = bands[i].start_bin * sample_rate / static_cast<double>(N);
            double max_freq = bands[i].end_bin * sample_rate / static_cast<double>(N);
            std::cout << "Band " << i << ": "
                    << bands[i].start_bin << "-" << bands[i].end_bin
                    << " (" << min_freq << "-" << max_freq << " Hz)" << std::endl;
        }
    }

    // Initialize vectors to store results and intermediate states
    std::vector<Component> components; // Stores the final extracted components
    std::vector<Complex> residual(nyquist_bin); // Stores the residual spectrum (updated iteratively)
    std::vector<Complex> original(nyquist_bin); // Stores the original spectrum (up to Nyquist) for reference

    // Initialize residual and original spectra (copying the first half of the input FFT)
    std::copy_n(fft.begin(), nyquist_bin, residual.begin());
    std::copy_n(fft.begin(), nyquist_bin, original.begin());
    residual[0] *= 0.0; // Remove DC component from residual
    original[0] *= 0.0; // Remove DC component from original (for comparison)

    // Calculate the total energy in the original spectrum (excluding DC)
    double total_energy = 0.0;
    for (const auto &val: residual) {
        total_energy += std::norm(val); // norm = magnitude squared
    }
    std::cout << "Total energy: " << total_energy << std::endl;

    // Maximum number of components to find in a single parallel batch
    // BATCH_SIZE = 128; // Example batch size (could be a class member or constant)

    // Minimum spectral distance required between peaks selected in the same batch
    // This helps avoid selecting multiple peaks that are part of the same underlying component's spread
    double min_spectral_distance = 2.0 / N; // Minimum distance in terms of fractional frequency (2 bins apart)
    if (fft.size() != sample_rate) {
        // Adjust if N is not equal to sample rate (common if windowing or padding is used)
        min_spectral_distance *= sample_rate;
    }

    // Data structures for the baseline naive FFT peak picking approach (if metrics are enabled)
    std::vector<std::pair<size_t, double> > magnitudes; // Stores (bin, magnitude) pairs for sorting
    std::vector<Complex> naive_spectrum; // Spectrum reconstructed by the naive method

    // Prepare data for the naive baseline if metrics are being collected
    if (metrics_file.is_open()) {
        // Initialize the naive spectrum with zeros
        naive_spectrum = std::vector<Complex>(nyquist_bin, Complex(0, 0));

        // Create (bin, magnitude) pairs from the original spectrum (excluding DC)
        magnitudes.reserve(nyquist_bin);
        for (size_t i = 1; i < nyquist_bin; i++) {
            magnitudes.push_back({i, std::abs(original[i])});
        }

        // Sort peaks by magnitude in descending order
        std::sort(magnitudes.begin(), magnitudes.end(),
                  [](const auto &a, const auto &b) {
                      return a.second > b.second; // Lambda for descending sort
                  });
    }

    // Counter for processing batches
    size_t batch_num = 0;

    // ======== MAIN DECONVOLUTION LOOP ========
    // This loop iteratively finds components, subtracts them, and updates the residual
    std::vector<Complex> current_spectrum(nyquist_bin, Complex(0, 0)); // Spectrum reconstructed by DKD method so far
    while (components.size() < MAX_COMPONENTS) {
        batch_num++;

        // --- PHASE 1: Find peak candidates in each band in parallel ---
        std::vector<std::vector<PeakInfo> > band_peaks(num_bands); // Stores peaks found in each band
        std::vector<std::thread> find_threads; // Threads for peak finding

        // Launch a thread for each frequency band to find peaks in the current residual
        for (size_t b = 0; b < num_bands; b++) {
            find_threads.emplace_back(findPeaksInBand, // Function to execute
                                      std::ref(residual), // Pass residual by reference
                                      std::ref(band_peaks), // Pass band_peaks structure by reference
                                      b, // Band index
                                      std::ref(bands[b]), // Pass specific band info by reference
                                      N); // Pass FFT size
        }

        // Wait for all peak finding threads to complete
        for (auto &t: find_threads) {
            if (t.joinable()) t.join();
        }

        // Collect all peak candidates found across all bands
        std::vector<PeakInfo> all_peaks;
        for (const auto &peaks: band_peaks) {
            all_peaks.insert(all_peaks.end(), peaks.begin(), peaks.end());
        }

        // Sort all collected peaks by magnitude (descending) to prioritize strongest peaks
        std::sort(all_peaks.begin(), all_peaks.end(),
                  [](const PeakInfo &a, const PeakInfo &b) {
                      return a.magnitude > b.magnitude;
                  });

        // If no peaks are found in the residual, the deconvolution is finished
        if (all_peaks.empty()) {
            break;
        }

        // Select a batch of peaks for optimization, ensuring they are spectrally separated
        std::vector<PeakInfo> selected_peaks;
        for (const auto &peak: all_peaks) {
            // Check if the current peak is sufficiently far from peaks already selected in this batch
            bool is_separated = true;
            for (const auto &selected: selected_peaks) {
                if (spectralDistance(peak, selected, N) < min_spectral_distance) {
                    is_separated = false; // Too close to an already selected peak
                    break;
                }
            }

            // If separated, add it to the batch
            if (is_separated) {
                selected_peaks.push_back(peak);
                // Stop if the batch size limit is reached
                if (selected_peaks.size() >= BATCH_SIZE) break;
            }
        }

        // Handle the case where no well-separated peaks are found (e.g., due to noise or close components)
        if (selected_peaks.empty()) {
            // If no separated peaks, just take the single strongest peak if available
            if (!all_peaks.empty()) {
                selected_peaks.push_back(all_peaks[0]);
            } else {
                // If even the strongest peak couldn't be added (or no peaks exist), break the loop
                break;
            }
        }

        // --- PHASE 2: Optimize selected peaks in parallel ---
        // Refine the frequency, phase, and amplitude for each selected peak
        std::vector<Component> batch_components(selected_peaks.size()); // Stores optimized components for this batch
        std::vector<std::thread> optimize_threads; // Threads for optimization

        optimize_threads.reserve(selected_peaks.size()); // Reserve space for efficiency
        // Launch a thread to optimize each selected peak
        for (size_t p = 0; p < selected_peaks.size(); p++) {
            optimize_threads.emplace_back(optimizePeak, // Function to execute
                                          std::ref(residual), // Pass residual by reference
                                          std::ref(batch_components), // Pass batch results vector by reference
                                          std::ref(selected_peaks), // Pass selected peaks by reference
                                          p, // Index of the peak to optimize
                                          N); // Pass FFT size
        }

        // Wait for all optimization threads to complete
        for (auto &t: optimize_threads) {
            if (t.joinable()) t.join();
        }

        // Filter the optimized components - keep only those with significant amplitude
        std::vector<Component> filtered_components;
        for (const auto &comp: batch_components) {
            // Use a small threshold to filter out negligible components
            if (comp.amplitude > 1e-16) {
                filtered_components.push_back(comp);
            }
        }

        // Add the valid components found in this batch to the overall list
        components.insert(components.end(), filtered_components.begin(), filtered_components.end());

        // --- PHASE 3: Update residual by subtracting the contribution of batch components in parallel ---
        if (!filtered_components.empty()) {
            // Determine the number of threads for the residual update phase
            const size_t update_threads = std::min(num_threads, nyquist_bin / 1000 + 1); // Adapt based on spectrum size
            const size_t bins_per_thread = nyquist_bin / update_threads;

            // Subtract each component's contribution from the residual
            for (const auto &component: filtered_components) {
                std::vector<std::thread> update_threads_vec; // Threads for updating residual for *this* component

                // Launch threads to update different frequency ranges of the residual
                for (size_t t = 0; t < update_threads; t++) {
                    size_t start_bin = t * bins_per_thread + 1; // Skip DC
                    size_t end_bin = (t == update_threads - 1) ? nyquist_bin : (t + 1) * bins_per_thread;

                    update_threads_vec.emplace_back(updateResidualRange, // Function to execute
                                                    std::ref(residual), // Pass residual by reference
                                                    std::ref(component), // Pass component to subtract by reference
                                                    start_bin, // Start bin for this thread
                                                    end_bin, // End bin for this thread
                                                    N); // Pass FFT size
                }

                // Wait for all update threads for the current component to complete before proceeding to the next component
                for (auto &t: update_threads_vec) {
                    if (t.joinable()) t.join();
                }
            }
        }

        // Update the spectrum reconstructed by the DKD method so far
        // current_spectrum = original - residual
        for (size_t i = 0; i < residual.size(); i++) {
            current_spectrum[i] = original[i] - residual[i];
        }

        // Calculate the energy remaining in the residual
        double residual_energy = 0.0;
        for (const auto &val: residual) {
            residual_energy += std::norm(val);
        }

        // Print progress information
        std::cout << "Batch " << batch_num << " added " << filtered_components.size() << " components. ";
        std::cout << "Total components: " << components.size() << std::endl;
        const double retention = residual_energy / total_energy; // Ratio of remaining energy
        std::cout << "% L2 norm retained: " << retention * 100.0 << std::endl; // Print as percentage

        // === COMPUTE AND RECORD METRICS (if requested) ===
        if (metrics_file.is_open()) {
            // --- Naive Baseline Reconstruction ---
            // Reconstruct the spectrum using the naive approach (top N peaks from original FFT)
            // where N is the number of components found by DKD so far.
            std::vector<Complex> current_naive_spectrum(nyquist_bin, Complex(0, 0));
            if (components.size() <= magnitudes.size()) {
                // Select the top 'components.size()' peaks based on pre-sorted magnitudes
                for (size_t i = 0; i < components.size(); i++) {
                    size_t bin = magnitudes[i].first; // Get the bin index of the i-th largest peak
                    current_naive_spectrum[bin] = original[bin]; // Use the original complex value at that bin
                }
            }

            // --- Calculate Metrics ---
            // Initialize metric variables
            double lsd = 0.0; // Log Spectral Distance (DKD)
            double naive_lsd = 0.0; // Log Spectral Distance (Naive)
            double mse = 0.0; // Mean Squared Error (DKD)
            double naive_mse = 0.0; // Mean Squared Error (Naive)
            // Shannon information calculation removed as per commented-out code
            // Cosine Similarity variables
            double dot_product = 0.0; // Dot product (Original . DKD_Recon)
            double orig_norm_sq = 0.0; // Squared norm of Original spectrum
            double recon_norm_sq = 0.0; // Squared norm of DKD reconstructed spectrum
            double naive_dot_product = 0.0; // Dot product (Original . Naive_Recon)
            double naive_norm_sq = 0.0; // Squared norm of Naive reconstructed spectrum

            // Helper function for safe log calculation (handles zero or negative values, though norm should prevent negatives)
            auto safe_log10 = [](double x) -> double {
                const double epsilon = 1e-10; // Small value to avoid log(0)
                return log10(std::max(x, epsilon));
            };

            // Calculate metrics by comparing reconstructed spectra to the original spectrum bin by bin
            for (size_t i = 1; i < nyquist_bin; i++) {
                // Iterate from bin 1 (skip DC)
                try {
                    double orig_mag_sq = std::norm(original[i]); // Original power at bin i
                    double recon_mag_sq = std::norm(current_spectrum[i]); // DKD reconstructed power at bin i
                    double naive_mag_sq = std::norm(current_naive_spectrum[i]); // Naive reconstructed power at bin i

                    // Convert powers to dB for LSD calculation
                    double orig_power_db = 10.0 * safe_log10(orig_mag_sq); // 10*log10(Power)
                    double recon_power_db = 10.0 * safe_log10(recon_mag_sq);
                    double naive_power_db = 10.0 * safe_log10(naive_mag_sq);

                    // Log Spectral Distance (LSD) - Sum of squared differences in dB
                    double db_diff = orig_power_db - recon_power_db;
                    double naive_db_diff = orig_power_db - naive_power_db;
                    lsd += db_diff * db_diff;
                    naive_lsd += naive_db_diff * naive_db_diff;

                    // Mean Squared Error (MSE) - Sum of squared differences in complex plane
                    Complex diff = original[i] - current_spectrum[i];
                    Complex naive_diff = original[i] - current_naive_spectrum[i];
                    mse += std::norm(diff); // norm() gives squared magnitude |diff|^2
                    naive_mse += std::norm(naive_diff); // |naive_diff|^2

                    // Cosine Similarity - Accumulate dot products and squared norms
                    // Dot product: real(a * conj(b))
                    dot_product += std::real(original[i] * std::conj(current_spectrum[i]));
                    naive_dot_product += std::real(original[i] * std::conj(current_naive_spectrum[i]));
                    // Squared norms: norm(a) = |a|^2
                    orig_norm_sq += orig_mag_sq;
                    recon_norm_sq += recon_mag_sq;
                    naive_norm_sq += naive_mag_sq;
                } catch (const std::exception &e) {
                    std::cerr << "Error calculating metrics at bin " << i << ": " << e.what() << std::endl;
                    // Continue processing other bins if an error occurs
                }
            }

            // Finalize metrics (average and take square root where necessary)
            const double num_bins = static_cast<double>(nyquist_bin - 1); // Number of bins used (excluding DC)

            // LSD: sqrt( mean( (dB_orig - dB_recon)^2 ) )
            lsd = std::sqrt(lsd / num_bins);
            naive_lsd = std::sqrt(naive_lsd / num_bins);

            // RMSE: sqrt( mean( |orig - recon|^2 ) )
            mse = std::sqrt(mse / num_bins);
            naive_mse = std::sqrt(naive_mse / num_bins);

            // Cosine Similarity: (a . b) / (||a|| * ||b||)
            double cosine_sim = 0.0;
            double naive_cosine_sim = 0.0;
            double orig_norm = std::sqrt(orig_norm_sq); // ||original||
            double recon_norm = std::sqrt(recon_norm_sq); // ||dkd_recon||
            double naive_norm = std::sqrt(naive_norm_sq); // ||naive_recon||

            // Calculate cosine similarity, avoiding division by zero
            if (orig_norm > 0 && recon_norm > 0) {
                cosine_sim = dot_product / (orig_norm * recon_norm);
                // Clamp value between -1 and 1 to handle potential floating-point inaccuracies
                cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));
            }
            if (orig_norm > 0 && naive_norm > 0) {
                naive_cosine_sim = naive_dot_product / (orig_norm * naive_norm);
                naive_cosine_sim = std::max(-1.0, std::min(1.0, naive_cosine_sim));
            }

            // Write the calculated metrics for this batch to the CSV file
            metrics_file << batch_num << ","
                    << components.size() << ","
                    << lsd << "," << mse << "," << cosine_sim << "," // DKD metrics
                    << naive_lsd << "," << naive_mse << "," << naive_cosine_sim // Naive metrics
                    << std::endl;
        }

        // Check for convergence: stop if the residual energy is below a threshold fraction of the total energy
        if (residual_energy < CONVERGENCE * total_energy) {
            std::cout << "Convergence threshold reached." << std::endl;
            break; // Exit the main deconvolution loop
        }

        // Also break if we somehow exceeded max components (should be handled by while condition, but good safeguard)
        if (components.size() >= MAX_COMPONENTS) {
            std::cout << "Maximum component limit reached." << std::endl;
            break;
        }
    } // End of main deconvolution loop (while)

    // Close the metrics file if it was opened
    if (metrics_file.is_open()) {
        metrics_file.close();
    }

    // Return the final list of extracted components
    return components;
}


/**
 * @brief Reconstructs the spectrum (up to N points) from Clean components.
 *
 * This function synthesizes the full complex spectrum by summing the
 * contributions of each component's Dirichlet kernel, including mirroring
 * for real-valued signals and phase correction.
 *
 * @param components Vector of extracted CleanDFT::Component objects.
 * @param N The desired size of the output spectrum (and the corresponding time-domain signal).
 * @param DC The DC component (value at bin 0) to add back to the spectrum.
 * @param max_magnitude (Unused in this function, but potentially for normalization elsewhere).
 * @return The reconstructed time-domain signal as a vector of doubles.
 */
std::vector<double> CleanDFT::decompressComponents(const std::vector<Component> &components,
                                                   const size_t N, Complex DC, const double max_magnitude) {
    // Initialize the full spectrum vector with zeros
    std::vector<Complex> spectrum(N, Complex(0, 0));

    // Phase correction is applied within the loop to ensure periodicity
    // when reconstructing from potentially non-integer frequencies.

    int i = 0; // Counter for progress reporting
    size_t length = components.size();
    // Iterate through each extracted component
    for (const auto &[true_frequency, true_phase, amplitude]: components) {
        i++;
        if (VERBOSE_OUTPUT && i % 100 == 0) {
            // Print progress occasionally
            std::cout << "Decompressing component: " << i << " / " << length << std::endl;
        }

        // Calculate the contribution of the component to each bin in the positive frequency range (1 to N-1)
        // We process up to N-1 because bin N is equivalent to bin 0 after FFT properties.
        // The Nyquist bin (N/2) will be handled correctly by the mirroring if N is even.
        for (size_t m = 1; m < N; m++) {
            // Calculate the difference between the component's true frequency and the bin frequency 'm'
            const double diff = true_frequency - static_cast<double>(m);

            // Apply phase correction: ensures the reconstructed sinusoid aligns correctly
            // within the N-point DFT window, accounting for non-integer frequencies.
            const double corrected_phase = true_phase + M_PI * diff / static_cast<double>(N);
            // Standard phase correction
            // Note: The original code had M_PI * diff / N. Testing needed to confirm which correction is theoretically sound/works best.

            // Calculate the component's value at bin 'm' using the Dirichlet kernel
            Complex component_value = amplitude * DirichletKernel::getValueAtBin(
                                          diff, corrected_phase, N
                                      );

            // Add the component's contribution to the spectrum bin 'm'
            spectrum[m] += component_value;

            // For a real-valued time signal, the negative frequency component (mirrored bin)
            // must be the complex conjugate of the positive frequency component.
            // We calculate the contribution *as if* there was a component at -true_frequency
            // and add its conjugate to the appropriate positive bin 'm'.
            // This is equivalent to summing the kernel centered at +freq and the kernel centered at -freq.

            // Determine the bin corresponding to the negative frequency (-m)
            const size_t mirror_bin = N - m;
            // Check bounds: mirror_bin should be valid if m < N

            const double mirror_diff = true_frequency - static_cast<double>(mirror_bin);
            const double mirror_phase_correction =
                    true_phase + M_PI * mirror_diff / static_cast<double>(N);
            // Apply same correction logic
            // Add the conjugate of the kernel value evaluated at the mirror frequency location
            spectrum[m] += std::conj(amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase_correction, N));

        }
    }

    // Restore the DC component
    spectrum[0] = DC;

    // Optional: Enforce conjugate symmetry explicitly after summing all components (might be safer)
    // for (size_t m = 1; m < N / 2; ++m) {
    //     spectrum[N - m] = std::conj(spectrum[m]);
    // }
    // if (N % 2 == 0) { // Handle Nyquist bin for even N (must be real)
    //     spectrum[N / 2] = Complex(std::real(spectrum[N / 2]), 0.0);
    // }


    // Optional: Write the reconstructed spectrum to a file for debugging
    if (VERBOSE_OUTPUT) {
        writeSpectrumToCSV(spectrum, "spectrum_before_ifft.csv");
    }

    // Compute the Inverse FFT to get the time-domain signal
    auto reconstructed = WaveProcessor::computeIFFT(spectrum);

    // Return the reconstructed time-domain signal
    return reconstructed;
}


/**
 * @brief Worker function for parallel component decompression.
 *
 * Calculates the spectral contribution for a subset of components over the full spectrum range.
 * This function is intended to be called by multiple threads.
 *
 * @param components The full vector of CleanDFT::Component objects.
 * @param N The size of the output spectrum.
 * @param start_idx The starting index of the components subset for this worker.
 * @param end_idx The ending index (exclusive) of the components subset for this worker.
 * @return A complex vector representing the partial spectrum generated by this worker's components.
 */
std::vector<Complex> decompressComponentsWorker(
    const std::vector<CleanDFT::Component> &components,
    size_t N,
    size_t start_idx,
    size_t end_idx) {
    // Initialize a local spectrum for this thread's results
    std::vector<Complex> local_spectrum(N, Complex(0, 0));

    // Iterate through the assigned subset of components
    for (size_t i = start_idx; i < end_idx; i++) {
        const auto &component = components[i];
        double true_frequency = component.true_frequency;
        double true_phase = component.true_phase;
        double amplitude = component.amplitude;

        // Calculate contribution to each bin (similar logic as the serial version)
        for (size_t m = 1; m < N; m++) {
            // Iterate through spectrum bins (skip DC)
            const double diff = true_frequency - static_cast<double>(m);
            // Apply phase correction (ensure consistency with serial version)
            const double corrected_phase = true_phase + M_PI * diff/ static_cast<double>(N);
            // Standard phase correction
            Complex component_value = amplitude * DirichletKernel::getValueAtBin(
                                          diff, corrected_phase, N
                                      );
            local_spectrum[m] += component_value;

            // Handle mirroring using the same logic as the serial function for consistency
            const size_t mirror_bin = N - m;

                // Check bounds
            const double mirror_diff = true_frequency - static_cast<double>(mirror_bin);
            const double mirror_phase_correction =
                    true_phase + M_PI * mirror_diff / static_cast<double>(N);
            local_spectrum[m] += std::conj(
                amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase_correction, N));

        }
    }

    // Return the partial spectrum calculated by this thread
    return local_spectrum;
}

/**
 * @brief Reconstructs the spectrum and time-domain signal from Clean components in parallel.
 *
 * Divides the components among multiple threads, each calculating a partial spectrum.
 * The partial spectra are then summed, and an IFFT is performed to get the time-domain signal.
 *
 * @param components Vector of extracted CleanDFT::Component objects.
 * @param N The desired size of the output spectrum and time-domain signal.
 * @param DC The DC component (value at bin 0) to add back to the spectrum.
 * @param max_magnitude (Unused in this function).
 * @param num_threads The number of threads to use for parallel decompression.
 * @return The reconstructed time-domain signal as a vector of doubles.
 */
std::vector<double> CleanDFT::decompressComponentsParallel(
    const std::vector<Component> &components,
    const size_t N,
    const Complex DC,
    const double max_magnitude,
    const size_t num_threads) {
    // Initialize the final spectrum vector with zeros
    std::vector<Complex> spectrum(N, Complex(0, 0));

    // Set the DC component
    spectrum[0] = DC;

    // Determine the number of components each thread will process
    // Use max(1, ...) to avoid division by zero if components.size() < num_threads
    size_t components_per_thread = std::max(size_t(1), components.size() / num_threads);

    // Vector to hold thread objects
    std::vector<std::thread> threads;
    // Vector to store the partial spectrum results from each thread
    std::vector<std::vector<Complex> > thread_results(num_threads);

    // Launch threads, dividing the component list among them
    for (size_t t = 0; t < num_threads; t++) {
        // Calculate the start and end index for the components this thread will handle
        size_t start_idx = t * components_per_thread;
        // The last thread takes any remaining components
        size_t end_idx = (t == num_threads - 1) ? components.size() : (t + 1) * components_per_thread;

        // Ensure the start index is valid (handles cases with few components)
        if (start_idx >= components.size()) continue; // No work for this thread

        // Launch the worker thread
        threads.emplace_back(
            // Lambda function executed by the thread
            [t, &thread_results, &components, N, start_idx, end_idx]() {
                // Capture necessary variables
                // Call the worker function and store its result in the corresponding slot
                thread_results[t] = decompressComponentsWorker(components, N, start_idx, end_idx);
            }
        );
    }

    // Wait for all threads to complete their execution
    for (auto &thread: threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Combine the partial results from all threads into the final spectrum
    for (const auto &local_spectrum: thread_results) {
        // Check if the local spectrum is not empty (possible if a thread had no work)
        if (!local_spectrum.empty()) {
            // Add the contribution from this thread's partial spectrum to the main spectrum
            // Start from bin 1 as DC is already handled and bin 0 is not calculated by workers
            for (size_t m = 1; m < N; m++) {
                spectrum[m] += local_spectrum[m];
            }
        }
    }

    // Optional: Enforce conjugate symmetry after combining results
    // for (size_t m = 1; m < N / 2; ++m) {
    //     spectrum[N - m] = std::conj(spectrum[m]);
    // }
    // if (N % 2 == 0) { // Handle Nyquist bin for even N
    //     spectrum[N / 2] = Complex(std::real(spectrum[N / 2]), 0.0);
    // }

    std::cout << "Combined spectrum size: " << spectrum.size() << std::endl;

    // Optional: Write the final reconstructed spectrum to a file for debugging
    if (VERBOSE_OUTPUT) {
        writeSpectrumToCSV(spectrum, "spectrum_before_ifft_parallel.csv");
    }

    // Compute the Inverse FFT on the final combined spectrum
    auto reconstructed = WaveProcessor::computeIFFT(spectrum);

    // Return the reconstructed time-domain signal
    return reconstructed;
}


/**
 * @brief Extracts components using a non-iterative parallel approach.
 *
 * Identifies all significant peaks in the initial FFT, then processes
 * these peaks in parallel to estimate component parameters (frequency, phase, amplitude)
 * directly from the original spectrum without iterative refinement or residual updates.
 *
 * @param fft Input spectrum (complex FFT coefficients).
 * @param sample_rate Sampling frequency (Hz). (Used implicitly via N).
 * @param num_threads Number of threads for parallel processing.
 * @param maxcomp Maximum number of components (peaks) to extract.
 * @return Vector of extracted CleanDFT::Component objects, sorted by amplitude.
 */
std::vector<CleanDFT::Component> CleanDFT::deconvolveNonIterativeParallel(
    const std::vector<Complex> &fft,
    const size_t sample_rate, // sample_rate is unused currently but kept for API consistency
    const size_t num_threads,
    const int maxcomp) {
    const size_t N = fft.size(); // FFT size
    const size_t nyquist_bin = N / 2; // Nyquist bin index
    MAX_COMPONENTS = maxcomp; // Set global max components limit

    // --- Step 1: Find all significant peaks across the spectrum ---
    std::vector<PeakInfo> all_peaks; // Store identified peaks
    // Iterate through the positive frequency bins (excluding DC and Nyquist)
    for (size_t i = 1; i < nyquist_bin; i++) {
        double magnitude = std::abs(fft[i]); // Magnitude at the current bin

        // Simple peak detection: Check if the magnitude is a local maximum
        // compared to its immediate neighbors.
        if (i > 1 && // Ensure we have a left neighbor (i-1)
            i < nyquist_bin - 1 && // Ensure we have a right neighbor (i+1)
            magnitude > std::abs(fft[i - 1]) && // Greater than left neighbor
            magnitude > std::abs(fft[i + 1])) {
            // Greater than right neighbor

            // Store peak information
            PeakInfo peak{};
            peak.bin = i; // Integer bin index of the peak
            peak.magnitude = magnitude; // Magnitude at the peak bin
            peak.frequency = static_cast<double>(i); // Initial frequency guess (the bin frequency itself)
            peak.phase = std::arg(fft[i]); // Phase at the peak bin
            all_peaks.push_back(peak);
        }
    }

    // Sort all found peaks by magnitude in descending order
    std::sort(all_peaks.begin(), all_peaks.end(),
              [](const PeakInfo &a, const PeakInfo &b) {
                  return a.magnitude > b.magnitude; // Lambda for descending sort
              });

    std::cout << "Found " << all_peaks.size() << " potential peaks (unfiltered)" << std::endl;

    // Limit the number of peaks to process based on MAX_COMPONENTS
    if (all_peaks.size() > MAX_COMPONENTS) {
        all_peaks.resize(MAX_COMPONENTS); // Keep only the strongest peaks
    }

    std::cout << "Processing top " << all_peaks.size() << " peaks" << std::endl;

    // --- Step 2: Process these peaks in parallel to refine parameters ---
    std::vector<Component> all_components; // Vector to store the final extracted components
    std::mutex components_mutex; // Mutex to protect access to all_components from multiple threads

    // Vector to hold thread objects
    std::vector<std::thread> threads;

    // Calculate the number of peaks each thread should process
    // Uses ceiling division: (numerator + denominator - 1) / denominator
    size_t peaks_per_thread = (all_peaks.size() + num_threads - 1) / num_threads;

    // Launch threads, assigning a subset of peaks to each
    for (size_t t = 0; t < num_threads; t++) {
        size_t start_idx = t * peaks_per_thread; // Start index for this thread's peaks
        size_t end_idx = std::min((t + 1) * peaks_per_thread, all_peaks.size()); // End index (exclusive)

        // Skip creating a thread if there are no peaks left for it
        if (start_idx >= all_peaks.size()) continue;

        // Launch the thread
        threads.emplace_back([&, start_idx, end_idx]() {
            // Capture necessary variables by reference or value
            std::vector<Component> thread_components; // Local vector for components found by this thread

            // Process the assigned range of peaks
            for (size_t i = start_idx; i < end_idx; i++) {
                const PeakInfo &peak = all_peaks[i]; // Get the current peak info

                // Basic check: Avoid processing peaks too close to DC (bin 0) or Nyquist (bin N/2)
                // as parameter estimation might be less reliable at the edges.
                if (peak.bin < 2 || peak.bin >= nyquist_bin - 1) continue; // Skip edge cases

                // Process this peak to find refined component parameters
                try {
                    // Refine frequency using an optimization method (e.g., golden section search)
                    // This searches around the initial peak bin for a more precise frequency maximum.
                    double true_freq = findOptimalFrequency(fft, peak.bin, peak.bin); // `fft` is the original spectrum

                    // Refine phase based on the optimized frequency
                    double true_phase = findOptimalPhase(fft, peak.bin, true_freq);

                    // Estimate the true amplitude using the Dirichlet kernel properties
                    // This corrects the observed peak magnitude for the kernel's spreading effect.
                    double amplitude = DirichletKernel::getAmplitudeAtBin(
                        true_freq, // The refined frequency
                        std::abs(fft[peak.bin]), // Magnitude observed at the original peak bin
                        N, // FFT size
                        peak.bin // The original peak bin index
                    );

                    // Add the component only if its estimated amplitude is significant
                    // This filters out components derived from very small peaks or noise.
                    if (amplitude > 1e-7) {
                        // Amplitude threshold
                        thread_components.push_back({true_freq, true_phase, amplitude});
                    }
                } catch (const std::exception &e) {
                    // Catch potential errors during optimization (e.g., numerical issues)
                    std::cerr << "Error processing peak at bin " << peak.bin
                            << ": " << e.what() << std::endl;
                }
            } // End loop over peaks assigned to this thread

            // Safely add the components found by this thread to the shared global list
            {
                // Scope for the lock guard
                std::lock_guard<std::mutex> lock(components_mutex); // Acquire lock
                all_components.insert(all_components.end(),
                                      thread_components.begin(),
                                      thread_components.end());
            } // Lock released automatically here
        }); // End of thread lambda
    } // End loop for launching threads

    // Wait for all processing threads to complete
    for (auto &thread: threads) {
        if (thread.joinable()) thread.join();
    }

    std::cout << "Extracted " << all_components.size() << " components non-iteratively in parallel" << std::endl;

    // Sort the final list of components by amplitude (descending)
    std::sort(all_components.begin(), all_components.end(),
              [](const Component &a, const Component &b) {
                  return a.amplitude > b.amplitude; // Lambda for descending sort
              });

    // Return the sorted list of components
    return all_components;
}


/**
 * @brief Resamples the signal represented by Clean components to a new sample rate.
 *
 * Adjusts component frequencies to the new sample rate and FFT size,
 * removes components above the new Nyquist frequency, and then reconstructs
 * the time-domain signal at the target sample rate using either parallel or
 * serial decompression.
 *
 * @param components Vector of CleanDFT::Component objects representing the signal.
 * @param N Original FFT size corresponding to the components.
 * @param original_sr Original sample rate (Hz).
 * @param target_sr Target sample rate (Hz).
 * @param DC The DC component (value at bin 0) from the original signal.
 * @param max_magnitude (Unused in this function).
 * @param use_parallel Whether to use parallel decompression for reconstruction.
 * @param num_threads Number of threads to use if parallel decompression is enabled.
 * @return Resampled time-domain signal as a vector of doubles.
 */
std::vector<double> CleanDFT::resample(const std::vector<Component> &components,
                                       const size_t N, // Original FFT size
                                       const size_t original_sr, // Original sample rate
                                       const size_t target_sr, // New sample rate
                                       const Complex DC, // Original DC component
                                       const double max_magnitude, // Unused
                                       const bool use_parallel,
                                       const size_t num_threads) {
    // Calculate the new FFT size (new_N) required to maintain the same signal duration
    // Duration = N / original_sr = new_N / target_sr
    // => new_N = N * (target_sr / original_sr)
    // Use ceil to avoid truncation and ensure the duration is not shortened.
    const auto new_N = static_cast<size_t>(
        std::ceil(static_cast<double>(N) * static_cast<double>(target_sr) / static_cast<double>(original_sr))
    );

    // Debug output: Verify calculated durations
    if (VERBOSE_OUTPUT) {
        std::cout << "Original N: " << N << ", SR: " << original_sr << ", Duration: " << static_cast<double>(N) /
                original_sr << "s\n";
        std::cout << "Target   N: " << new_N << ", SR: " << target_sr << ", Duration: " << static_cast<double>(new_N) /
                target_sr << "s\n";
    }

    // Calculate the Nyquist frequency for the target sample rate
    const double new_nyquist_hz = static_cast<double>(target_sr) / 2.0;

    // Vector to store components that are valid at the new sample rate
    std::vector<Component> valid_components;
    valid_components.reserve(components.size()); // Reserve space for efficiency

    // Iterate through the original components
    for (const auto &[true_frequency, true_phase, amplitude]: components) {
        // Convert the component's true frequency (in original bins) to Hz
        const double freq_hz = true_frequency * static_cast<double>(original_sr) / static_cast<double>(N);

        // Check if the component's frequency is below the new Nyquist frequency
        if (freq_hz < new_nyquist_hz) {
            // If valid, scale the frequency to the new bin system (new_N points, target_sr)
            // freq_hz = new_freq_bin * target_sr / new_N
            // => new_freq_bin = freq_hz * new_N / target_sr
            const double new_freq_bin = freq_hz * static_cast<double>(new_N) / static_cast<double>(target_sr);

            // Phase adjustment: The original code keeps the phase the same.
            // This assumes the phase reference (time t=0) remains consistent.
            // A potential adjustment factor might be needed if windowing or frame boundaries shift,
            // but keeping phase constant is a common first approach.
            // Original commented-out phase scaling:
            // const double phase_scale_factor = (static_cast<double>(target_sr) * static_cast<double>(N)) / (static_cast<double>(original_sr) * static_cast<double>(new_N));
            // const double new_phase = true_phase * phase_scale_factor; // Potential adjustment
            const double new_phase = true_phase; // Keep phase unchanged

            // Add the scaled component to the list of valid components
            valid_components.push_back({new_freq_bin, new_phase, amplitude});

            // Debug output for the first few scaled components
            if (VERBOSE_OUTPUT && valid_components.size() <= 5) {
                std::cout << "Comp Valid: Orig Freq " << freq_hz << " Hz (" << true_frequency << " bins) -> "
                        << "New Freq " << new_freq_bin * static_cast<double>(target_sr) / static_cast<double>(new_N) <<
                        " Hz ("
                        << new_freq_bin << " bins)\n";
            }
        } else {
            // Debug output for components filtered out
            if (VERBOSE_OUTPUT && components.size() - valid_components.size() <= 5) {
                std::cout << "Comp Filtered: Orig Freq " << freq_hz << " Hz >= New Nyquist " << new_nyquist_hz <<
                        " Hz\n";
            }
        }
    }
    std::cout << "Kept " << valid_components.size() << " components out of " << components.size() << " for resampling."
            << std::endl;


    // Reconstruct the time-domain signal from the valid, scaled components
    std::vector<double> resampled_signal;
    if (use_parallel) {
        // Use parallel decompression
        resampled_signal = decompressComponentsParallel(valid_components, new_N, DC, max_magnitude, num_threads);
    } else {
        // Use serial decompression
        resampled_signal = decompressComponents(valid_components, new_N, DC, max_magnitude);
    }

    // Return the reconstructed signal at the target sample rate
    return resampled_signal;
}


/**
 * @brief Finds the bin with the maximum magnitude in the residual spectrum.
 *
 * Searches the spectrum (excluding DC and near-Nyquist bins) to find the
 * bin index corresponding to the highest amplitude, representing the strongest
 * remaining frequency component.
 *
 * @param residual The current residual spectrum (complex FFT coefficients).
 * @param nyquist_bin The index of the Nyquist frequency bin (N/2).
 * @return The index of the bin with the maximum magnitude. Returns 0 if no peak found (e.g., all zero).
 */
int CleanDFT::findPeakBin(const std::vector<Complex> &residual, size_t nyquist_bin) {
    int peak_bin = 0; // Initialize peak bin index to 0 (DC)
    double max_magnitude = 0.0; // Initialize max magnitude found so far

    // Iterate through bins from 1 (skip DC) up to, but not including, nyquist_bin.
    // Also added a constraint to avoid peaks very close to Nyquist (0.99 factor).
    for (size_t i = 1; i < nyquist_bin; i++) {
        const double magnitude = std::abs(residual[i]); // Get magnitude at bin i

        // Check if this magnitude is the largest found so far AND
        // if the bin is not too close to the Nyquist frequency.
        // The 0.99 factor prevents selecting peaks right at the edge where estimation might fail.
        if (magnitude > max_magnitude && static_cast<double>(i) < static_cast<double>(nyquist_bin) * 0.99) {
            max_magnitude = magnitude; // Update max magnitude
            peak_bin = static_cast<int>(i); // Update peak bin index
        }
    }

    // Return the index of the bin containing the peak magnitude
    return peak_bin;
}

/**
 * @brief Checks if the exit conditions for the iterative deconvolution are met.
 *
 * Determines whether to stop the iterative process based on the location
 * and magnitude of the current peak found in the residual spectrum.
 *
 * @param peak_bin The bin index of the currently found peak.
 * @param nyquist_bin The index of the Nyquist frequency bin (N/2).
 * @param residual The current residual spectrum.
 * @return True if the exit conditions are met (stop iteration), False otherwise.
 */
bool CleanDFT::isExitConditionMet(int peak_bin, size_t nyquist_bin,
                                  const std::vector<Complex> &residual) {
    // Condition 1: Peak bin is too close to the Nyquist frequency.
    // Processing peaks very near Nyquist can be numerically unstable or less meaningful.
    // The 0.99 factor defines "too close".
    if (static_cast<double>(peak_bin) >= static_cast<double>(nyquist_bin) * 0.99) {
        if (VERBOSE_OUTPUT) std::cout << "Exit condition: Peak bin " << peak_bin << " too close to Nyquist " <<
                            nyquist_bin << std::endl;
        return true; // Stop iteration
    }

    // Condition 2: The magnitude of the peak is below a minimum threshold.
    // If the strongest remaining component is very weak, further iterations might just model noise.
    if (peak_bin > 0 && peak_bin < residual.size()) {
        // Ensure peak_bin is a valid index
        double max_magnitude = std::abs(residual[peak_bin]);
        if (max_magnitude < 1e-6) {
            // Magnitude threshold
            if (VERBOSE_OUTPUT) std::cout << "Exit condition: Peak magnitude " << max_magnitude << " too low." <<
                                std::endl;
            return true; // Stop iteration
        }
    } else {
        // If peak_bin is invalid (e.g., 0 was returned from findPeakBin), stop.
        if (VERBOSE_OUTPUT) std::cout << "Exit condition: Invalid peak bin " << peak_bin << "." << std::endl;
        return true;
    }


    // If neither exit condition is met, continue the iteration
    return false;
}
