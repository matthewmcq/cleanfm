//
// Created by Matthew McQuistion on 12/17/24.
//

#include "cleandft.h"
#include "threadpool.h"
#include <fstream>
#include <iostream>


CleanDFT::CleanDFT() = default;

Complex CleanDFT::computeDC(const std::vector<Component> &components, const size_t N) {
    Complex new_dc = 0;
    for (const auto &[true_frequency, true_phase, amplitude]: components) {
        // Compute how each component contributes to bin 0
        new_dc += amplitude * DirichletKernel::getValueAtBin(
            -true_frequency,
            true_phase,
            N
        );
    }
    return new_dc;
}

double CleanDFT::spectralDistance(const PeakInfo &peak1, const PeakInfo &peak2, size_t N) {
    // Simple distance based on frequency bins
    double bin_distance = std::abs(peak1.frequency - peak2.frequency);

    // Normalize by N to get relative distance in [0,1]
    return bin_distance / N;
}


double CleanDFT::computeCorrelation(const std::vector<Complex> &data_slice, const std::vector<Complex> &kernel,
                                    const bool isPhase) {
    Complex numerator = 0;
    for (size_t i = 0; i < data_slice.size(); i++) {
        numerator += data_slice[i] * std::conj(kernel[i]);
    }

    double data_norm = 0;
    double kernel_norm = 0;
    for (size_t i = 0; i < data_slice.size(); i++) {
        data_norm += std::abs(data_slice[i]) * std::abs(data_slice[i]);
        kernel_norm += std::abs(kernel[i]) * std::abs(kernel[i]);
    }

    if (isPhase) {
        // For phase correlation, we want the real part to distinguish between phases
        return std::real(numerator) / (std::sqrt(data_norm) * std::sqrt(kernel_norm));
    }
    // For frequency correlation, we want the magnitude
    return std::abs(numerator) / (std::sqrt(data_norm) * std::sqrt(kernel_norm));
}

double CleanDFT::calculateCorrelation(
    const std::vector<Complex> &residual,
    double carrier_freq, double carrier_phase, double carrier_amp,
    double mod_index, double lfo_freq, double lfo_index, size_t N) {
    // Generate the FM spectrum
    std::vector<Complex> fm_spectrum(residual.size(), Complex(0, 0));

    // Add contribution of both modulators
    addCombinedModulators(
        fm_spectrum, N, carrier_freq, carrier_phase, carrier_amp,
        mod_index, lfo_freq, lfo_index);

    // Calculate correlation coefficient
    Complex numerator = 0;
    double norm_residual = 0;
    double norm_fm = 0;

    for (size_t i = 1; i < residual.size(); i++) {
        numerator += residual[i] * std::conj(fm_spectrum[i]);
        norm_residual += std::norm(residual[i]);
        norm_fm += std::norm(fm_spectrum[i]);
    }

    if (norm_residual < 1e-10 || norm_fm < 1e-10) {
        return 0.0;
    }

    return std::abs(numerator) / (std::sqrt(norm_residual) * std::sqrt(norm_fm));
}

double CleanDFT::findOptimalPhase(const std::vector<Complex> &fft, const int center_bin, const double frequency) {
    // Create bins array relative to center bin
    std::vector<int> bins;
    for (int i = center_bin - PHASE_WINDOW_SIZE;
         i <= center_bin + PHASE_WINDOW_SIZE; i++) {
        if (i >= 0 && i < fft.size()) {
            bins.push_back(i);
        }
    }

    // Get data slice using these bins
    std::vector<Complex> data_slice;
    data_slice.reserve(bins.size());
    for (const int bin: bins) {
        data_slice.push_back(fft[bin]);
    }

    auto evaluate = [&](const double phase) {
        const std::vector<Complex> kernel = DirichletKernel::generateKernel(
            bins, frequency, phase, fft.size(), std::abs(fft[center_bin])
        );
        return computeCorrelation(data_slice, kernel, true);
    };
    return goldenSectionSearch(evaluate, -1.0 * M_PI, 1.0 * M_PI);
}

double CleanDFT::findOptimalFrequency(const std::vector<Complex> &fft, const int center_bin,
                                      const double test_frequency) {
    // Create bins array relative to center bin
    std::vector<int> bins;
    for (int i = center_bin - FREQUENCY_WINDOW_SIZE;
         i <= center_bin + FREQUENCY_WINDOW_SIZE; i++) {
        if (i >= 0 && i < fft.size()) {
            bins.push_back(i);
        }
    }

    // Get data slice using these bins
    std::vector<Complex> data_slice;
    data_slice.reserve(bins.size());
    for (const int bin: bins) {
        data_slice.push_back(fft[bin]);
    }

    const double center_phase = std::arg(fft[center_bin]);

    auto evaluate = [&](const double freq) {
        const std::vector<Complex> kernel = DirichletKernel::generateKernel(
            bins, freq, center_phase, fft.size(), std::abs(fft[center_bin])
        );
        return computeCorrelation(data_slice, kernel, false);
    };
    return goldenSectionSearch(evaluate, test_frequency - 0.5, test_frequency + 0.5);
}

std::vector<CleanDFT::Component> CleanDFT::deconvolveDirichletKernel(const std::vector<Complex> &fft,
                                                                     const size_t sample_rate) {
    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;

    std::vector<Component> components;
    std::vector<Complex> residual(nyquist_bin);

    std::copy_n(fft.begin(), nyquist_bin, residual.begin());

    residual[0] *= 0.0; // remove DC

    // Calculate total energy
    double total_energy = 0.0;
    for (const auto &val: residual) {
        total_energy += std::norm(val);
    }

    std::cout << "total energy: " << total_energy << std::endl;
    while (components.size() < MAX_COMPONENTS) {
        // Find peak bin
        int peak_bin = 0;
        double max_magnitude = 0.0;
        int count = 0;
        for (int i = 1; i < nyquist_bin; i++) {
            // start @ 1 to skip DC since we deal w it separately
            const double magnitude = std::abs(residual[i]);
            // if (magnitude > std::abs(residual[i - 1])) count++;
            if (magnitude > max_magnitude && i < static_cast<double>(nyquist_bin) * 0.99) {
                max_magnitude = magnitude;
                peak_bin = i;
            }
        }
        // std::cout << "count: " << count << std::endl;

        // Add sanity check
        if (static_cast<double>(peak_bin) >= static_cast<double>(nyquist_bin) * 0.99 || max_magnitude < 1e-6) {
            std::cout << "Peak bin too high or magnitude too low, stopping..." << std::endl;
            std::cout << peak_bin << " " << peak_bin * sample_rate / N << std::endl;
            break;
        }

        // Find optimal frequency
        const double true_freq = findOptimalFrequency(residual, peak_bin, peak_bin);


        const double true_phase = findOptimalPhase(residual, peak_bin, true_freq);

        const double amplitude = DirichletKernel::getAmplitudeAtBin(true_freq, std::abs(residual[peak_bin]), N,
                                                                    peak_bin);

        components.push_back({true_freq, true_phase, amplitude});

        // Update residual
        for (size_t i = 1; i < N / 2; i++) {
            // Only go up to N/2
            // Positive frequency
            const double diff = true_freq - static_cast<double>(i);
            const double corrected_phase = true_phase + M_PI * diff / static_cast<double>(N);
            residual[i] -= amplitude * DirichletKernel::getValueAtBin(diff, corrected_phase, N);

            // Mirror frequency
            const size_t mirror_bin = N - i;
            const double mirror_diff = true_freq - static_cast<double>(mirror_bin);
            const double mirror_phase = true_phase + M_PI * mirror_diff / static_cast<double>(N);
            // Mirror should be complex conjugate
            residual[i] -= std::conj(amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
        }

        // Calculate residual energy
        double residual_energy = 0.0;
        for (const auto &val: residual) {
            residual_energy += std::norm(val);
        }

        // Print progress
        std::cout << "Component " << components.size() << ": true bin="
                << true_freq <<
                ": old bin="
                << peak_bin
                << ", phase=" << true_phase <<
                std::endl;
        const double retention = residual_energy / total_energy * 100;

        std::cout << "% L2 norm retained: " << retention << std::endl;

        if (constexpr double THRESHOLD = 1e-6; residual_energy < THRESHOLD * total_energy) {
            break;
        }
    }

    return components;
}

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

// Thread worker function for finding peaks in a band
void findPeaksInBand(
    const std::vector<Complex>& residual,
    std::vector<std::vector<PeakInfo>>& band_peaks,
    size_t band_index,
    const FrequencyBand& band,
    size_t N
) {
    std::vector<PeakInfo> peaks;

    // Find local maxima in this band
    for (size_t i = band.start_bin + 1; i < band.end_bin - 1; i++) {
        double magnitude = std::abs(residual[i]);
        if (magnitude > std::abs(residual[i - 1]) &&
            magnitude > std::abs(residual[i + 1]) &&
            magnitude > 1e-10) {

            // Calculate spectral centroid
            double weighted_sum = 0;
            double sum_weights = 0;
            for (int j = std::max(static_cast<int>(i) - 5, 0);
                 j <= std::min(static_cast<int>(i) + 5, static_cast<int>(band.end_bin));
                 j++) {
                double weight = std::abs(residual[j]);
                weighted_sum += j * weight;
                sum_weights += weight;
            }
            double centroid = sum_weights > 0 ? weighted_sum / sum_weights : i;

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

    // Sort peaks by magnitude (descending)
    std::sort(peaks.begin(), peaks.end(),
        [](const PeakInfo& a, const PeakInfo& b) {
            return a.magnitude > b.magnitude;
        }
    );

    // Store results directly (no mutex needed since we write to separate indices)
    band_peaks[band_index] = std::move(peaks);
}

// Thread worker function for optimizing a peak
void optimizePeak(
    const std::vector<Complex>& residual,
    std::vector<CleanDFT::Component>& batch_components,
    const std::vector<PeakInfo>& selected_peaks,
    size_t peak_index,
    size_t N
) {
    const PeakInfo& peak = selected_peaks[peak_index];

    // Find optimal frequency
    double true_freq = CleanDFT::findOptimalFrequency(residual, peak.bin, peak.bin);

    // Find optimal phase
    double true_phase = CleanDFT::findOptimalPhase(residual, peak.bin, true_freq);

    // Compute amplitude
    double amplitude = DirichletKernel::getAmplitudeAtBin(
        true_freq, std::abs(residual[peak.bin]), N, peak.bin);

    // Store result directly (no mutex needed since we write to separate indices)
    batch_components[peak_index] = CleanDFT::Component{true_freq, true_phase, amplitude};
}

// Update residual for a component in a specific frequency range
void updateResidualRange(
    std::vector<Complex>& residual,
    const CleanDFT::Component& component,
    size_t start_bin,
    size_t end_bin,
    size_t N
) {
    for (size_t i = start_bin; i < end_bin; i++) {
        // Positive frequency
        double diff = component.true_frequency - static_cast<double>(i);
        double corrected_phase = component.true_phase + M_PI * diff / static_cast<double>(N);
        residual[i] -= component.amplitude * DirichletKernel::getValueAtBin(diff, corrected_phase, N);

        // Mirror frequency (if needed)
        size_t mirror_bin = N - i;
        if (mirror_bin < N) {
            double mirror_diff = component.true_frequency - static_cast<double>(mirror_bin);
            double mirror_phase = component.true_phase + M_PI * mirror_diff / static_cast<double>(N);
            // Mirror should be complex conjugate
            residual[i] -= std::conj(component.amplitude *
                                     DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
        }
    }
}

// Main parallel deconvolution function
std::vector<CleanDFT::Component> CleanDFT::deconvolveParallelDirichlet(
    const std::vector<Complex>& fft, const size_t sample_rate, const size_t num_threads) {

    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;

    // Create bands - more bands than threads for better load balancing
    const size_t num_bands = num_threads ;
    std::vector<FrequencyBand> bands(num_bands);

    const size_t bins_per_band = nyquist_bin / num_bands;
    for (size_t i = 0; i < num_bands; i++) {
        bands[i].start_bin = i * bins_per_band + 1; // Skip DC
        bands[i].end_bin = (i == num_bands - 1) ? nyquist_bin : (i + 1) * bins_per_band;
    }

    std::vector<Component> components;
    std::vector<Complex> residual(nyquist_bin);

    // Initialize residual
    std::copy_n(fft.begin(), nyquist_bin, residual.begin());
    residual[0] *= 0.0; // Remove DC

    // Calculate total energy
    double total_energy = 0.0;
    for (const auto& val : residual) {
        total_energy += std::norm(val);
    }

    std::cout << "Total energy: " << total_energy << std::endl;

    // Maximum number of components to find in one batch
    const size_t batch_size = 256;  // Reduced batch size for better L2 norm

    // Minimum spectral distance required between peaks in the same batch
    // Increased minimum distance for better component separation
    const double min_spectral_distance =  sample_rate / N;

    // Use a fixed thread approach without lambdas
    while (components.size() < MAX_COMPONENTS) {
        // --- PHASE 1: Find peak candidates in each band ---
        std::vector<std::vector<PeakInfo>> band_peaks(num_bands);
        std::vector<std::thread> find_threads;

        for (size_t b = 0; b < num_bands; b++) {
            find_threads.emplace_back(findPeaksInBand,
                std::ref(residual), std::ref(band_peaks), b, std::ref(bands[b]), N);
        }

        // Wait for all peak finding threads to complete
        for (auto& t : find_threads) {
            if (t.joinable()) t.join();
        }

        // Collect all peak candidates
        std::vector<PeakInfo> all_peaks;
        for (const auto& peaks : band_peaks) {
            all_peaks.insert(all_peaks.end(), peaks.begin(), peaks.end());
        }

        // Sort all peaks by magnitude (descending)
        std::sort(all_peaks.begin(), all_peaks.end(),
            [](const PeakInfo& a, const PeakInfo& b) {
                return a.magnitude > b.magnitude;
            }
        );

        // If no peaks found, we're done
        if (all_peaks.empty()) {
            break;
        }

        // Select a batch of peaks that are spectrally separated
        std::vector<PeakInfo> selected_peaks;
        for (const auto& peak : all_peaks) {
            // Check if this peak is far enough from all already selected peaks
            bool is_separated = true;
            for (const auto& selected : selected_peaks) {
                if (spectralDistance(peak, selected, N) < min_spectral_distance) {
                    is_separated = false;
                    break;
                }
            }

            if (is_separated) {
                selected_peaks.push_back(peak);
                if (selected_peaks.size() >= batch_size) break;
            }
        }

        if (selected_peaks.empty()) {
            // If we can't find any well-separated peaks, try with a single peak
            if (!all_peaks.empty()) {
                selected_peaks.push_back(all_peaks[0]);
            } else {
                break;
            }
        }

        // --- PHASE 2: Optimize selected peaks in parallel ---
        std::vector<Component> batch_components(selected_peaks.size());
        std::vector<std::thread> optimize_threads;

        optimize_threads.reserve(selected_peaks.size());
        for (size_t p = 0; p < selected_peaks.size(); p++) {
            optimize_threads.emplace_back(optimizePeak,
                std::ref(residual), std::ref(batch_components),
                std::ref(selected_peaks), p, N);
        }

        // Wait for all optimization threads to complete
        for (auto& t : optimize_threads) {
            if (t.joinable()) t.join();
        }

        // For each component, verify it's a significant improvement
        std::vector<Component> filtered_components;
        for (const auto& comp : batch_components) {
            // Only keep components with amplitude above threshold
            if (comp.amplitude > 1e-8) {
                filtered_components.push_back(comp);
            }
        }

        // Add to overall component list
        components.insert(components.end(), filtered_components.begin(), filtered_components.end());

        // --- PHASE 3: Update residual with batch components in parallel ---
        if (!filtered_components.empty()) {
            // Divide the frequency range for parallel processing
            const size_t update_threads = std::min(num_threads, nyquist_bin / 1000 + 1);
            const size_t bins_per_thread = nyquist_bin / update_threads;

            for (const auto& component : filtered_components) {
                std::vector<std::thread> update_threads_vec;

                // Update different portions of the residual in parallel
                for (size_t t = 0; t < update_threads; t++) {
                    size_t start_bin = t * bins_per_thread + 1; // Skip DC
                    size_t end_bin = (t == update_threads - 1) ? nyquist_bin : (t + 1) * bins_per_thread;

                    update_threads_vec.emplace_back(updateResidualRange,
                        std::ref(residual), std::ref(component), start_bin, end_bin, N);
                }

                // Wait for all update threads to complete before processing next component
                for (auto& t : update_threads_vec) {
                    if (t.joinable()) t.join();
                }
            }
        }

        // Calculate residual energy
        double residual_energy = 0.0;
        for (const auto& val : residual) {
            residual_energy += std::norm(val);
        }

        // Print progress
        std::cout << "Batch added " << filtered_components.size() << " components. ";
        std::cout << "Total components: " << components.size() << std::endl;
        const double retention = residual_energy / total_energy * 100;
        std::cout << "% L2 norm retained: " << retention << std::endl;

        // Check convergence - using slightly more strict threshold
        if (residual_energy < 0.5e-6 * total_energy) {
            break;
        }
    }

    return components;
}


std::vector<double> CleanDFT::decompressComponents(const std::vector<Component> &components,
                                                   const size_t N) {
    std::vector spectrum(N, Complex(0, 0));
    spectrum[0] = computeDC(components, N);


    // Phase correction to ensure periodicity
    for (const auto &[true_frequency, true_phase, amplitude]: components) {
        for (size_t m = 1; m < N; m++) {
            const double diff = true_frequency - static_cast<double>(m);
            // Add phase correction term to ensure periodicity
            const double corrected_phase = true_phase + M_PI * diff / static_cast<double>(N);
            spectrum[m] += amplitude * DirichletKernel::getValueAtBin(
                diff, corrected_phase, N
            );
        }
    }

    // Handle symmetry

    for (size_t m = 1; m <= N / 2; m++) {
        spectrum[N - m] += std::conj(spectrum[m]); // * Complex(0, 1);
    }

    // Compensate for tail reflections across Nyquist
    for (size_t m = 1; m <= N / 2; m++) {
        spectrum[m] = std::conj(spectrum[N - m]); // * Complex(0, 1);
    }

    writeSpectrumToCSV(spectrum, "spectrum_before_ifft.csv");

    auto reconstructed = WaveProcessor::computeIFFT(spectrum);

    return reconstructed;
}

std::vector<double> CleanDFT::resample(const std::vector<Component> &components,
                                       const size_t N,
                                       const size_t original_sr,
                                       const size_t target_sr) {
    const auto new_N = static_cast<size_t>(
        std::ceil(static_cast<double>(N) * static_cast<double>(target_sr) / static_cast<double>(original_sr))
    );

    // Debug output
    std::cout << "Original duration: " << static_cast<double>(N) / static_cast<double>(original_sr) << "s\n";
    std::cout << "Expected new duration: " << static_cast<double>(new_N) / static_cast<double>(target_sr) << "s\n";

    const double new_nyquist = static_cast<double>(target_sr) / 2.0;
    std::vector<Component> valid_components;
    valid_components.reserve(components.size());

    for (const auto &[true_frequency, true_phase, amplitude]: components) {
        if (const double freq_hz = true_frequency * static_cast<double>(original_sr) / static_cast<double>(N);
            freq_hz < new_nyquist) {
            const double new_freq_bin = freq_hz * static_cast<double>(new_N) / static_cast<double>(target_sr);
            // Convert Hz to new bins
            const double new_phase = true_phase * (static_cast<double>(target_sr) * static_cast<double>(N)) / (
                                         static_cast<double>(original_sr) * static_cast<double>(new_N));
            valid_components.push_back({new_freq_bin, new_phase, amplitude});
            // Debug output for a few components
            if (valid_components.size() <= 3) {
                std::cout << "Original freq: " << freq_hz << "Hz, "
                        << "Scaled freq: " << new_freq_bin * static_cast<double>(target_sr) / static_cast<double>(new_N)
                        << "Hz\n";
            }
        }
    }

    auto reconstructed = decompressComponents(valid_components, new_N);
    return reconstructed;
}

#include <cmath>
#include "cleandft.h"
#include <fstream>
#include <iostream>
#include <cmath>

// Implementation for findPeakBin
int CleanDFT::findPeakBin(const std::vector<Complex> &residual, size_t nyquist_bin) {
    int peak_bin = 0;
    double max_magnitude = 0.0;

    for (int i = 1; i < nyquist_bin; i++) {
        // Start at 1 to skip DC
        const double magnitude = std::abs(residual[i]);
        if (magnitude > max_magnitude && i < static_cast<double>(nyquist_bin) * 0.99) {
            max_magnitude = magnitude;
            peak_bin = i;
        }
    }

    return peak_bin;
}

// Implementation for isExitConditionMet
bool CleanDFT::isExitConditionMet(int peak_bin, size_t nyquist_bin,
                                  const std::vector<Complex> &residual) {
    // Check if peak bin is too high (near Nyquist)
    if (static_cast<double>(peak_bin) >= static_cast<double>(nyquist_bin) * 0.99) {
        std::cout << "Peak bin too high, stopping..." << std::endl;
        return true;
    }

    // Check if magnitude is too low
    double max_magnitude = std::abs(residual[peak_bin]);
    if (max_magnitude < 1e-6) {
        std::cout << "Magnitude too low, stopping..." << std::endl;
        return true;
    }

    return false;
}

void CleanDFT::printProgress(size_t component_count, double carrier_freq, int peak_bin,
                             double carrier_phase, double mod_index, double lfo_freq, double lfo_index,
                             const std::vector<Complex> &residual, double total_energy) {
    // Calculate residual energy
    double residual_energy = 0.0;
    for (const auto &val: residual) {
        residual_energy += std::norm(val);
    }

    // Calculate percentage of energy retained
    const double retention = residual_energy / total_energy * 100;
    const double captured = 100.0 - retention;

    // Print progress information
    std::cout << "FM Component " << component_count << ": carrier="
            << carrier_freq
            // ", old bin=" << peak_bin
            << ", phase=" << carrier_phase
            << ", mod_index=" << mod_index
            << ", lfo_freq=" << lfo_freq
            << ", lfo_index=" << lfo_index << std::endl;

    std::cout << "% L2 norm retained: " << retention
            << "%, captured: " << captured << "%" << std::endl;
}

// Bessel function implementation
double CleanDFT::besselJ(int n, double x) {
    // Handle negative orders
    if (n < 0) {
        return (n % 2 == 0 ? 1 : -1) * besselJ(-n, x);
    }

    // Handle special case n=0
    if (n == 0) {
        if (std::abs(x) < 1e-10) return 1.0;

        // Series approximation for small x
        if (std::abs(x) < 3.0) {
            double sum = 1.0;
            double term = 1.0;
            double x2 = x * x / 4.0;

            for (int k = 1; k <= 15; ++k) {
                term *= -x2 / (k * k);
                sum += term;

                if (std::abs(term) < 1e-15 * std::abs(sum)) break;
            }

            return sum;
        }
    }

    // Handle special case n=1
    if (n == 1) {
        if (std::abs(x) < 1e-10) return 0.0;

        // Series approximation for small x
        if (std::abs(x) < 3.0) {
            double sum = 0.0;
            double term = x / 2.0;
            double x2 = x * x / 4.0;

            sum = term;

            for (int k = 1; k <= 15; ++k) {
                term *= -x2 / (k * (k + 1));
                sum += term;

                if (std::abs(term) < 1e-15 * std::abs(sum)) break;
            }

            return sum;
        }
    }

    // For larger n or x values, use recurrence relation but avoid stack overflow
    if (n > 1) {
        if (std::abs(x) < 1e-10) return 0.0; // Avoid division by zero

        // For small n, direct recurrence is safe
        if (n <= 15) {
            double j0 = besselJ(0, x);
            double j1 = besselJ(1, x);

            double jn = j1;
            double jnm1 = j0;
            double jnp1;

            for (int i = 1; i < n; ++i) {
                jnp1 = (2.0 * i / x) * jn - jnm1;
                jnm1 = jn;
                jn = jnp1;
            }

            return jn;
        }
        // For larger n, use approximate value
        return std::pow(0.5 * x / n, n) / std::tgamma(n + 1);
    }

    // Fallback for large x (approximate)
    return std::sqrt(2.0 / (M_PI * std::abs(x))) *
           std::cos(std::abs(x) - n * M_PI / 2.0 - M_PI / 4.0);
}

// Implementation for includeSideband
bool CleanDFT::includeSideband(int sideband_number, SidebandMode mode) {
    // Always include the carrier (sideband 0)
    if (sideband_number < 0) {
        return false;
    }

    if (sideband_number == 0) {
        return true;
    }

    int absN = std::abs(sideband_number);

    switch (mode) {
        case SidebandMode::ALL_SIDEBANDS:
            return true;

        case SidebandMode::EVEN_SIDEBANDS:
            return absN % 2 == 0;

        case SidebandMode::ODD_SIDEBANDS:
            return absN % 2 != 0;

        default:
            return true;
    }
}

// std::tuple<double, double, double> CleanDFT::findOptimalFMParameters(
//     double carrier_freq, double carrier_phase, double carrier_amp,
//     const std::vector<Complex> &residual, size_t N) {
//
//     // Starting parameters
//     double mod_index = 0.5;      // Start at middle of range
//     double lfo_ratio = 0.05;     // Start with a small ratio
//     double lfo_index = 0.5;      // Start at middle of range
//
//     // Learning rates
//     const double lr_mod = 0.05;
//     const double lr_lfo_ratio = 0.01;
//     const double lr_lfo_index = 0.05;
//
//     // Parameter bounds
//     const double min_mod_index = 0.0, max_mod_index = 1.0;
//     const double min_lfo_ratio = 0.0, max_lfo_ratio = 0.1;
//     const double min_lfo_index = 0.0, max_lfo_index = 1.0;
//
//     // Compute initial correlation
//     double lfo_freq = carrier_freq * lfo_ratio;
//     double best_correlation = evaluateCorrelation(
//         residual, N, carrier_freq, carrier_phase, carrier_amp,
//         mod_index, lfo_freq, lfo_index);
//
//     // Track best parameters found
//     double best_mod_index = mod_index;
//     double best_lfo_freq = lfo_freq;
//     double best_lfo_index = lfo_index;
//
//     // Maximum iterations and convergence criteria
//     const int max_iterations = 50;
//     const double convergence_threshold = 0.0001;
//     double prev_correlation = 0.0;
//
//     std::cout << "Initial correlation: " << best_correlation << std::endl;
//
//     // Gradient descent main loop
//     for (int iter = 0; iter < max_iterations; iter++) {
//         // Save previous values
//         double prev_mod_index = mod_index;
//         double prev_lfo_ratio = lfo_ratio;
//         double prev_lfo_index = lfo_index;
//         prev_correlation = best_correlation;
//
//         // Compute numerical gradients
//         // For mod_index
//         double grad_mod = computeGradient(
//             residual, N, carrier_freq, carrier_phase, carrier_amp,
//             mod_index, lfo_freq, lfo_index, "mod_index");
//
//         // For lfo_ratio
//         double grad_lfo_ratio = computeGradient(
//             residual, N, carrier_freq, carrier_phase, carrier_amp,
//             mod_index, lfo_freq, lfo_index, "lfo_ratio");
//
//         // For lfo_index
//         double grad_lfo_index = computeGradient(
//             residual, N, carrier_freq, carrier_phase, carrier_amp,
//             mod_index, lfo_freq, lfo_index, "lfo_index");
//
//         // Update parameters with gradient ascent (since we want to maximize correlation)
//         mod_index += lr_mod * grad_mod;
//         lfo_ratio += lr_lfo_ratio * grad_lfo_ratio;
//         lfo_index += lr_lfo_index * grad_lfo_index;
//
//         // Clamp parameters to allowed ranges
//         mod_index = std::clamp(mod_index, min_mod_index, max_mod_index);
//         lfo_ratio = std::clamp(lfo_ratio, min_lfo_ratio, max_lfo_ratio);
//         lfo_index = std::clamp(lfo_index, min_lfo_index, max_lfo_index);
//
//         // Update lfo_freq based on new ratio
//         lfo_freq = carrier_freq * lfo_ratio;
//
//         // Evaluate new correlation
//         double correlation = evaluateCorrelation(
//             residual, N, carrier_freq, carrier_phase, carrier_amp,
//             mod_index, lfo_freq, lfo_index);
//
//         // Update best parameters if we found better ones
//         if (correlation > best_correlation) {
//             best_correlation = correlation;
//             best_mod_index = mod_index;
//             best_lfo_freq = lfo_freq;
//             best_lfo_index = lfo_index;
//         } else {
//             // If correlation didn't improve, reduce learning rates and try again
//             mod_index = prev_mod_index;
//             lfo_ratio = prev_lfo_ratio;
//             lfo_index = prev_lfo_index;
//             lfo_freq = carrier_freq * lfo_ratio;
//
//             // Reduce learning rates
//             lr_mod *= 0.5;
//             lr_lfo_ratio *= 0.5;
//             lr_lfo_index *= 0.5;
//         }
//
//         std::cout << "Iteration " << iter << ": correlation=" << correlation
//                  << ", mod_index=" << mod_index
//                  << ", lfo_ratio=" << lfo_ratio
//                  << ", lfo_index=" << lfo_index << std::endl;
//
//         // Check convergence
//         if (std::abs(correlation - prev_correlation) < convergence_threshold ||
//             (lr_mod < 1e-6 && lr_lfo_ratio < 1e-6 && lr_lfo_index < 1e-6)) {
//             std::cout << "Converged after " << iter << " iterations" << std::endl;
//             break;
//         }
//     }
//
//     // Run a random restart if correlation is low
//     if (best_correlation < 0.5) {
//         std::cout << "Correlation is low, trying random restart" << std::endl;
//
//         // Random starting points for better exploration
//         std::vector<std::tuple<double, double, double>> starting_points = {
//             {0.2, 0.02, 0.3},
//             {0.7, 0.07, 0.7},
//             {0.1, 0.01, 0.9},
//             {0.9, 0.09, 0.1}
//         };
//
//         for (const auto& [start_mod, start_ratio, start_lfo_idx] : starting_points) {
//             double random_mod_index = start_mod;
//             double random_lfo_ratio = start_ratio;
//             double random_lfo_index = start_lfo_idx;
//             double random_lfo_freq = carrier_freq * random_lfo_ratio;
//
//             double random_correlation = evaluateCorrelation(
//                 residual, N, carrier_freq, carrier_phase, carrier_amp,
//                 random_mod_index, random_lfo_freq, random_lfo_index);
//
//             if (random_correlation > best_correlation) {
//                 best_correlation = random_correlation;
//                 best_mod_index = random_mod_index;
//                 best_lfo_freq = random_lfo_freq;
//                 best_lfo_index = random_lfo_index;
//             }
//         }
//     }
//
//     std::cout << "Found optimal combined parameters: mod_index=" << best_mod_index
//              << ", lfo_freq=" << best_lfo_freq
//              << ", lfo_index=" << best_lfo_index
//              << ", correlation=" << best_correlation << std::endl;
//
//     return {best_mod_index, best_lfo_freq, best_lfo_index};
// }

// Helper function to evaluate correlation for a specific parameter set
double CleanDFT::evaluateCorrelation(
    const std::vector<Complex> &residual, size_t N,
    double carrier_freq, double carrier_phase, double carrier_amp,
    double mod_index, double lfo_freq, double lfo_index) {
    try {
        // Generate the FM spectrum
        std::vector<Complex> fm_spectrum(residual.size(), Complex(0, 0));

        // Add contribution of both modulators
        addCombinedModulators(
            fm_spectrum, N, carrier_freq, carrier_phase, carrier_amp,
            mod_index, lfo_freq, lfo_index);

        // Calculate correlation
        Complex numerator = 0;
        double norm_residual = 0;
        double norm_fm = 0;

        for (size_t i = 1; i < residual.size(); i++) {
            numerator += residual[i] * std::conj(fm_spectrum[i]);
            norm_residual += std::norm(residual[i]);
            norm_fm += std::norm(fm_spectrum[i]);
        }

        if (norm_residual < 1e-10 || norm_fm < 1e-10) {
            return 0.0;
        }

        return std::abs(numerator) / (std::sqrt(norm_residual) * std::sqrt(norm_fm));
    } catch (const std::exception &e) {
        std::cerr << "Error evaluating correlation: " << e.what() << std::endl;
        return 0.0;
    }
}

// Compute numerical gradient for a parameter
double CleanDFT::computeGradient(
    const std::vector<Complex> &residual, size_t N,
    double carrier_freq, double carrier_phase, double carrier_amp,
    double mod_index, double lfo_freq, double lfo_index,
    const std::string &param_name) {
    // Step size for numerical differentiation
    double epsilon = 0.01;

    double base_correlation = evaluateCorrelation(
        residual, N, carrier_freq, carrier_phase, carrier_amp,
        mod_index, lfo_freq, lfo_index);

    double perturbed_correlation;

    if (param_name == "mod_index") {
        perturbed_correlation = evaluateCorrelation(
            residual, N, carrier_freq, carrier_phase, carrier_amp,
            mod_index + epsilon, lfo_freq, lfo_index);
    } else if (param_name == "lfo_ratio") {
        perturbed_correlation = evaluateCorrelation(
            residual, N, carrier_freq, carrier_phase, carrier_amp,
            mod_index, lfo_freq + carrier_freq * epsilon, lfo_index);
    } else if (param_name == "lfo_index") {
        perturbed_correlation = evaluateCorrelation(
            residual, N, carrier_freq, carrier_phase, carrier_amp,
            mod_index, lfo_freq, lfo_index + epsilon);
    } else {
        throw std::invalid_argument("Unknown parameter name");
    }

    return (perturbed_correlation - base_correlation) / epsilon;
}

std::tuple<double, double, double, double> CleanDFT::findOptimalFMParameters(
    double carrier_freq, double carrier_phase, double carrier_amp,
    const std::vector<Complex> &residual, size_t N) {
    // Initial parameters
    double mod_index = 0.000001;
    // double lfo_ratio = 0.001;
    double lfo_index = 1.0;
    double lfo_freq = 1.0; // carrier_freq * lfo_ratio;

    // Bounds
    const double min_mod_index = 0.0, max_mod_index = 1.43;
    // const double min_lfo_ratio = 0.0, max_lfo_ratio = 0.005;
    const double min_lfo_freq = 0.01, max_lfo_freq = 2.0;
    const double min_lfo_index = 0.01, max_lfo_index = 1.4347; //2.13;

    double best_correlation = evaluateCorrelation(
        residual, N, carrier_freq, carrier_phase, carrier_amp,
        mod_index, lfo_freq, lfo_index);

    std::cout << "Initial correlation: " << best_correlation << std::endl;

    // Coordinate descent

    // Optimize lfo_ratio
    auto lfo_freq_objective = [&](double freq) {
        return evaluateCorrelation(
            residual, N, carrier_freq, carrier_phase, carrier_amp,
            mod_index, freq, lfo_index);
    };

    lfo_freq = goldenSectionSearch(lfo_freq_objective, min_lfo_freq, max_lfo_freq);
    // lfo_freq = carrier_freq * lfo_ratio;

    std::cout << "Best LFO modulator: " << lfo_freq << std::endl;

    // Optimize lfo_index
    auto lfo_index_objective = [&](double lfo_idx) {
        return evaluateCorrelation(
            residual, N, carrier_freq, carrier_phase, carrier_amp,
            mod_index, lfo_freq, lfo_idx);
    };

    lfo_index = goldenSectionSearch(lfo_index_objective, min_lfo_index, max_lfo_index);
    std::cout << "Best LFO index: " << lfo_index << std::endl;

    // Optimize mod_index
    // auto mod_index_objective = [&](double mod_idx) {
    //     return evaluateCorrelation(
    //         residual, N, carrier_freq, carrier_phase, carrier_amp,
    //         mod_idx, lfo_freq, lfo_index);
    // };
    //
    // mod_index = goldenSectionSearch(mod_index_objective, min_mod_index, max_mod_index);

    std::cout << "Best Mod Index: " << mod_index << std::endl;


    // Update best correlation
    best_correlation = evaluateCorrelation(
        residual, N, carrier_freq, carrier_phase, carrier_amp,
        mod_index, lfo_freq, lfo_index);

    std::cout << "Best Correlation: " << best_correlation << std::endl;

    std::vector<Complex> fm_spectrum(residual.size(), Complex(0, 0));
    addCombinedModulators(
        fm_spectrum, N, carrier_freq, carrier_phase, carrier_amp,
        mod_index, lfo_freq, lfo_index);

    // Calculate the optimal scalar using projection
    Complex numerator = 0;
    double denominator = 0;

    for (size_t i = 1; i < residual.size(); i++) {
        numerator += residual[i] * std::conj(fm_spectrum[i]);
        denominator += std::norm(fm_spectrum[i]);
    }

    // Optimal scalar is the projection coefficient
    double optimal_scalar = std::abs(numerator) / denominator;
    std::cout << "Optimal Scalar: " << optimal_scalar << std::endl;

    std::cout << "Found optimal parameters: mod_index=" << mod_index
            << ", lfo_freq=" << lfo_freq
            << ", lfo_index=" << lfo_index
            << ", optimal_scalar=" << optimal_scalar
            << ", correlation=" << best_correlation << std::endl;

    return {mod_index, lfo_freq, lfo_index, optimal_scalar};
}


// Helper function to add combined contribution of both modulators
void CleanDFT::addCombinedModulators(
    std::vector<Complex> &spectrum, size_t N,
    double carrier_freq, double carrier_phase, double carrier_amp,
    double mod_index, double lfo_freq, double lfo_index) {
    // Add harmonic modulator sidebands (if active)
    if (mod_index > 0.0000001) {
        int max_sideband = std::min(10, static_cast<int>(mod_index * 2 + 5));
        if (mod_index < 0.0001) {
            max_sideband = 0;
        }
        // const int max_sideband = 20;
        for (int n = -max_sideband; n <= max_sideband; n++) {
            bool flip = false;
            // Calculate harmonic sideband frequency
            double harmonic_freq = carrier_freq + n * carrier_freq;

            if (harmonic_freq <= 0) {
                // flip = true;
                // harmonic_freq = -harmonic_freq;
                continue;
            }

            // Skip if outside valid range
            if (harmonic_freq >= static_cast<double>(N) / 2) continue;

            // Calculate harmonic sideband amplitude
            double harmonic_amp = besselJ(std::abs(n), mod_index);

            if (n < 0 && n % 2 != 0) harmonic_amp = -harmonic_amp;
            if (!flip) harmonic_amp = std::abs(harmonic_amp);

            if (RAISED_BESSEL) {
                harmonic_amp = (harmonic_amp + RAISED_BESSEL_CONSTANT) / (1 + RAISED_BESSEL_CONSTANT);
            }

            // Calculate harmonic sideband phase
            double harmonic_phase = carrier_phase;

            // If LFO is also active, add LFO sidebands around each harmonic sideband
            if (lfo_index > 0 && lfo_freq > 0) {
                const int max_lfo_sideband = std::min(10, static_cast<int>(lfo_index * 3 + 3));

                for (int m = -max_lfo_sideband; m <= max_lfo_sideband; m++) {
                    // Calculate combined sideband frequency
                    bool flip = false;
                    double combined_freq = harmonic_freq + m * lfo_freq;

                    // Skip if outside valid range
                    if (combined_freq <= 0) {
                        // flip = true;
                        // combined_freq = -combined_freq;
                        continue;
                    }
                    if (combined_freq >= static_cast<double>(N) / 2) continue;

                    // Calculate LFO sideband amplitude
                    double lfo_amp = besselJ(std::abs(m), lfo_index);
                    if (m < 0 && m % 2 != 0) lfo_amp = -lfo_amp;
                    if (!flip) lfo_amp = std::abs(lfo_amp);

                    if (RAISED_BESSEL) {
                        lfo_amp = (lfo_amp + RAISED_BESSEL_CONSTANT) / (1 + RAISED_BESSEL_CONSTANT);
                    }


                    // Combined amplitude is product of carrier, harmonic, and LFO
                    double combined_amp = carrier_amp * harmonic_amp * lfo_amp;

                    // Combined phase includes both modulators
                    double combined_phase = harmonic_phase;

                    // Add this combined sideband
                    for (size_t i = 1; i < spectrum.size(); i++) {
                        double diff = combined_freq - static_cast<double>(i);
                        double corrected_phase = combined_phase + M_PI * diff / static_cast<double>(N);
                        spectrum[i] += combined_amp * DirichletKernel::getValueAtBin(diff, corrected_phase, N);

                        // Add mirroring across Nyquist
                        const size_t mirror_bin = N - i;
                        if (mirror_bin < spectrum.size() && mirror_bin > i) {
                            const double mirror_diff = combined_freq - static_cast<double>(mirror_bin);
                            const double mirror_phase = combined_phase + M_PI * mirror_diff / static_cast<double>(N);
                            // Mirror should be complex conjugate
                            spectrum[mirror_bin] += std::conj(
                                combined_amp * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
                        }
                    }
                }
            } else {
                // If LFO is not active, just add the harmonic sideband
                double sideband_amp = carrier_amp * harmonic_amp;

                for (size_t i = 1; i < spectrum.size(); i++) {
                    double diff = harmonic_freq - static_cast<double>(i);
                    double corrected_phase = harmonic_phase + M_PI * diff / static_cast<double>(N);
                    spectrum[i] += sideband_amp * DirichletKernel::getValueAtBin(diff, corrected_phase, N);

                    // Add mirroring across Nyquist
                    const size_t mirror_bin = N - i;
                    if (mirror_bin < spectrum.size() && mirror_bin > i) {
                        const double mirror_diff = harmonic_freq - static_cast<double>(mirror_bin);
                        const double mirror_phase = harmonic_phase + M_PI * mirror_diff / static_cast<double>(N);
                        // Mirror should be complex conjugate
                        spectrum[mirror_bin] += std::conj(
                            sideband_amp * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
                    }
                }
            }
        }
    } else if (lfo_index > 0.00001 && lfo_freq > 0.00001) {
        // If only LFO is active, add LFO sidebands around the carrier
        const int max_lfo_sideband = std::min(10, static_cast<int>(lfo_index * 3 + 3));

        for (int m = -max_lfo_sideband; m <= max_lfo_sideband; m++) {
            // Skip carrier
            if (m == 0) continue;

            // Calculate LFO sideband frequency
            double lfo_sideband_freq = carrier_freq + m * lfo_freq;

            // Skip if outside valid range
            if (lfo_sideband_freq <= 0 || lfo_sideband_freq >= static_cast<double>(N) / 2) continue;

            // Calculate LFO sideband amplitude
            double lfo_amp = besselJ(std::abs(m), lfo_index);
            if (m < 0 && m % 2 != 0) lfo_amp = -lfo_amp;
            lfo_amp = std::abs(lfo_amp);

            if (RAISED_BESSEL) {
                lfo_amp = (lfo_amp + RAISED_BESSEL_CONSTANT) / (1 + RAISED_BESSEL_CONSTANT);
            }

            double sideband_amp = carrier_amp * lfo_amp;

            // Calculate LFO sideband phase
            double lfo_phase = carrier_phase;

            // Add this LFO sideband
            for (size_t i = 1; i < spectrum.size(); i++) {
                double diff = lfo_sideband_freq - static_cast<double>(i);
                double corrected_phase = lfo_phase + M_PI * diff / static_cast<double>(N);
                spectrum[i] += sideband_amp * DirichletKernel::getValueAtBin(diff, corrected_phase, N);

                // Add mirroring across Nyquist
                const size_t mirror_bin = N - i;
                if (mirror_bin < spectrum.size() && mirror_bin > i) {
                    const double mirror_diff = lfo_sideband_freq - static_cast<double>(mirror_bin);
                    const double mirror_phase = lfo_phase + M_PI * mirror_diff / static_cast<double>(N);
                    // Mirror should be complex conjugate
                    spectrum[mirror_bin] += std::conj(
                        sideband_amp * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
                }
            }
        }
    }
}

void CleanDFT::subtractFMOperator(
    std::vector<Complex> &residual, size_t N,
    double carrier_freq, double carrier_phase, double carrier_amp,
    double mod_index, double lfo_freq, double lfo_index, double scalar) {
    // Create a temporary spectrum
    std::vector<Complex> spectrum(residual.size(), Complex(0, 0));

    // Add contribution of both modulators
    addCombinedModulators(
        spectrum, N, carrier_freq, carrier_phase, carrier_amp,
        mod_index, lfo_freq, lfo_index);

    // Apply the scalar to the entire FM operator
    for (size_t i = 1; i < spectrum.size(); i++) {
        spectrum[i] *= scalar;
    }

    // Subtract the spectrum from residual
    for (size_t i = 1; i < residual.size(); i++) {
        residual[i] -= spectrum[i];
    }
}

// Implementation for main deconvolveFMDirichletKernel method
std::vector<CleanDFT::FMComponent> CleanDFT::deconvolveFMDirichletKernel(
    const std::vector<Complex> &fft, const size_t sample_rate) {
    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;

    std::vector<FMComponent> fm_components;
    std::vector<Complex> residual(nyquist_bin);

    // Safety check
    if (fft.size() < nyquist_bin) {
        std::cerr << "FFT size too small" << std::endl;
        return fm_components;
    }

    // Copy only up to nyquist
    for (size_t i = 0; i < nyquist_bin; i++) {
        if (i < fft.size()) {
            residual[i] = fft[i];
        } else {
            residual[i] = Complex(0, 0);
        }
    }

    // Remove DC
    residual[0] = Complex(0, 0);

    // Calculate total energy
    double total_energy = 0.0;
    for (const auto &val: residual) {
        total_energy += std::norm(val);
    }

    std::cout << "Total energy: " << total_energy << std::endl;


    int iteration = 0;

    while (fm_components.size() < MAX_FM_COMPONENTS && iteration < MAX_FM_ITERATIONS) {
        iteration++;

        try {
            // Find peak bin
            int peak_bin = findPeakBin(residual, nyquist_bin);

            // Check exit conditions
            if (isExitConditionMet(peak_bin, nyquist_bin, residual)) {
                break;
            }

            // Ensure peak_bin is valid
            if (peak_bin <= 0 || peak_bin >= static_cast<int>(residual.size())) {
                std::cerr << "Invalid peak bin: " << peak_bin << std::endl;
                break;
            }

            // Find optimal frequency and phase for carrier
            const double carrier_freq = findOptimalFrequency(residual, peak_bin, peak_bin);
            const double carrier_phase = findOptimalPhase(residual, peak_bin, carrier_freq);
            const double carrier_amp = DirichletKernel::getAmplitudeAtBin(
                carrier_freq, std::abs(residual[peak_bin]), N, peak_bin);

            std::cout << "Found carrier: freq=" << carrier_freq
                    << ", phase=" << carrier_phase
                    << ", amp=" << carrier_amp << std::endl;

            // Find optimal FM parameters
            auto [best_mod_index, best_lfo_freq, best_lfo_index, optimal_coeff] =
                    findOptimalFMParameters(carrier_freq, carrier_phase, carrier_amp, residual, N);

            // Calculate correlation to evaluate quality
            double correlation = calculateCorrelation(
                residual, carrier_freq, carrier_phase, carrier_amp,
                best_mod_index, static_cast<double>(best_lfo_freq), best_lfo_index, N);

            // Only add if correlation is decent
            if (correlation > 0.01) {
                // Store the FM component
                fm_components.push_back({
                    carrier_freq, carrier_phase, carrier_amp,
                    best_mod_index, best_lfo_freq, best_lfo_index
                });

                // Subtract the entire FM operator from the residual
                subtractFMOperator(
                    residual, N, carrier_freq, carrier_phase, carrier_amp,
                    best_mod_index, best_lfo_freq, best_lfo_index, optimal_coeff);

                // Print progress
                printProgress(fm_components.size(), carrier_freq, peak_bin, carrier_phase,
                              best_mod_index, best_lfo_freq, best_lfo_index, residual, total_energy);

                // Check if we've captured enough energy
                double residual_energy = 0.0;
                for (const auto &val: residual) {
                    residual_energy += std::norm(val);
                }

                if (residual_energy < 0.001 * total_energy) {
                    std::cout << "Converged: residual energy below threshold" << std::endl;
                    break;
                }
            } else {
                std::cout << "Skipping component with low correlation: " << correlation << std::endl;

                // Still subtract a basic component to make progress
                for (size_t i = 1; i < residual.size(); i++) {
                    double diff = carrier_freq - static_cast<double>(i);
                    double corrected_phase = carrier_phase + M_PI * diff / static_cast<double>(N);
                    residual[i] -= carrier_amp * DirichletKernel::getValueAtBin(diff, corrected_phase, N);
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Error in iteration " << iteration << ": " << e.what() << std::endl;
            // Continue to next iteration
            continue;
        }
    }

    if (iteration >= MAX_FM_ITERATIONS) {
        std::cout << "Stopped after maximum iterations" << std::endl;
    }

    return fm_components;
}

// Implementation for writeFMComponentsToCSV
void CleanDFT::writeFMComponentsToCSV(const std::vector<FMComponent> &components,
                                      size_t N, size_t sample_rate, const std::string &filename) {
    std::ofstream file(filename);
    file << "carrier_freq_hz,phase,amplitude,mod_index,lfo_freq_hz,lfo_index\n";

    for (const auto &comp: components) {
        // Convert frequencies from bins to Hz
        double freq_hz = comp.carrier_freq * sample_rate / N;
        double lfo_freq_hz = comp.lfo_freq * sample_rate / N;

        file << freq_hz << ","
                << comp.phase << ","
                << comp.amplitude << ","
                << comp.mod_index << ","
                << lfo_freq_hz << ","
                << comp.lfo_index << "\n";
    }

    file.close();
    std::cout << "Saved FM components to " << filename << std::endl;
}

std::vector<double> CleanDFT::decompressFMComponents(const std::vector<FMComponent> &components,
                                                     size_t N) {
    // Create spectrum in the frequency domain
    std::vector<Complex> spectrum(N, Complex(0, 0));

    // Calculate DC component
    spectrum[0] = computeFMDC(components, N);

    // For each FM component
    for (const auto &comp: components) {
        // Add contribution of both modulators for this component
        addCombinedModulators(
            spectrum, N, comp.carrier_freq, comp.phase, comp.amplitude,
            comp.mod_index, comp.lfo_freq, comp.lfo_index);
    }

    // Handle symmetry
    for (size_t m = 1; m <= N / 2; m++) {
        spectrum[N - m] = std::conj(spectrum[m]);
    }

    for (size_t m = 1; m <= N / 2; m++) {
        spectrum[m] = std::conj(spectrum[N - m]);
    }

    // Convert to time domain
    auto reconstructed = WaveProcessor::computeIFFT(spectrum);

    return reconstructed;
}

Complex CleanDFT::computeFMDC(const std::vector<FMComponent> &components, const size_t N) {
    Complex dc = 0;

    for (const auto &comp: components) {
        // Create a temporary spectrum
        std::vector<Complex> temp_spectrum(1, Complex(0, 0));

        // Add contribution of both modulators but to DC only
        double carrier_freq = comp.carrier_freq;
        double carrier_phase = comp.phase;
        double carrier_amp = comp.amplitude;
        double mod_index = comp.mod_index;
        double lfo_freq = comp.lfo_freq;
        double lfo_index = comp.lfo_index;

        // First, consider harmonic modulator
        if (mod_index > 0.01) {
            const int max_sideband = std::min(20, static_cast<int>(mod_index * 2 + 5));

            for (int n = -max_sideband; n <= max_sideband; n++) {
                double harmonic_freq = carrier_freq + n * carrier_freq;
                double harmonic_amp = besselJ(std::abs(n), mod_index);
                if (n < 0 && n % 2 != 0) harmonic_amp = -harmonic_amp;
                harmonic_amp = std::abs(harmonic_amp);
                double harmonic_phase = carrier_phase + n * M_PI / 2.0;

                // If LFO active, consider combined sidebands
                if (lfo_index > 0.01 && lfo_freq > 0.001) {
                    const int max_lfo_sideband = std::min(10, static_cast<int>(lfo_index * 3 + 3));

                    for (int m = -max_lfo_sideband; m <= max_lfo_sideband; m++) {
                        double combined_freq = harmonic_freq + m * lfo_freq;
                        double lfo_amp = besselJ(std::abs(m), lfo_index);
                        if (m < 0 && m % 2 != 0) lfo_amp = -lfo_amp;
                        lfo_amp = std::abs(lfo_amp);

                        double combined_amp = carrier_amp * harmonic_amp * lfo_amp;
                        double combined_phase = harmonic_phase + m * M_PI / 4.0;

                        // Compute how this sideband contributes to DC
                        dc += combined_amp * DirichletKernel::getValueAtBin(
                            -combined_freq, combined_phase, N
                        );
                    }
                } else {
                    // Just harmonic sideband
                    double sideband_amp = carrier_amp * harmonic_amp;

                    dc += sideband_amp * DirichletKernel::getValueAtBin(
                        -harmonic_freq, harmonic_phase, N
                    );
                }
            }
        } else if (lfo_index > 0.01 && lfo_freq > 0.001) {
            // Only LFO active
            const int max_lfo_sideband = std::min(10, static_cast<int>(lfo_index * 3 + 3));

            for (int m = -max_lfo_sideband; m <= max_lfo_sideband; m++) {
                if (m == 0) continue;

                double lfo_sideband_freq = carrier_freq + m * lfo_freq;
                double lfo_amp = besselJ(std::abs(m), lfo_index);
                if (m < 0 && m % 2 != 0) lfo_amp = -lfo_amp;
                lfo_amp = std::abs(lfo_amp);
                double sideband_amp = carrier_amp * lfo_amp;
                double lfo_phase = carrier_phase + m * M_PI / 4.0;

                dc += sideband_amp * DirichletKernel::getValueAtBin(
                    -lfo_sideband_freq, lfo_phase, N
                );
            }
        }
    }

    return dc;
}
