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
        // new_dc += amplitude * DirichletKernel::getValueAtBin(
        //     -true_frequency,
        //     true_phase,
        //     N
        // );

        // new_dc += amplitude * DirichletKernel::getValueAtBin(
        //     -N - true_frequency,
        //     true_phase + M,
        //     N
        // );
        const double diff = -true_frequency;
        const double phase = true_phase + M_PI * diff / static_cast<double>(N);
        new_dc += amplitude * DirichletKernel::getValueAtBin(diff, phase, N);

        const double mirror_diff = - (N - true_frequency);
        const double mirror_phase = true_phase + M_PI * mirror_diff / static_cast<double>(N);

        // Mirror should be complex conjugate
        new_dc += std::conj(amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
    }
    Complex ret = Complex(new_dc.real(), 0);
    return ret;
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
                                                                     const size_t sample_rate, const int maxcomp) {
    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;
    MAX_COMPONENTS = maxcomp;
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
        const double retention = residual_energy / total_energy;

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
        // size_t mirror_bin = N - i;
        // if (mirror_bin < N) {
        //     double mirror_diff = component.true_frequency - static_cast<double>(mirror_bin);
        //     double mirror_phase = component.true_phase + M_PI * mirror_diff / static_cast<double>(N);
        //     // Mirror should be complex conjugate
        //     residual[i] -= std::conj(component.amplitude *
        //                              DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
        //
        //
        // }
        const size_t mirror_bin = N - i;
        const double mirror_diff = component.true_frequency - static_cast<double>(mirror_bin);
        const double mirror_phase = component.true_phase + M_PI * mirror_diff / static_cast<double>(N);
        // Mirror should be complex conjugate
        residual[i] -= std::conj(component.amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
    }
}


// Check for band overlap and coverage
bool validateBands(const std::vector<FrequencyBand>& bands, size_t nyquist_bin) {
    // Create a vector to track which bins are covered
    std::vector<bool> bin_covered(nyquist_bin, false);
    std::vector<bool> bin_overlap(nyquist_bin, false);

    // Check each band
    for (const auto& band : bands) {
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

std::vector<FrequencyBand> divideSpectrumByThresholds(
    const std::vector<Complex>& spectrum,
    size_t num_bands,
    size_t nyquist_bin)
{
    // Find the maximum magnitude in the spectrum
    double max_magnitude = 0.0;
    for (size_t i = 1; i < nyquist_bin; i++) {
        max_magnitude = std::max(max_magnitude, std::abs(spectrum[i]));
    }

    // Create threshold levels for band division
    std::vector<double> thresholds;
    double min_threshold = max_magnitude * 0.001; // Lower limit to avoid noise

    // Generate logarithmically spaced thresholds
    for (size_t i = 0; i < num_bands; i++) {
        double threshold = max_magnitude * std::pow(min_threshold / max_magnitude,
                                                 static_cast<double>(i) / (num_bands - 1));
        thresholds.push_back(threshold);
    }

    // Sort spectrum bins by magnitude
    std::vector<std::pair<size_t, double>> bin_magnitudes;
    bin_magnitudes.reserve(nyquist_bin - 1);

    for (size_t i = 1; i < nyquist_bin; i++) {
        bin_magnitudes.emplace_back(i, std::abs(spectrum[i]));
    }

    // Sort in descending order of magnitude
    std::sort(bin_magnitudes.begin(), bin_magnitudes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Distribute bins into bands based on magnitude
    std::vector<std::vector<size_t>> band_bins(num_bands);

    for (const auto& [bin, magnitude] : bin_magnitudes) {
        // Find appropriate band for this bin
        size_t band_idx = 0;
        while (band_idx < thresholds.size() && magnitude < thresholds[band_idx]) {
            band_idx++;
        }

        if (band_idx >= num_bands) {
            band_idx = num_bands - 1; // Place in the last band if below all thresholds
        }

        band_bins[band_idx].push_back(bin);
    }

    // Create contiguous frequency bands by merging nearby bins
    std::vector<FrequencyBand> bands;

    for (size_t band_idx = 0; band_idx < num_bands; band_idx++) {
        if (band_bins[band_idx].empty()) {
            continue; // Skip empty bands
        }

        // Sort bins in ascending order
        std::sort(band_bins[band_idx].begin(), band_bins[band_idx].end());

        // Identify contiguous segments
        size_t current_start = band_bins[band_idx][0];
        size_t current_end = current_start;

        for (size_t i = 1; i < band_bins[band_idx].size(); i++) {
            if (band_bins[band_idx][i] == current_end + 1) {
                // Contiguous bin, extend the current segment
                current_end = band_bins[band_idx][i];
            } else {
                // Gap in bins, create a new band for the completed segment
                FrequencyBand band;
                band.start_bin = current_start;
                band.end_bin = current_end + 1; // end_bin is exclusive
                bands.push_back(band);

                // Start a new segment
                current_start = band_bins[band_idx][i];
                current_end = current_start;
            }
        }

        // Add the final segment
        FrequencyBand band;
        band.start_bin = current_start;
        band.end_bin = current_end + 1; // end_bin is exclusive
        bands.push_back(band);
    }

    // Merge very small bands if needed (optimization)
    if (bands.size() > num_bands * 2) {
        std::vector<FrequencyBand> merged_bands;
        merged_bands.reserve(num_bands);

        // Merge based on band width
        std::sort(bands.begin(), bands.end(),
                 [](const FrequencyBand& a, const FrequencyBand& b) {
                     return (a.end_bin - a.start_bin) < (b.end_bin - b.start_bin);
                 });

        // Keep the largest num_bands bands
        for (size_t i = bands.size() - num_bands; i < bands.size(); i++) {
            merged_bands.push_back(bands[i]);
        }

        // Sort by start_bin for consistency
        std::sort(merged_bands.begin(), merged_bands.end(),
                 [](const FrequencyBand& a, const FrequencyBand& b) {
                     return a.start_bin < b.start_bin;
                 });

        bands = merged_bands;
    }

    return bands;
}

// Function to divide spectrum into bands with approximately equal energy content
std::vector<FrequencyBand> divideSpectrumByEnergy(
    const std::vector<Complex>& spectrum,
    size_t num_bands,
    size_t nyquist_bin)
{
    // Calculate the total energy in the spectrum (skipping DC)
    double total_energy = 0.0;
    for (size_t i = 1; i < nyquist_bin; i++) {
        total_energy += std::norm(spectrum[i]);
    }

    // Target energy per band
    double target_energy_per_band = total_energy / num_bands;

    // Create bands
    std::vector<FrequencyBand> bands;
    bands.reserve(num_bands);

    size_t current_start = 1; // Skip DC bin
    double accumulated_energy = 0.0;

    for (size_t band_idx = 0; band_idx < num_bands - 1; band_idx++) {
        double band_energy = 0.0;
        size_t bin = current_start;

        // Accumulate bins until we reach the target energy for this band
        while (bin < nyquist_bin && band_energy < target_energy_per_band) {
            band_energy += std::norm(spectrum[bin]);
            bin++;
        }

        // Create the band
        FrequencyBand band;
        band.start_bin = current_start;
        band.end_bin = bin;
        bands.push_back(band);

        // Update for the next band
        current_start = bin;
        accumulated_energy += band_energy;
    }

    // Add the final band (to avoid any rounding issues)
    FrequencyBand final_band;
    final_band.start_bin = current_start;
    final_band.end_bin = nyquist_bin;
    bands.push_back(final_band);

    if (!validateBands(bands, nyquist_bin)) {
        std::cerr << "Error in energy-based band division, falling back to uniform" << std::endl;
        // Return uniform bands as fallback
        std::vector<FrequencyBand> uniform_bands(num_bands);
        const size_t bins_per_band = nyquist_bin / num_bands;
        for (size_t i = 0; i < num_bands; i++) {
            uniform_bands[i].start_bin = i * bins_per_band + 1;  // Skip DC
            uniform_bands[i].end_bin = (i == num_bands - 1) ? nyquist_bin : (i + 1) * bins_per_band;
        }
        return uniform_bands;
    }

    return bands;
}

// Enhanced version that accounts for perceptual factors and magnitude distribution
std::vector<FrequencyBand> divideSpectrumAdaptively(
    const std::vector<Complex>& spectrum,
    size_t num_bands,
    size_t nyquist_bin,
    size_t sample_rate)
{
    // Calculate weighted energy profile across the spectrum
    std::vector<double> bin_weights(nyquist_bin);
    double total_weighted_energy = 0.0;

    for (size_t i = 1; i < nyquist_bin; i++) {
        double magnitude = std::abs(spectrum[i]);

        // Apply frequency-dependent weighting
        // Higher frequencies get boosted to counter natural magnitude decay
        double freq_hz = i * static_cast<double>(sample_rate) / (2.0 * nyquist_bin);

        // Logarithmic weighting based on frequency perception
        // This helps preserve high-frequency components
        double freq_weight = 1.0 + 0.6 * std::log10(1.0 + freq_hz / 1000.0);

        // Calculate weighted energy
        bin_weights[i] = magnitude * magnitude * freq_weight;
        total_weighted_energy += bin_weights[i];
    }

    // Target weighted energy per band
    double target_per_band = total_weighted_energy / num_bands;

    // Create bands with adaptive boundaries
    std::vector<FrequencyBand> bands;
    bands.reserve(num_bands);

    size_t current_start = 1; // Skip DC bin
    double accumulated_energy = 0.0;

    for (size_t band_idx = 0; band_idx < num_bands - 1; band_idx++) {
        double band_energy = 0.0;
        size_t bin = current_start;

        // Accumulate bins until we reach the target energy for this band
        while (bin < nyquist_bin && band_energy < target_per_band) {
            band_energy += bin_weights[bin];
            bin++;
        }

        // Create the band
        FrequencyBand band;
        band.start_bin = current_start;
        band.end_bin = bin;
        bands.push_back(band);

        // Update for the next band
        current_start = bin;
        accumulated_energy += band_energy;
    }

    // Add the final band (to avoid any rounding issues)
    FrequencyBand final_band;
    final_band.start_bin = current_start;
    final_band.end_bin = nyquist_bin;
    bands.push_back(final_band);

    // Debug output to verify band distribution
    std::cout << "Adaptive band division:" << std::endl;
    for (size_t i = 0; i < bands.size(); i++) {
        double band_width = bands[i].end_bin - bands[i].start_bin;
        double min_freq = bands[i].start_bin * sample_rate / (2.0 * nyquist_bin);
        double max_freq = bands[i].end_bin * sample_rate / (2.0 * nyquist_bin);

        std::cout << "Band " << i
                  << ": bins " << bands[i].start_bin << "-" << bands[i].end_bin
                  << " (width: " << band_width << " bins, "
                  << min_freq << "-" << max_freq << " Hz)" << std::endl;
    }

    return bands;
}

std::vector<CleanDFT::Component> CleanDFT::deconvolveMultiPassNonIterative(
    const std::vector<Complex>& fft,
    const size_t sample_rate,
    const size_t num_threads,
    const int maxcomp) {

    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;
    MAX_COMPONENTS = maxcomp;

    // Initialize residual with original spectrum
    std::vector<Complex> residual(nyquist_bin);
    std::copy_n(fft.begin(), nyquist_bin, residual.begin());
    residual[0] = Complex(0, 0); // Remove DC

    // Calculate total energy
    double total_energy = 0.0;
    for (const auto& val : residual) {
        total_energy += std::norm(val);
    }

    // Vector to hold all components across passes
    std::vector<Component> all_components;

    // Multiple passes to catch different frequencies
    const int NUM_PASSES = 3; // Adjust based on your needs

    for (int pass = 0; pass < NUM_PASSES; pass++) {
        std::cout << "Starting pass " << (pass+1) << "/" << NUM_PASSES << std::endl;

        // Find peaks in current residual
        std::vector<PeakInfo> pass_peaks;

        // Find local maxima in current residual
        for (size_t i = 1; i < nyquist_bin; i++) {
            double magnitude = std::abs(residual[i]);

            if (i > 1 && i < nyquist_bin - 1 &&
                magnitude > std::abs(residual[i-1]) &&
                magnitude > std::abs(residual[i+1]) &&
                magnitude > 1e-6) {

                PeakInfo peak;
                peak.bin = i;
                peak.magnitude = magnitude;
                peak.frequency = i; // Initial guess
                peak.phase = std::arg(residual[i]);
                pass_peaks.push_back(peak);
            }
        }

        // If no peaks found, add systematic sampling for later passes
        if (pass > 0 && pass_peaks.size() < maxcomp / NUM_PASSES) {
            size_t step = nyquist_bin / (maxcomp / NUM_PASSES);
            if (step < 1) step = 1;

            for (size_t i = 1; i < nyquist_bin; i += step) {
                if (std::abs(residual[i]) > 1e-7) {
                    PeakInfo peak;
                    peak.bin = i;
                    peak.magnitude = std::abs(residual[i]);
                    peak.frequency = i;
                    peak.phase = std::arg(residual[i]);
                    pass_peaks.push_back(peak);
                }
            }
        }

        // Sort by magnitude
        std::sort(pass_peaks.begin(), pass_peaks.end(),
            [](const PeakInfo& a, const PeakInfo& b) {
                return a.magnitude > b.magnitude;
            }
        );

        // Limit number of peaks per pass
        size_t max_peaks_per_pass = maxcomp / NUM_PASSES;
        if (pass_peaks.size() > max_peaks_per_pass) {
            pass_peaks.resize(max_peaks_per_pass);
        }

        std::cout << "Processing " << pass_peaks.size() << " peaks in pass " << (pass+1) << std::endl;

        // Process these peaks in parallel
        std::vector<Component> pass_components;
        std::mutex components_mutex;

        // Calculate peaks per thread
        size_t peaks_per_thread = (pass_peaks.size() + num_threads - 1) / num_threads;
        std::vector<std::thread> threads;

        for (size_t t = 0; t < num_threads; t++) {
            size_t start_idx = t * peaks_per_thread;
            size_t end_idx = std::min((t + 1) * peaks_per_thread, pass_peaks.size());

            if (start_idx >= pass_peaks.size()) continue;

            threads.emplace_back([&, start_idx, end_idx]() {
                std::vector<Component> thread_components;

                for (size_t i = start_idx; i < end_idx; i++) {
                    const PeakInfo& peak = pass_peaks[i];

                    // Skip if too close to boundaries
                    if (peak.bin < 2 || peak.bin > nyquist_bin - 2) continue;

                    try {
                        // Find optimal frequency using the current residual
                        double true_freq = findOptimalFrequency(residual, peak.bin, peak.bin);

                        // Find optimal phase
                        double true_phase = findOptimalPhase(residual, peak.bin, true_freq);

                        // Determine true amplitude
                        double amplitude = DirichletKernel::getAmplitudeAtBin(
                            true_freq, std::abs(residual[peak.bin]), N, peak.bin);

                        // Only add if significant
                        if (amplitude > 1e-7) {
                            thread_components.push_back({true_freq, true_phase, amplitude});
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error processing peak: " << e.what() << std::endl;
                    }
                }

                // Add to pass components
                {
                    std::lock_guard<std::mutex> lock(components_mutex);
                    pass_components.insert(pass_components.end(),
                                          thread_components.begin(),
                                          thread_components.end());
                }
            });
        }

        // Wait for threads to complete
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }

        // Sort components by amplitude
        std::sort(pass_components.begin(), pass_components.end(),
            [](const Component& a, const Component& b) {
                return a.amplitude > b.amplitude;
            }
        );

        // Add components to master list
        all_components.insert(all_components.end(),
                            pass_components.begin(),
                            pass_components.end());

        // Update residual with these components
        for (const auto& comp : pass_components) {
            for (size_t i = 1; i < nyquist_bin; i++) {
                double diff = comp.true_frequency - static_cast<double>(i);
                double corrected_phase = comp.true_phase + M_PI * diff / static_cast<double>(N);
                residual[i] -= comp.amplitude * DirichletKernel::getValueAtBin(
                    diff, corrected_phase, N);

                // Handle mirroring
                const size_t mirror_bin = N - i;
                if (mirror_bin < N) {
                    const double mirror_diff = comp.true_frequency - static_cast<double>(mirror_bin);
                    const double mirror_phase = comp.true_phase + M_PI * mirror_diff / static_cast<double>(N);
                    residual[mirror_bin] -= std::conj(comp.amplitude *
                        DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
                }
            }
        }

        // Calculate residual energy
        double residual_energy = 0.0;
        for (const auto& val : residual) {
            residual_energy += std::norm(val);
        }

        double retention = residual_energy / total_energy;
        std::cout << "After pass " << (pass+1) << ": "
                  << pass_components.size() << " components extracted, "
                  << retention * 100 << "% energy remains" << std::endl;

        // Early exit if good convergence
        if (retention < 0.001 || pass_components.empty()) {
            std::cout << "Early exit after pass " << (pass+1) << std::endl;
            break;
        }
    }

    // Filter out redundant components
    std::vector<Component> filtered_components;

    // Sort all components by amplitude
    std::sort(all_components.begin(), all_components.end(),
        [](const Component& a, const Component& b) {
            return a.amplitude > b.amplitude;
        }
    );

    // Keep only the most significant ones
    if (all_components.size() > MAX_COMPONENTS) {
        all_components.resize(MAX_COMPONENTS);
    }

    std::cout << "Final component count: " << all_components.size() << std::endl;

    return all_components;
}

std::vector<FrequencyBand> divideSpectrumHybrid(
    const std::vector<Complex>& spectrum,
    size_t num_bands,
    size_t nyquist_bin,
    size_t sample_rate) {

    // First, calculate how many bands to allocate to each method
    size_t perceptual_bands = num_bands / 2;
    size_t uniform_bands = num_bands - perceptual_bands;

    std::cout << "Creating hybrid division with " << perceptual_bands
              << " perceptual bands and " << uniform_bands
              << " uniform bands" << std::endl;

    // Create the perceptual bands focusing on critical listening ranges
    std::vector<FrequencyBand> bands;

    // Define critical frequency boundaries in Hz
    // These correspond approximately to auditory critical bands
    const std::vector<double> critical_boundaries = {
        20, 100, 200, 300, 400, 510, 630, 770, 920, // Low range
        1080, 1270, 1480, 1720 //2000, 2320, 2700, 3150, // Mid range (perceptually important)
        //3700, 4400 //, 5300, 6400, 7700, 9500, 12000, 15500 // High range
    };

    // Ensure we don't go beyond Nyquist
    double nyquist_freq = sample_rate / 2.0;

    // Map critical boundaries to bins
    std::vector<size_t> critical_bins;
    for (double freq : critical_boundaries) {
        if (freq < nyquist_freq) {
            size_t bin = static_cast<size_t>(freq * nyquist_bin / nyquist_freq);
            if (bin < nyquist_bin && bin >= 1) {
                critical_bins.push_back(bin);
            }
        }
    }

    // Add final bin
    critical_bins.push_back(nyquist_bin);

    // If we have too many critical bands, merge some
    while (critical_bins.size() > perceptual_bands + 1) {
        // Find the smallest width band to merge
        size_t min_width_idx = 0;
        size_t min_width = nyquist_bin;

        for (size_t i = 0; i < critical_bins.size() - 1; i++) {
            size_t width = critical_bins[i+1] - critical_bins[i];
            if (width < min_width) {
                min_width = width;
                min_width_idx = i;
            }
        }

        // Merge by removing the boundary
        critical_bins.erase(critical_bins.begin() + min_width_idx + 1);
    }

    // If we have too few critical bands, interpolate
    while (critical_bins.size() < perceptual_bands + 1) {
        // Find the largest width band to split
        size_t max_width_idx = 0;
        size_t max_width = 0;

        for (size_t i = 0; i < critical_bins.size() - 1; i++) {
            size_t width = critical_bins[i+1] - critical_bins[i];
            if (width > max_width) {
                max_width = width;
                max_width_idx = i;
            }
        }

        // Split this band
        size_t split_point = (critical_bins[max_width_idx] + critical_bins[max_width_idx+1]) / 2;
        critical_bins.insert(critical_bins.begin() + max_width_idx + 1, split_point);
    }

    // Create the perceptual bands
    for (size_t i = 0; i < perceptual_bands; i++) {
        FrequencyBand band;
        band.start_bin = critical_bins[i];
        band.end_bin = critical_bins[i+1];
        bands.push_back(band);
    }

    // Now create uniform/magnitude-weighted bands
    // Compute energy in unallocated regions
    std::vector<bool> bin_allocated(nyquist_bin, false);

    // Mark the bins that are already allocated
    for (const auto& band : bands) {
        for (size_t i = band.start_bin; i < band.end_bin; i++) {
            if (i < bin_allocated.size()) {
                bin_allocated[i] = true;
            }
        }
    }

    // Create a list of unallocated bins with their magnitudes
    std::vector<std::pair<size_t, double>> unallocated_bins;
    for (size_t i = 1; i < nyquist_bin; i++) {
        if (!bin_allocated[i]) {
            unallocated_bins.push_back({i, std::abs(spectrum[i])});
        }
    }

    // Sort by magnitude (descending) for the magnitude-weighted approach
    std::sort(unallocated_bins.begin(), unallocated_bins.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Divide unallocated bins into groups based on magnitude
    size_t bins_per_uniform_band = unallocated_bins.size() / uniform_bands;
    if (bins_per_uniform_band < 1) bins_per_uniform_band = 1;

    std::vector<size_t> remaining_boundaries;
    for (size_t i = 0; i < unallocated_bins.size(); i += bins_per_uniform_band) {
        remaining_boundaries.push_back(unallocated_bins[i].first);
    }

    // Sort boundaries for consistent band structure
    std::sort(remaining_boundaries.begin(), remaining_boundaries.end());

    // Create the uniform/magnitude bands
    for (size_t i = 0; i < remaining_boundaries.size() - 1; i++) {
        FrequencyBand band;
        band.start_bin = remaining_boundaries[i];
        band.end_bin = remaining_boundaries[i+1];

        // Only add if it's a valid band
        if (band.start_bin < band.end_bin &&
            band.start_bin >= 1 &&
            band.end_bin <= nyquist_bin) {
            bands.push_back(band);
        }

        if (bands.size() >= num_bands) break;
    }

    // Final check: ensure we cover the entire spectrum
    std::sort(bands.begin(), bands.end(),
              [](const FrequencyBand& a, const FrequencyBand& b) {
                  return a.start_bin < b.start_bin;
              });

    // Fix any gaps
    for (size_t i = 0; i < bands.size() - 1; i++) {
        if (bands[i].end_bin != bands[i+1].start_bin) {
            bands[i].end_bin = bands[i+1].start_bin;
        }
    }

    // Special handling for the first and last band
    if (!bands.empty()) {
        bands[0].start_bin = 1;  // Always start at bin 1 (skipping DC)
        bands.back().end_bin = nyquist_bin;  // End at Nyquist
    }

    // Calculate frequency ranges for reporting
    std::cout << "Final band division:" << std::endl;
    for (size_t i = 0; i < bands.size(); i++) {
        double min_freq = bands[i].start_bin * sample_rate / (2.0 * nyquist_bin);
        double max_freq = bands[i].end_bin * sample_rate / (2.0 * nyquist_bin);

        std::cout << "Band " << i << ": "
                  << bands[i].start_bin << "-" << bands[i].end_bin
                  << " (" << min_freq << "-" << max_freq << " Hz)";

        // Indicate which method created this band
        if (i < perceptual_bands) {
            std::cout << " [Perceptual]";
        } else {
            std::cout << " [Magnitude]";
        }
        std::cout << std::endl;
    }

    return bands;
}


// Main parallel deconvolution function
std::vector<CleanDFT::Component> CleanDFT::deconvolveParallelDirichlet(
    const std::vector<Complex>& fft, const size_t sample_rate, const size_t num_threads, const int maxcomp) {


    MAX_COMPONENTS = maxcomp;
    std::cout << maxcomp << std::endl;
    std::cout << BATCH_SIZE << std::endl;
    if (maxcomp < BATCH_SIZE) {
        BATCH_SIZE = MAX_COMPONENTS;
    }
    std::cout << BATCH_SIZE << std::endl;
    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;

    std::vector<FrequencyBand> bands;
    const size_t num_bands = num_threads ;

    switch (SPECTRAL_DIVISION_METHOD) {
        case SpectralDivisionMethod::UNIFORM:
            // Original uniform division
            bands.resize(num_threads);
            {
                const size_t bins_per_band = nyquist_bin / num_threads;
                for (size_t i = 0; i < num_threads; i++) {
                    bands[i].start_bin = i * bins_per_band + 1;  // Skip DC
                    bands[i].end_bin = (i == num_threads - 1) ? nyquist_bin : (i + 1) * bins_per_band;
                }
            }
            std::cout << "Using uniform spectral division" << std::endl;
            break;
        case SpectralDivisionMethod::HYBRID:
            bands = divideSpectrumHybrid(fft, num_threads, nyquist_bin, sample_rate);
        std::cout << "Using hybrid spectral division" << std::endl;
        break;

        case SpectralDivisionMethod::EQUAL_ENERGY:
            bands = divideSpectrumByEnergy(fft, num_threads, nyquist_bin);
            std::cout << "Using equal energy spectral division" << std::endl;
            break;

        case SpectralDivisionMethod::PERCEPTUAL_WEIGHTING:
            bands = divideSpectrumAdaptively(fft, num_threads, nyquist_bin, sample_rate);
            std::cout << "Using perceptually weighted spectral division" << std::endl;
            break;

        case SpectralDivisionMethod::MAGNITUDE_THRESHOLD:
            bands = divideSpectrumByThresholds(fft, num_threads, nyquist_bin);
            std::cout << "Using magnitude threshold spectral division" << std::endl;
            break;

        default:
            // Fallback to uniform division
            bands.resize(num_threads);
            {
                const size_t bins_per_band = nyquist_bin / num_threads;
                for (size_t i = 0; i < num_threads; i++) {
                    bands[i].start_bin = i * bins_per_band + 1;  // Skip DC
                    bands[i].end_bin = (i == num_threads - 1) ? nyquist_bin : (i + 1) * bins_per_band;
                }
            }
            std::cout << "Using default uniform spectral division" << std::endl;
    }

    // Output band information for debugging
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

    // std::vector<FrequencyBand> bands(num_bands);
    //
    // const size_t bins_per_band = nyquist_bin / num_bands;
    // for (size_t i = 0; i < num_bands; i++) {
    //     bands[i].start_bin = i * bins_per_band + 1; // Skip DC
    //     bands[i].end_bin = (i == num_bands - 1) ? nyquist_bin : (i + 1) * bins_per_band;
    // }

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
    // BATCH_SIZE = 256;  // Reduced batch size for better L2 norm

    // Minimum spectral distance required between peaks in the same batch
    // Increased minimum distance for better component separation

    double min_spectral_distance =  1 / N;
    // if (fft.size() != sample_rate) {
    //     min_spectral_distance *= sample_rate;
    // }

    // Use a fixed thread approach without lambdas
    int component_limit = MAX_COMPONENTS;
    while (components.size() < component_limit) {
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
                if (selected_peaks.size() >= BATCH_SIZE) break;
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
            if (comp.amplitude > 1e-16) {
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
        const double retention = residual_energy / total_energy;
        std::cout << "% L2 norm retained: " << retention << std::endl;

        // Check convergence - using slightly more strict threshold
        if (residual_energy < 0.5e-6 * total_energy || filtered_components.empty()) {
            if (!filtered_components.empty()) {
                std::cout << filtered_components.size() << " components. ";
            }
            break;
        }
    }

    return components;
}

// Main parallel deconvolution function with accuracy metrics
std::vector<CleanDFT::Component> CleanDFT::deconvolveParallelDirichletWithMetrics(
    const std::vector<Complex>& fft, const size_t sample_rate, const size_t num_threads,
    const std::string& metrics_csv_file, const int maxcomp) {

    MAX_COMPONENTS = maxcomp;

    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;

    // Initialize CSV file for metrics
    std::ofstream metrics_file;
    if (!metrics_csv_file.empty()) {
        metrics_file.open(metrics_csv_file);
        // Write CSV header with cosine similarity added
        metrics_file << "batch,components,lsd,rmse,cosine_sim,naive_lsd,naive_rmse,naive_cosine_sim" << std::endl;
    }

    // ======== SPECTRAL BAND DIVISION (unchanged) ========
    std::vector<FrequencyBand> bands;
    const size_t num_bands = num_threads;

    switch (SPECTRAL_DIVISION_METHOD) {
        case SpectralDivisionMethod::UNIFORM:
            // Original uniform division
            bands.resize(num_threads);
            {
                const size_t bins_per_band = nyquist_bin / num_threads;
                for (size_t i = 0; i < num_threads; i++) {
                    bands[i].start_bin = i * bins_per_band + 1;  // Skip DC
                    bands[i].end_bin = (i == num_threads - 1) ? nyquist_bin : (i + 1) * bins_per_band;
                }
            }
            std::cout << "Using uniform spectral division" << std::endl;
            break;

        // [other cases remain unchanged]

        default:
            // Fallback to uniform division
            bands.resize(num_threads);
            {
                const size_t bins_per_band = nyquist_bin / num_threads;
                for (size_t i = 0; i < num_threads; i++) {
                    bands[i].start_bin = i * bins_per_band + 1;  // Skip DC
                    bands[i].end_bin = (i == num_threads - 1) ? nyquist_bin : (i + 1) * bins_per_band;
                }
            }
            std::cout << "Using default uniform spectral division" << std::endl;
    }

    // Debug output for bands (unchanged)
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
    std::vector<Complex> original(nyquist_bin);

    // Initialize residual
    std::copy_n(fft.begin(), nyquist_bin, residual.begin());
    std::copy_n(fft.begin(), nyquist_bin, original.begin());
    residual[0] *= 0.0; // Remove DC
    original[0] *= 0.0;

    // Calculate total energy
    double total_energy = 0.0;
    for (const auto& val : residual) {
        total_energy += std::norm(val);
    }

    std::cout << "Total energy: " << total_energy << std::endl;

    // Maximum number of components to find in one batch
    // BATCH_SIZE = 128;  // Reduced batch size for better L2 norm

    // Minimum spectral distance required between peaks in the same batch
    double min_spectral_distance =  2 / N;
    if (fft.size() != sample_rate) {
        min_spectral_distance *= sample_rate;
    }

    // Create a vector of (bin, magnitude) pairs for the baseline naive approach
    std::vector<std::pair<size_t, double>> magnitudes;
    std::vector<Complex> naive_spectrum;

    // If we're collecting metrics, prepare the naive baseline
    if (metrics_file.is_open()) {
        // Create a copy of the spectrum
        naive_spectrum = std::vector<Complex>(nyquist_bin, Complex(0, 0));

        // Create a vector of (bin, magnitude) pairs
        magnitudes.reserve(nyquist_bin);
        for (size_t i = 1; i < nyquist_bin; i++) {
            magnitudes.push_back({i, std::abs(original[i])});
        }

        // Sort by magnitude (descending)
        std::sort(magnitudes.begin(), magnitudes.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            }
        );
    }

    // Batch counter
    size_t batch_num = 0;

    // ======== MAIN DECONVOLUTION LOOP ========
    std::vector<Complex> current_spectrum(nyquist_bin, Complex(0, 0));
    while (components.size() < MAX_COMPONENTS) {
        batch_num++;

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
                if (selected_peaks.size() >= BATCH_SIZE) break;
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
            if (comp.amplitude > 1e-16) {
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

        // current_spectrum = std::vector<Complex>(nyquist_bin, Complex(0, 0));
        for (size_t i = 0; i < residual.size(); i++) {
            current_spectrum[i] = original[i] - residual[i];
        }


        // Calculate residual energy
        double residual_energy = 0.0;
        for (const auto& val : residual) {
            residual_energy += std::norm(val);
        }

        // Print progress
        std::cout << "Batch " << batch_num << " added " << filtered_components.size() << " components. ";
        std::cout << "Total components: " << components.size() << std::endl;
        const double retention = residual_energy / total_energy;
        std::cout << "% L2 norm retained: " << retention << std::endl;

        // === COMPUTE METRICS (if requested) ===
        if (metrics_file.is_open()) {

            // For the naive approach, take the top N components where N = components.size()
            std::vector<Complex> current_naive_spectrum(nyquist_bin, Complex(0, 0));

            if (components.size() <= magnitudes.size()) {
                // Use the top components.size() magnitude bins
                for (size_t i = 0; i < components.size(); i++) {
                    size_t bin = magnitudes[i].first;
                    current_naive_spectrum[bin] = original[bin];
                }
            }

            // Calculate metrics

            // 1. Log Spectral Distance (LSD)
            double lsd = 0.0;
            double naive_lsd = 0.0;

            // 2. Root Mean Squared Error (RMSE)
            double mse = 0.0;
            double naive_mse = 0.0;

            // 3. Shannon Information
            double shannon_info = 0.0;
            double naive_shannon_info = 0.0;

            // 4. Cosine Similarity
            double dot_product = 0.0;
            double orig_norm = 0.0;
            double recon_norm = 0.0;
            double naive_dot_product = 0.0;
            double naive_norm = 0.0;

            // Add helper function for log2 that handles negative values
            auto safe_log2 = [](double x) -> double {
                return x > 0 ? log2(x) : 0.0;
            };

            // Sum of squared magnitudes for normalization
            double total_orig_mag = 0.0;
            for (size_t i = 1; i < nyquist_bin; i++) {
                total_orig_mag += std::norm(original[i]);
            }
            if (total_orig_mag < 1e-10) total_orig_mag = 1.0;  // Avoid division by zero

            // Calculate metrics across frequency bins
            for (size_t i = 1; i < nyquist_bin; i++) {
                try {
                    // Original power in dB
                    double orig_power_db = log10(std::norm(original[i]) + 1e-10);

                    // Reconstructed power in dB
                    double recon_power_db = log10(std::norm(current_spectrum[i]) + 1e-10);
                    double naive_power_db = log10(std::norm(current_naive_spectrum[i]) + 1e-10);

                    // LSD calculation
                    double db_diff = orig_power_db - recon_power_db;
                    double naive_db_diff = orig_power_db - naive_power_db;

                    lsd += db_diff * db_diff;
                    naive_lsd += naive_db_diff * naive_db_diff;

                    // MSE calculation
                    Complex diff = original[i] - current_spectrum[i];
                    Complex naive_diff = original[i] - current_naive_spectrum[i];

                    mse += std::norm(diff);
                    naive_mse += std::norm(naive_diff);

                    // Cosine similarity calculation
                    // Calculate dot product and norms
                    dot_product += std::real(original[i] * std::conj(current_spectrum[i]));
                    orig_norm += std::norm(original[i]);
                    recon_norm += std::norm(current_spectrum[i]);

                    naive_dot_product += std::real(original[i] * std::conj(current_naive_spectrum[i]));
                    naive_norm += std::norm(current_naive_spectrum[i]);

                    // Shannon information - use normalized magnitudes as probabilities
                    double orig_mag = std::norm(original[i]) / total_orig_mag;
                    double recon_mag = std::norm(current_spectrum[i]) / total_orig_mag;
                    double naive_mag = std::norm(current_naive_spectrum[i]) / total_orig_mag;

                    // Calculate information content (entropy) - higher is better
                    if (orig_mag > 1e-10) {
                        naive_shannon_info -= orig_mag * safe_log2(orig_mag);
                    }

                    if (recon_mag > 1e-10) {
                        shannon_info -= recon_mag * safe_log2(recon_mag);
                    }
                }
                catch (std::exception& e) {
                    std::cerr << "Error calculating metrics at bin " << i << ": " << e.what() << std::endl;
                    // Continue with next bin
                }
            }

            // Finalize metrics
            lsd = sqrt(lsd / (nyquist_bin - 1));
            naive_lsd = sqrt(naive_lsd / (nyquist_bin - 1));

            mse /= (nyquist_bin - 1);
            naive_mse /= (nyquist_bin - 1);

            mse = std::sqrt(mse);
            naive_mse = std::sqrt(naive_mse);


            // Finalize cosine similarity
            double cosine_sim = 0.0;
            double naive_cosine_sim = 0.0;

            if (orig_norm > 0 && recon_norm > 0) {
                cosine_sim = dot_product / (sqrt(orig_norm) * sqrt(recon_norm));
                // Ensure it's between -1 and 1 due to potential numerical issues
                cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));
            }

            if (orig_norm > 0 && naive_norm > 0) {
                naive_cosine_sim = naive_dot_product / (sqrt(orig_norm) * sqrt(naive_norm));
                naive_cosine_sim = std::max(-1.0, std::min(1.0, naive_cosine_sim));
            }

            // Write metrics to CSV
            metrics_file << batch_num << ","
                      << components.size() << ","
                      << lsd << ","
                      << mse << ","
                      // << shannon_info << ","
                      << cosine_sim << ","
                      << naive_lsd << ","
                      << naive_mse << ","
                      // << naive_shannon_info << ","
                      << naive_cosine_sim << std::endl;
        }

        // Check convergence - using slightly more strict threshold
        if (residual_energy < 0.5e-6 * total_energy) {
            break;
        }
    }

    // Close metrics file if open
    if (metrics_file.is_open()) {
        metrics_file.close();
    }

    return components;
}


std::vector<double> CleanDFT::decompressComponents(const std::vector<Component> &components,
                                                   const size_t N, Complex DC, const double max_magnitude) {
    std::vector spectrum(N, Complex(0, 0));





    // Phase correction to ensure periodicity
    int i = 0;
    size_t length = components.size();
    for (const auto &[true_frequency, true_phase, amplitude]: components) {
        i++;
        std::cout << "Decompressing component: " << i << " / " << length << std::endl;
        for (size_t m = 1; m < N; m++) {

            const double diff = true_frequency - static_cast<double>(m);
            // Add phase correction term to ensure periodicity
            const double corrected_phase = true_phase + M_PI * diff / static_cast<double>(N);
            spectrum[m] += amplitude * DirichletKernel::getValueAtBin(
                diff, corrected_phase, N
            );

            const size_t mirror_bin = N - m;
            const double mirror_diff = true_frequency - static_cast<double>(mirror_bin);
            const double mirror_phase = true_phase + M_PI * mirror_diff / static_cast<double>(N);
            // Mirror should be complex conjugate
            spectrum[m] += std::conj(amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));

        }
    }

    // Handle symmetry

    // for (size_t m = 1; m <= N / 2; m++) {
    //     spectrum[N - m] += std::conj(spectrum[m]); // * Complex(0, 1);
    // }

    // // Compensate for tail reflections across Nyquist
    // for (size_t m = 1; m <= N / 2; m++) {
    //     spectrum[m] = std::conj(spectrum[N - m]); // * Complex(0, 1);
    // }

    spectrum[0] = DC;

    writeSpectrumToCSV(spectrum, "spectrum_before_ifft.csv");

    auto reconstructed = WaveProcessor::computeIFFT(spectrum);

    return reconstructed;
}

// Define a worker function for component decompression
std::vector<Complex> decompressComponentsWorker(
    const std::vector<CleanDFT::Component>& components,
    size_t N,
    size_t start_idx,
    size_t end_idx)
{
    std::vector<Complex> local_spectrum(N, Complex(0, 0));

    for (size_t i = start_idx; i < end_idx; i++) {
        const auto& component = components[i];
        double true_frequency = component.true_frequency;
        double true_phase = component.true_phase;
        double amplitude = component.amplitude;

        for (size_t m = 1; m < N; m++) {
            const double diff = true_frequency - static_cast<double>(m);
            const double corrected_phase = true_phase + M_PI * diff / static_cast<double>(N);
            local_spectrum[m] += amplitude * DirichletKernel::getValueAtBin(
                diff, corrected_phase, N
            );

            const size_t mirror_bin = N - m;
            const double mirror_diff = true_frequency - static_cast<double>(mirror_bin);
            const double mirror_phase = true_phase + M_PI * mirror_diff / static_cast<double>(N);
            // Mirror should be complex conjugate
            local_spectrum[m] += std::conj(amplitude * DirichletKernel::getValueAtBin(mirror_diff, mirror_phase, N));
        }
    }

    return local_spectrum;
}

// Main decompression function using threads
std::vector<double> CleanDFT::decompressComponentsParallel(
    const std::vector<Component> &components,
    const size_t N,
    const Complex DC,
    const double max_magnitude)
{
    // Initialize spectrum vector
    std::vector<Complex> spectrum(N, Complex(0, 0));

    // Calculate DC component
    spectrum[0] = DC;

    // Determine thread count
    size_t num_threads = std::thread::hardware_concurrency();
    size_t components_per_thread = std::max(size_t(1), components.size() / num_threads);

    // Create threads
    std::vector<std::thread> threads;
    std::vector<std::vector<Complex>> thread_results(num_threads);

    for (size_t t = 0; t < num_threads; t++) {
        size_t start_idx = t * components_per_thread;
        size_t end_idx = (t == num_threads - 1) ? components.size() : (t + 1) * components_per_thread;

        // Skip if this thread would have no components
        if (start_idx >= components.size()) continue;

        // Launch thread
        threads.emplace_back(
            [t, &thread_results, &components, N, start_idx, end_idx]() {
                thread_results[t] = decompressComponentsWorker(components, N, start_idx, end_idx);
            }
        );
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Combine results from all threads
    for (const auto& local_spectrum : thread_results) {
        if (!local_spectrum.empty()) {
            for (size_t m = 1; m < N; m++) {
                spectrum[m] += local_spectrum[m];
            }
        }
    }

    // // Handle symmetry
    // for (size_t m = 1; m <= N / 2; m++) {
    //     spectrum[N - m] += std::conj(spectrum[m]);
    // }
    //
    // // Compensate for tail reflections
    // for (size_t m = 1; m <= N / 2; m++) {
    //     spectrum[m] = std::conj(spectrum[N - m]);
    // }

    writeSpectrumToCSV(spectrum, "spectrum_before_ifft.csv");

    // Compute inverse FFT
    auto reconstructed = WaveProcessor::computeIFFT(spectrum);

    return reconstructed;
}

std::vector<double> CleanDFT::decompressComponentsEnhanced(
    const std::vector<Component> &components,
    const size_t N,
    Complex DC,
    const double max_magnitude) {

    std::vector<Complex> spectrum(N, Complex(0, 0));

    // Step 1: Sort components by frequency for better handling of phase relationships
    std::vector<Component> sorted_components = components;
    std::sort(sorted_components.begin(), sorted_components.end(),
        [](const Component& a, const Component& b) {
            return a.true_frequency < b.true_frequency;
        }
    );

    // Step 2: Group components by frequency proximity for phase coordination
    std::vector<std::vector<Component>> frequency_groups;
    if (!sorted_components.empty()) {
        std::vector<Component> current_group = {sorted_components[0]};
        double group_center_freq = sorted_components[0].true_frequency;

        for (size_t i = 1; i < sorted_components.size(); i++) {
            // If frequency is within 0.5 bin of group center, add to group
            if (std::abs(sorted_components[i].true_frequency - group_center_freq) < 0.5) {
                current_group.push_back(sorted_components[i]);
            } else {
                // Start a new group
                frequency_groups.push_back(current_group);
                current_group = {sorted_components[i]};
                group_center_freq = sorted_components[i].true_frequency;
            }
        }

        // Add the last group
        if (!current_group.empty()) {
            frequency_groups.push_back(current_group);
        }
    }

    std::cout << "Grouped components into " << frequency_groups.size() << " frequency bands" << std::endl;

    // Step 3: Process each frequency group
    for (const auto& group : frequency_groups) {
        // For each group, compute the center frequency and average phase
        double avg_freq = 0.0;
        double avg_phase = 0.0;
        double total_amp = 0.0;

        for (const auto& comp : group) {
            avg_freq += comp.true_frequency * comp.amplitude;
            avg_phase += comp.true_phase * comp.amplitude;
            total_amp += comp.amplitude;
        }

        if (total_amp > 0) {
            avg_freq /= total_amp;
            avg_phase /= total_amp;

            // Apply this combined component to the spectrum
            for (size_t m = 1; m < N; m++) {
                const double diff = avg_freq - static_cast<double>(m);
                const double corrected_phase = avg_phase + M_PI * diff / static_cast<double>(N);

                Complex bin_contribution = total_amp * DirichletKernel::getValueAtBin(
                    diff, corrected_phase, N
                );

                spectrum[m] += bin_contribution;
            }
        }
    }

    // Step 4: Handle DC component more carefully
    // Compute DC from components to ensure consistency
    Complex computed_DC = Complex(0, 0);
    for (const auto& comp : components) {
        // Contribution to DC bin
        double diff = -comp.true_frequency;  // Negated for DC bin (0)
        double corrected_phase = comp.true_phase + M_PI * diff / static_cast<double>(N);
        computed_DC += comp.amplitude * DirichletKernel::getValueAtBin(diff, corrected_phase, N);
    }

    // Use a weighted blend of original and computed DC
    spectrum[0] = Complex(0.7 * DC.real() + 0.3 * computed_DC.real(), 0);

    // Step 5: Ensure proper symmetry across Nyquist
    for (size_t m = 1; m <= N/2; m++) {
        spectrum[N-m] = std::conj(spectrum[m]);
    }

    // Save spectrum for debugging
    writeSpectrumToCSV(spectrum, "enhanced_spectrum_before_ifft.csv");

    // Step 6: Apply IFFT with improved phase handling
    auto reconstructed = WaveProcessor::computeIFFT(spectrum);

    // // Step 7: Apply a subtle low-frequency correction (if needed)
    // if (N > 100) {  // Only for reasonable-sized FFTs
    //     // Apply a gentle high-pass filter to reduce any low-frequency rumble
    //     const size_t cutoff_bin = 3;  // Adjust based on your audio content
    //     for (size_t i = 0; i < reconstructed.size(); i++) {
    //         double window = 1.0 - std::exp(-static_cast<double>(i) / (N / cutoff_bin));
    //         reconstructed[i] *= window;
    //     }
    // }

    return reconstructed;
}

double CleanDFT::refinePhaseWithResidualAnalysis(
    const std::vector<Complex>& spectrum,
    double frequency,
    double initial_phase,
    double amplitude,
    int center_bin,
    size_t N) {

    // Step 1: Define local window for analysis
    const int window_size = 5;
    std::vector<int> bins;
    std::vector<Complex> local_spectrum;

    for (int i = center_bin - window_size; i <= center_bin + window_size; i++) {
        if (i > 0 && i < spectrum.size()) {
            bins.push_back(i);
            local_spectrum.push_back(spectrum[i]);
        }
    }

    if (bins.empty()) return initial_phase; // Safety check

    // Step 2: Project the component with our initial estimate
    std::vector<Complex> kernel_projection;
    for (int bin : bins) {
        double diff = frequency - static_cast<double>(bin);
        double corrected_phase = initial_phase + M_PI * diff / static_cast<double>(N);
        kernel_projection.push_back(amplitude * DirichletKernel::getValueAtBin(
            diff, corrected_phase, N));
    }

    // Step 3: Calculate local residual after removing our component
    std::vector<Complex> local_residual;
    for (size_t i = 0; i < local_spectrum.size(); i++) {
        local_residual.push_back(local_spectrum[i] - kernel_projection[i]);
    }

    // Step 4: Analyze how the residual would affect our phase estimate

    // Calculate how much the residual affects the center bin
    Complex center_residual = Complex(0, 0);
    for (size_t i = 0; i < bins.size(); i++) {
        if (bins[i] == center_bin) {
            center_residual = local_residual[i];
            break;
        }
    }

    // Calculate how much the residual affects adjacent bins
    Complex left_residual = Complex(0, 0);
    Complex right_residual = Complex(0, 0);

    for (size_t i = 0; i < bins.size(); i++) {
        if (bins[i] == center_bin - 1) left_residual = local_residual[i];
        if (bins[i] == center_bin + 1) right_residual = local_residual[i];
    }

    // Step 5: Determine how the residual distorts the phase

    // Original component at center bin
    Complex original_component = Complex(0, 0);
    for (size_t i = 0; i < bins.size(); i++) {
        if (bins[i] == center_bin) {
            original_component = kernel_projection[i];
            break;
        }
    }

    // Calculate how center residual would shift our phase
    double phase_distortion = 0.0;
    if (std::abs(original_component) > 0) {
        // How much the residual would rotate the component
        Complex combined = original_component + center_residual;
        phase_distortion = std::arg(combined) - std::arg(original_component);
    }

    // // Weights based on the relative confidence in our correction
    // double confidence_weight = std::abs(original_component) /
    //                          (std::abs(original_component) + std::abs(center_residual));
    //
    // // Apply phase correction, weighted by confidence
    // double phase_correction = phase_distortion * confidence_weight;
    double refined_phase = initial_phase - phase_distortion;

    // Normalize phase to [-π, π]
    while (refined_phase > M_PI) refined_phase -= 2 * M_PI;
    while (refined_phase <= -M_PI) refined_phase += 2 * M_PI;

    return refined_phase;
}
std::vector<CleanDFT::Component> CleanDFT::deconvolveNonIterativeParallel(
    const std::vector<Complex>& fft,
    const size_t sample_rate,
    const size_t num_threads,
    const int maxcomp) {

    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;
    MAX_COMPONENTS = maxcomp;

    // Step 1: Find all significant peaks across the spectrum
    std::vector<PeakInfo> all_peaks;
    for (size_t i = 1; i < nyquist_bin; i++) {
        double magnitude = std::abs(fft[i]);

        // Simple peak detection - local maximum with meaningful magnitude
        if (i > 1 && i < nyquist_bin - 1 &&
            magnitude > std::abs(fft[i-1]) &&
            magnitude > std::abs(fft[i+1])) {

            PeakInfo peak{};
            peak.bin = i;
            peak.magnitude = magnitude;
            peak.frequency = i; // Initial guess
            peak.phase = std::arg(fft[i]);
            all_peaks.push_back(peak);
        }
    }

    // Sort peaks by magnitude (descending)
    std::sort(all_peaks.begin(), all_peaks.end(),
        [](const PeakInfo& a, const PeakInfo& b) {
            return a.magnitude > b.magnitude;
        }
    );

    std::cout << "Unfiltered Found " << all_peaks.size() << " significant peaks" << std::endl;

    // Limit total number of peaks to process
    if (all_peaks.size() > MAX_COMPONENTS) {
        all_peaks.resize(MAX_COMPONENTS);
    }

    std::cout << "Found " << all_peaks.size() << " significant peaks" << std::endl;

    // Step 2: Process these peaks in parallel
    std::vector<Component> all_components;
    std::mutex components_mutex; // For safely appending components

    // Create thread pool
    std::vector<std::thread> threads;

    // Calculate peaks per thread
    size_t peaks_per_thread = (all_peaks.size() + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start_idx = t * peaks_per_thread;
        size_t end_idx = std::min((t + 1) * peaks_per_thread, all_peaks.size());

        if (start_idx >= all_peaks.size()) continue;

        threads.emplace_back([&, start_idx, end_idx]() {
            std::vector<Component> thread_components;

            for (size_t i = start_idx; i < end_idx; i++) {
                const PeakInfo& peak = all_peaks[i];

                // Skip if peak frequency is too close to 0 or Nyquist
                if (peak.bin < 2 || peak.bin > nyquist_bin - 2) continue;

                // Process this peak
                try {
                    // Find optimal frequency using golden section search
                    double true_freq = findOptimalFrequency(fft, peak.bin, peak.bin);

                    // Find optimal phase
                    double true_phase = findOptimalPhase(fft, peak.bin, true_freq);

                    // Determine true amplitude
                    double amplitude = DirichletKernel::getAmplitudeAtBin(
                        true_freq, std::abs(fft[peak.bin]), N, peak.bin);

                    // Phase refinement with local residual analysis
                    // double refined_phase = refinePhaseWithResidualAnalysis(
                    //     fft, true_freq, true_phase, amplitude, peak.bin, N);

                    // Only add component if amplitude is significant
                    if (amplitude > 1e-7) {
                        thread_components.push_back({true_freq, true_phase, amplitude});
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing peak at bin " << peak.bin
                              << ": " << e.what() << std::endl;
                }
            }

            // Add this thread's components to the global list
            std::lock_guard<std::mutex> lock(components_mutex);
            all_components.insert(all_components.end(),
                                 thread_components.begin(),
                                 thread_components.end());
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        if (thread.joinable()) thread.join();
    }

    std::cout << "Extracted " << all_components.size() << " components in parallel" << std::endl;

    // Optional: Sort components by amplitude for better compression
    std::sort(all_components.begin(), all_components.end(),
        [](const Component& a, const Component& b) {
            return a.amplitude > b.amplitude;
        }
    );

    return all_components;
}

std::vector<double> CleanDFT::resample(const std::vector<Component> &components,
                                       const size_t N,
                                       const size_t original_sr,
                                       const size_t target_sr,
                                       const Complex DC,
                                       const double max_magnitude) {
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




    auto reconstructed = decompressComponents(valid_components, new_N, DC, max_magnitude);
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
    double mod_index = DEFAULT_HARMONIC_MOD_INDEX;
    double lfo_index = DEFAULT_LFO_INDEX;
    double lfo_freq = DEFAULT_LFO_FREQ; // carrier_freq * lfo_ratio;

    // Bounds
    const double min_lfo_freq = MIN_LFO_FREQ, max_lfo_freq = MAX_LFO_FREQ;
    const double min_lfo_index = MIN_LFO_INDEX, max_lfo_index = MAX_LFO_INDEX; //2.13;

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
