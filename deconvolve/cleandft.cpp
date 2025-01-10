//
// Created by Matthew McQuistion on 12/17/24.
//

#include "cleandft.h"
#include <iostream>


CleanDFT::CleanDFT() = default;

double CleanDFT::computeCorrelation(const std::vector<Complex> &data_slice, const std::vector<Complex> &kernel,
                                    const bool isPhase) {
    Complex numerator = 0;
    for (size_t i = 0; i < data_slice.size(); i++) {
        numerator += data_slice[i] * std::conj(kernel[i]);
    }

    double data_norm = 0;
    double kernel_norm = 0;
    for (size_t i = 0; i < data_slice.size(); i++) {
        data_norm += std::norm(data_slice[i]);
        kernel_norm += std::norm(kernel[i]);
    }

    if (isPhase) {
        // For phase correlation, we want the real part to distinguish between phases
        return std::real(numerator) / (std::sqrt(data_norm) * std::sqrt(kernel_norm));
    }
    // For frequency correlation, we want the magnitude
    return std::abs(numerator) / (std::sqrt(data_norm) * std::sqrt(kernel_norm));
}

double CleanDFT::findOptimalPhase(const std::vector<Complex> &fft, const int center_bin, const double frequency) {
    // Create bins array relative to center bin
    std::vector<int> bins;
    for (int i = center_bin - PHASE_WINDOW_SIZE;
         i <= center_bin + PHASE_WINDOW_SIZE; i++) {
        if (i >= 0 && i < fft.size() / 2) {
            bins.push_back(i);
        }
         }

    // Get data slice using these bins
    std::vector<Complex> data_slice;
    data_slice.reserve(bins.size());
    for (const int bin: bins) {
        data_slice.push_back(fft[bin]);
    }

    // Phase search interval
    double a = -2.0 * M_PI;
    double b = 2.0 * M_PI;

    // Initial points
    double c = b - (b - a) / PHI;
    double d = a + (b - a) / PHI;

    // Function to evaluate correlation at a phase
    auto evaluate = [&](const double phase) {
        const std::vector<Complex> kernel = DirichletKernel::generateKernel(
            bins, frequency, phase, fft.size(), std::abs(fft[center_bin])
        );
        return computeCorrelation(data_slice, kernel, true);
    };

    // Golden section search
    constexpr double tolerance = 1e-7;
    double fc = evaluate(c);
    double fd = evaluate(d);

    while (std::abs(b - a) > tolerance) {
        if (fc > fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / PHI;
            fc = evaluate(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / PHI;
            fd = evaluate(d);
        }
    }

    return (a + b) / 2.0;  // Return midpoint of final interval
}
double CleanDFT::findOptimalFrequency(const std::vector<Complex> &fft, const int center_bin,
                                     const double test_frequency) {
    // Create bins array relative to center bin
    std::vector<int> bins;
    for (int i = center_bin - FREQUENCY_WINDOW_SIZE;
         i <= center_bin + FREQUENCY_WINDOW_SIZE; i++) {
        if (i >= 0 && i < fft.size() / 2) {
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

    // Frequency search interval
    double a = test_frequency - 0.5;
    double b = test_frequency + 0.5;

    // Initial points
    double c = b - (b - a) / PHI;
    double d = a + (b - a) / PHI;

    // Function to evaluate correlation at a frequency
    auto evaluate = [&](const double freq) {
        const std::vector<Complex> kernel = DirichletKernel::generateKernel(
            bins, freq, center_phase, fft.size(), std::abs(fft[center_bin])
        );
        return computeCorrelation(data_slice, kernel, false);
    };

    // Golden section search
    constexpr double tolerance = 1e-7;
    double fc = evaluate(c);
    double fd = evaluate(d);

    while (std::abs(b - a) > tolerance) {
        if (fc > fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / PHI;
            fc = evaluate(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / PHI;
            fd = evaluate(d);
        }
    }

    return (a + b) / 2.0;  // Return midpoint of final interval
}

std::vector<CleanDFT::Component> CleanDFT::deconvolveDirichletKernel(const std::vector<Complex> &fft,
                                                                     const size_t sample_rate) {
    const size_t N = fft.size();
    const size_t nyquist_bin = N / 2;

    std::vector<Component> components;
    std::vector<Complex> residual(nyquist_bin);
    std::copy_n(fft.begin(), nyquist_bin, residual.begin());

    // Calculate total energy
    double total_energy = 0.0;
    for (const auto &val: residual) {
        total_energy += std::norm(val);
    }

    while (components.size() < MAX_COMPONENTS) {
        // Find peak bin
        int peak_bin = 0;
        double max_magnitude = 0.0;
        for (int i = 0; i < nyquist_bin; i++) {
            if (const double magnitude = std::abs(residual[i]);
                magnitude > max_magnitude && i < static_cast<double>(nyquist_bin) * 0.9) {
                // Limit to 90% of Nyquist
                max_magnitude = magnitude;
                peak_bin = i;
            }
        }

        // Add sanity check
        if (static_cast<double>(peak_bin) > static_cast<double>(nyquist_bin) * 0.9 || max_magnitude < 1e-6) {
            std::cout << "Peak bin too high or magnitude too low, stopping..." << std::endl;
            break;
        }

        // Find optimal frequency
        const double true_freq = findOptimalFrequency(residual, peak_bin, peak_bin);
        const double true_phase = findOptimalPhase(residual, peak_bin, true_freq);
        // Calculate amplitude
        const double amplitude =
                DirichletKernel::getAmplitudeAtBin(true_freq, std::abs(residual[peak_bin]), N, peak_bin);


        // Generate subtraction kernel
        std::vector<int> kernel_bins;
        for (int m = std::max(0, peak_bin - ITER_WINDOW_SIZE);
             m < std::min(static_cast<int>(nyquist_bin), peak_bin + ITER_WINDOW_SIZE + 1); m++) {
            kernel_bins.push_back(m);
        }

        std::vector<Complex> subtract = DirichletKernel::generateKernel(
            kernel_bins,
            true_freq,
            true_phase,
            N,
            std::abs(residual[peak_bin])
        );

        components.push_back({true_freq, true_phase, amplitude});


        // Update residual
        for (size_t i = 0; i < kernel_bins.size(); i++) {
            residual[kernel_bins[i]] -= amplitude * subtract[i];
        }

        // Calculate residual energy
        double residual_energy = 0.0;
        for (const auto &val: residual) {
            residual_energy += std::norm(val);
        }

        // Print progress
        std::cout << "Component " << components.size() << ": freq="
                << true_freq * static_cast<double>(sample_rate) / static_cast<double>(N) << ", phase=" << true_phase <<
                std::endl;
        const double retention = residual_energy / total_energy * 100;
        std::cout << "% L2 norm retained: " << retention << std::endl;
        // std::cout << "% Energy norm retained: " << std::pow(retention,2) / 100 << std::endl;

        if (constexpr double THRESHOLD = 1e-6; residual_energy < THRESHOLD * total_energy) {
            break;
        }
    }

    return components;
}
