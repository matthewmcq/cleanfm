//
// Created by Matthew McQuistion on 12/17/24.
//

#include "cleandft.h"

#include <fstream>
#include <iostream>


CleanDFT::CleanDFT() = default;

Complex CleanDFT::computeDC(const std::vector<Component>& components, const size_t N) {
    Complex new_dc = 0;
    for (const auto&[true_frequency, true_phase, amplitude] : components) {
        // Compute how each component contributes to bin 0
        new_dc += amplitude * DirichletKernel::getValueAtBin(
            -true_frequency,
            true_phase,
            N
        );
    }
    return new_dc;
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

    auto evaluate = [&](const double phase) {
        const std::vector<Complex> kernel = DirichletKernel::generateKernel(
            bins, frequency, phase, fft.size() , std::abs(fft[center_bin])
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

    // std::vector<Complex> residual_check(nyquist_bin);
    std::copy_n(fft.begin(), nyquist_bin, residual.begin());

    residual[0] *= 0.0; // remove DC



    // std::copy_n(fft.begin(), nyquist_bin, residual_check.begin());

    // Calculate total energy
    double total_energy = 0.0;
    for (const auto &val: residual) {
        total_energy += std::norm(val);
    }
    double prev_retention = 0.0;
    std::cout << "total energy: " << total_energy << std::endl;

    while (components.size() < MAX_COMPONENTS) {
        // Find peak bin
        int peak_bin = 0;
        double max_magnitude = 0.0;
        for (int i = 1; i < nyquist_bin; i++) {
            // start @ 1 to skip DC since we deal w it separately
            const double magnitude = std::abs(residual[i]);
            if
                (magnitude > max_magnitude && i < static_cast<double>(nyquist_bin) * 0.49) {
                // ^^^^Not sure why we need 1/4 Nyquist -- TODO: FIX LATER!!!!!
                // Limit to 90% of Nyquist
                max_magnitude = magnitude;
                peak_bin = i;

            }
        }

        // Add sanity check
        if (static_cast<double>(peak_bin) > static_cast<double>(nyquist_bin) * 0.49 || max_magnitude < 1e-6) {
            std::cout << "Peak bin too high or magnitude too low, stopping..." << std::endl;
            break;
        }

        // remove peak bin from residual_check
        // residual_check[peak_bin] -= residual[peak_bin];

        // Find optimal frequency
        const double true_freq = findOptimalFrequency(residual, peak_bin, peak_bin);
        // std::cout << peak_bin * static_cast<double>(sample_rate) / static_cast<double>(N) << std::endl;
        // std::cout << true_freq * static_cast<double>(sample_rate) / static_cast<double>(N) << std::endl;
        const double true_phase =  findOptimalPhase(residual, peak_bin, true_freq);

        const double amplitude = DirichletKernel::getAmplitudeAtBin(true_freq, std::abs(residual[peak_bin]), N, peak_bin);

        // Calculate amplitude



        // Generate subtraction kernel
        // std::vector<int> kernel_bins;
        // for (int m = std::max(0, peak_bin - ITER_WINDOW_SIZE);
        //      m < std::min(static_cast<int>(nyquist_bin), peak_bin + ITER_WINDOW_SIZE + 1); m++) {
        //     kernel_bins.push_back(m);
        // }
        // std::vector<int> kernel_bins;
        // kernel_bins.reserve(nyquist_bin);
        // for (int m = 0; m < nyquist_bin; m++) {
        //     kernel_bins.push_back(m);
        // }



        // std::vector<Complex> subtract = DirichletKernel::generateKernel(
        //     kernel_bins,
        //     true_freq,
        //     true_phase,
        //     N,
        //     std::abs(residual[peak_bin])
        // );

        // Debug before subtraction:
        // std::cout << "At peak bin " << peak_bin << ":\n";
        // for (int i = 0; i < kernel_bins.size(); i++) {
        //     std::cout << "Bin " << kernel_bins[i]
        //               << " - Actual mag: " << std::abs(residual[kernel_bins[i]])
        //               << " Kernel mag: " << std::abs(subtract[i] * amplitude) << "\n"
        //                 << " - Actual phase: " << std::arg(residual[kernel_bins[i]])
        //               << " Kernel phase: " << std::arg(subtract[i] * amplitude) << "\n";
        // }

        components.push_back({true_freq, true_phase, amplitude});


        // Update residual
        for (size_t i = 1; i < residual.size(); i++) {
            double diff = true_freq - static_cast<double>(i);
            residual[i] -=  amplitude * DirichletKernel::getValueAtBin(diff, true_phase, N);
        }

        // Calculate residual energy
        double residual_energy = 0.0;
        // double residual_energy_check = 0.0;
        for (const auto &val: residual) {
            residual_energy += std::norm(val);
        }

        // for (const auto &val: residual_check) {
        //     residual_energy_check += std::norm(val);
        // }

        // Print progress
        std::cout << "Component " << components.size() << ": true bin="
                << true_freq * sample_rate / static_cast<double>(N) <<
                ": old bin="
                << peak_bin
                << ", phase=" << true_phase <<
                std::endl;
        const double retention = residual_energy / total_energy * 100;

        if (std::abs(prev_retention - retention) < 1e-6) {
            break;
        }
        prev_retention = retention;
        // const double retention_check = residual_energy_check / total_energy * 100;
        std::cout << "% L2 norm retained: " << retention << std::endl;
        // std::cout << "% L2 norm retained (check): " << retention_check << std::endl;

        if (constexpr double THRESHOLD = 1e-6; residual_energy < THRESHOLD * total_energy) {
            break;
        }
    }

    return components;
}

void writeSpectrumToCSV(const std::vector<Complex>& spectrum, const std::string& filename) {
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


std::vector<double> CleanDFT::decompressComponents(const std::vector<Component> &components,
                                                   const size_t N,
                                                   const Complex DC) {
    std::vector<Complex> spectrum(N, Complex(0, 0));
    spectrum[0] = computeDC(components, N);


    // Phase correction to ensure periodicity
    for (const auto &[true_frequency, true_phase, amplitude]: components) {
        for (size_t m = 1; m < N ; m++) {
            const double diff = true_frequency - static_cast<double>(m);
            // Add phase correction term to ensure periodicity
            double corrected_phase =  true_phase + M_PI * diff / N;
            spectrum[m] += amplitude * DirichletKernel::getValueAtBin(
                diff, corrected_phase, N
            );
        }
    }

    // Handle symmetry



    for (size_t m = 1; m <= N/ 2; m++) {
        spectrum[N - m] += std::conj(spectrum[m]); // * Complex(0, 1);
    }

    for (size_t m = 1; m <= N/ 2; m++) {
        spectrum[m] = std::conj(spectrum[N-m]); // * Complex(0, 1);
    }

    writeSpectrumToCSV(spectrum, "spectrum_before_ifft.csv");

    auto reconstructed = WaveProcessor::computeIFFT(spectrum);

    // Apply window to smooth boundaries
    // for (size_t i = 0; i < reconstructed.size(); i++) {
    //     double window = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (N - 1)));  // Hann window
    //     reconstructed[i] *= window;
    // }

    return reconstructed;
}

std::vector<double> CleanDFT::resample(const std::vector<Component> &components,
                                       const size_t N,
                                       const size_t original_sr,
                                       const size_t target_sr,
                                       const Complex DC) {
    const size_t new_N = static_cast<size_t>(
        std::ceil(static_cast<double>(N) * target_sr / original_sr)
    );

    // Debug output
    std::cout << "Original duration: " << static_cast<double>(N) / original_sr << "s\n";
    std::cout << "Expected new duration: " << static_cast<double>(new_N) / target_sr << "s\n";

    const double new_nyquist = target_sr / 2.0;
    std::vector<Component> valid_components;
    valid_components.reserve(components.size());

    for (const auto &comp: components) {
        const double freq_hz = comp.true_frequency * original_sr / N;
        if (freq_hz < new_nyquist) {
            const double freq_hz = comp.true_frequency * original_sr / N; // Convert original bin to Hz
            const double new_freq_bin = freq_hz * new_N / target_sr; // Convert Hz to new bins
            const double new_phase = comp.true_phase * (target_sr * N) / (original_sr * new_N);
            valid_components.push_back({new_freq_bin, new_phase, comp.amplitude});
            // Debug output for a few components
            if (valid_components.size() <= 3) {
                std::cout << "Original freq: " << freq_hz << "Hz, "
                        << "Scaled freq: " << new_freq_bin * target_sr / new_N << "Hz\n";
            }
        }
    }

    auto reconstructed = decompressComponents(valid_components, new_N, DC);

    // // Try fixing circular shift
    // const size_t shift_amount = reconstructed.size()/5;  // .2 seconds as you mentioned
    // ranges::rotate(reconstructed.begin(),
    //            reconstructed.begin() + reconstructed.size() - shift_amount,
    //            reconstructed.end());
    //
    // std::cout << "Final reconstructed length: " << reconstructed.size()
    //           << " samples at " << target_sr << "Hz\n"
    //           << "Final duration: " << static_cast<double>(reconstructed.size())/target_sr << "s\n";

    return reconstructed;
}

