//
// Created by Matthew McQuistion on 12/17/24.
//

#ifndef CLEANDFT_H
#define CLEANDFT_H
#include "../constants.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <tuple>
#include "../fft/waveprocessor.h"
#include "threadpool.h"
#include "dirichletkernel.h"

// Define sideband mode enum
enum class SidebandMode {
    ALL_SIDEBANDS, // All sidebands
    EVEN_SIDEBANDS, // Only even-numbered sidebands
    ODD_SIDEBANDS // Only odd-numbered sidebands
};


class CleanDFT {
public:
    struct Component {
        double true_frequency;
        double true_phase;
        double amplitude;
    };

    struct FMComponent {
        double carrier_freq; // Carrier frequency in bins
        double phase; // Phase of carrier
        double amplitude; // Amplitude of carrier
        double mod_index; // Modulation index for harmonic modulator (1:1 ratio)
        double lfo_freq; // Frequency of LFO modulator (in bins)
        double lfo_index; // Modulation index for LFO modulator
        double scalar;
    };

    std::mutex residual_mutex;

    explicit CleanDFT();

    static Complex computeDC(const std::vector<Component> &components, size_t N);

    static double spectralDistance(const PeakInfo &peak1, const PeakInfo &peak2, size_t N);

    static double computeCorrelation(const std::vector<Complex> &data_slice, const std::vector<Complex> &kernel,
                                     bool isPhase);

    static double calculateCorrelation(const std::vector<Complex> &residual, double carrier_freq, double carrier_phase,
                                       double carrier_amp, double mod_index, double lfo_freq, double lfo_index,
                                       size_t N);

    static double findOptimalPhase(const std::vector<Complex> &fft, int center_bin, double frequency);

    static double findOptimalFrequency(const std::vector<Complex> &fft, int center_bin, double test_frequency);

    static std::vector<Component> deconvolveDirichletKernel(const std::vector<Complex> &fft, size_t sample_rate,  int maxcomp);


    // New FM deconvolution method
    static std::vector<FMComponent> deconvolveFMDirichletKernel(const std::vector<Complex> &fft, size_t sample_rate);

    static std::vector<CleanDFT::Component> deconvolveHybridParallel(const std::vector<Complex> &fft, size_t sample_rate,
                                                              size_t num_threads, int maxcomp);

    static std::vector<CleanDFT::Component> deconvolveMultiPassNonIterative(const std::vector<Complex> &fft,
                                                                     size_t sample_rate,
                                                                     size_t num_threads, int maxcomp);

    static std::vector<CleanDFT::Component> deconvolveParallelDirichlet(const std::vector<Complex> &fft,
                                                                        size_t sample_rate,
                                                                        size_t num_threads,  int maxcomp);

    static std::vector<Component> deconvolveParallelDirichletWithMetrics(
        const std::vector<Complex> &fft, const size_t sample_rate, const size_t num_threads,
        const std::string &metrics_csv_file, int maxcomp);

    static std::vector<double> decompressComponents(const std::vector<Component> &components,
                                                    size_t N, Complex DC, double max_magnitude);

    static std::vector<double> decompressComponentsParallel(const std::vector<Component> &components, size_t N, Complex DC, double max_magnitude);

    static std::vector<double> decompressComponentsEnhanced(const std::vector<Component> &components, size_t N, Complex DC,
                                                            double max_magnitude);

    static double refinePhaseWithResidualAnalysis(const std::vector<Complex> &spectrum, double frequency, double initial_phase,
                                           double amplitude, int center_bin, size_t N);

    static std::vector<CleanDFT::Component> deconvolveNonIterativeParallel(const std::vector<Complex> &fft, size_t sample_rate,
                                                                           size_t num_threads, int maxcomp);

    // New method to decompress FM components
    static std::vector<double> decompressFMComponents(const std::vector<FMComponent> &components,
                                                      size_t N);

    static Complex computeFMDC(const std::vector<FMComponent> &components, size_t N);

    static std::vector<double> resample(const std::vector<Component> &components, size_t N,
                                        size_t original_sr, size_t target_sr, Complex DC, double max_magnitude);

    // New helper functions for FM deconvolution
    static int findPeakBin(const std::vector<Complex> &residual, size_t nyquist_bin);

    static bool isExitConditionMet(int peak_bin, size_t nyquist_bin,
                                   const std::vector<Complex> &residual);

    static std::tuple<double, double, double, double> findOptimalFMParameters(
        double carrier_freq, double carrier_phase, double carrier_amp,
        const std::vector<Complex> &residual, size_t N);

    template<class F>
    static double goldenSectionSearch(const F &objective, double a, double b, double tol);

    static double evaluateCorrelation(const std::vector<Complex> &residual, size_t N, double carrier_freq,
                                      double carrier_phase,
                                      double carrier_amp, double mod_index, double lfo_freq, double lfo_index);

    static double computeGradient(const std::vector<Complex> &residual, size_t N, double carrier_freq,
                                  double carrier_phase,
                                  double carrier_amp, double mod_index, double lfo_freq, double lfo_index,
                                  const std::string &param_name);

    static void addCombinedModulators(std::vector<Complex> &spectrum, size_t N, double carrier_freq,
                                      double carrier_phase,
                                      double carrier_amp, double mod_index, double lfo_freq, double lfo_index);

    static void subtractFMOperator(
        std::vector<Complex> &residual, size_t N,
        double carrier_freq, double carrier_phase, double carrier_amp,
        double mod_index, double lfo_freq, double lfo_index, double scalar);

    static void printProgress(size_t component_count, double carrier_freq, int peak_bin,
                              double carrier_phase, double mod_index,
                              double lfo_freq, double lfo_index, const std::vector<Complex> &residual,
                              double total_energy);

    static bool includeSideband(int sideband_number, SidebandMode mode);

    static double besselJ(int n, double x);

    // Save FM components to CSV
    static void writeFMComponentsToCSV(const std::vector<FMComponent> &components,
                                       size_t N, size_t sample_rate, const std::string &filename);

private:
    template<typename F>
    static double goldenSectionSearch(const F &objective, double a, double b) {
        double c = b - (b - a) / PHI;
        double d = a + (b - a) / PHI;

        double fc = objective(c);
        double fd = objective(d);

        while (std::abs(b - a) > GSS_TOLERANCE) {
            if (fc > fd) {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) / PHI;
                fc = objective(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) / PHI;
                fd = objective(d);
            }
        }

        return (a + b) / 2.0;
    }
};


#endif //CLEANDFT_H
