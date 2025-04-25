/**
* @file cleandft.h
* @author Matthew McQuistion
* @date 4/25/25
* @brief Main header file for the Dirichlet Kernel Deconvolution (DKD) algorithm
*
* This file defines the CleanDFT class which implements the windowless approach to
* spectral analysis described in the thesis. The DKD algorithm extracts precise
* frequency components from discrete spectra by modeling and removing spectral leakage
* patterns, offering an alternative to traditional windowing techniques.
*/

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

/**
* @class CleanDFT
* @brief Main class implementing the Dirichlet Kernel Deconvolution algorithm
*
* CleanDFT provides methods for:
* - Extracting precise frequency components from DFT spectra
* - Handling spectral leakage without windowing
* - Parallel processing for improved performance
* - Signal reconstruction from extracted components
* - Resampling with preserved spectral characteristics
*/
class CleanDFT {
public:
   /**
    * @struct Component
    * @brief Represents a single extracted frequency component
    *
    * Each component contains the true frequency (with sub-bin precision),
    * phase, and amplitude after correction for spectral leakage.
    */
   struct Component {
       double true_frequency;  ///< Precise frequency in fractional bins
       double true_phase;      ///< Phase in radians
       double amplitude;       ///< Corrected amplitude
   };

   std::mutex residual_mutex;  ///< Mutex for thread-safe residual updates

   /**
    * @brief Default constructor
    */
   explicit CleanDFT();

   /**
    * @brief Computes the DC component from extracted components
    *
    * Calculates how each extracted component contributes to the DC bin,
    * accounting for kernel effects from all frequencies.
    *
    * @param components Vector of extracted frequency components
    * @param N FFT size
    * @return Complex DC value
    */
   static Complex computeDC(const std::vector<Component> &components, size_t N);

   /**
    * @brief Calculates spectral distance between two peaks
    *
    * Used to ensure adequate separation when selecting peaks for parallel processing.
    * Prevents interference between simultaneously processed components.
    *
    * @param peak1 First peak information
    * @param peak2 Second peak information
    * @param N FFT size
    * @return Normalized spectral distance
    */
   static double spectralDistance(const PeakInfo &peak1, const PeakInfo &peak2, size_t N);

   /**
    * @brief Computes correlation between spectral data and Dirichlet kernel
    *
    * Core function used in frequency and phase optimization. Measures how well
    * a Dirichlet kernel with given parameters matches the observed spectrum.
    *
    * @param data_slice Local spectrum around the peak
    * @param kernel Dirichlet kernel values
    * @param isPhase True for phase correlation, false for frequency correlation
    * @return Normalized correlation coefficient
    */
   static double computeCorrelation(const std::vector<Complex> &data_slice,
                                  const std::vector<Complex> &kernel,
                                  bool isPhase);

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
   static double findOptimalPhase(const std::vector<Complex> &fft, int center_bin, double frequency);

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
   static double findOptimalFrequency(const std::vector<Complex> &fft, int center_bin, double test_frequency);

   /**
    * @brief Single-threaded DKD algorithm
    *
    * Iteratively extracts frequency components from the spectrum by identifying
    * peaks, optimizing parameters, and subtracting Dirichlet kernel responses.
    *
    * @param fft Input spectrum
    * @param sample_rate Sampling frequency
    * @param maxcomp Maximum number of components to extract
    * @return Vector of extracted components
    */
   static std::vector<Component> deconvolveDirichletKernel(const std::vector<Complex> &fft,
                                                         size_t sample_rate,
                                                         int maxcomp);

   /**
    * @brief Parallel implementation of DKD algorithm
    *
    * Divides the spectrum into bands and processes multiple peaks simultaneously,
    * significantly improving performance while maintaining accuracy.
    *
    * @param fft Input spectrum
    * @param sample_rate Sampling frequency
    * @param num_threads Number of threads to use
    * @param maxcomp Maximum number of components to extract
    * @return Vector of extracted components
    */
   static std::vector<Component> deconvolveParallelDirichlet(const std::vector<Complex> &fft,
                                                           size_t sample_rate,
                                                           size_t num_threads,
                                                           int maxcomp);

   /**
    * @brief Parallel DKD with metrics collection
    *
    * Same as deconvolveParallelDirichlet but also collects accuracy metrics
    * during extraction for analysis and comparison purposes.
    *
    * @param fft Input spectrum
    * @param sample_rate Sampling frequency
    * @param num_threads Number of threads to use
    * @param metrics_csv_file Path to save metrics
    * @param maxcomp Maximum number of components to extract
    * @return Vector of extracted components
    */
   static std::vector<Component> deconvolveParallelDirichletWithMetrics(
       const std::vector<Complex> &fft, size_t sample_rate, size_t num_threads,
       const std::string &metrics_csv_file, int maxcomp);

   /**
    * @brief Reconstructs time-domain signal from components
    *
    * Synthesizes the original signal by summing Dirichlet kernels for each
    * extracted component, properly handling phase corrections and DC.
    *
    * @param components Vector of extracted components
    * @param N FFT size
    * @param DC DC component
    * @param max_magnitude Maximum magnitude for normalization
    * @return Reconstructed time-domain signal
    */
   static std::vector<double> decompressComponents(const std::vector<Component> &components,
                                                 size_t N, Complex DC, double max_magnitude);

   /**
    * @brief Parallel version of signal reconstruction
    *
    * Multi-threaded implementation of decompressComponents for faster
    * reconstruction of signals with many components.
    *
    * @param components Vector of extracted components
    * @param N FFT size
    * @param DC DC component
    * @param max_magnitude Maximum magnitude for normalization
    * @param num_threads Number of threads to use
    * @return Reconstructed time-domain signal
    */
   static std::vector<double> decompressComponentsParallel(const std::vector<Component> &components,
                                                         size_t N, Complex DC,
                                                         double max_magnitude,
                                                         size_t num_threads);

   /**
    * @brief Non-iterative parallel deconvolution
    *
    * Extracts all components in a single pass without iterative refinement.
    * Faster but may have phase accuracy limitations for complex signals.
    *
    * @param fft Input spectrum
    * @param sample_rate Sampling frequency
    * @param num_threads Number of threads to use
    * @param maxcomp Maximum number of components to extract
    * @return Vector of extracted components
    */
   static std::vector<Component> deconvolveNonIterativeParallel(const std::vector<Complex> &fft,
                                                              size_t sample_rate,
                                                              size_t num_threads,
                                                              int maxcomp);

   /**
    * @brief Resamples signal using DKD components
    *
    * Implements frequency-domain resampling by remapping component frequencies
    * to a new sample rate, preserving spectral characteristics better than
    * traditional time-domain methods.
    *
    * @param components Vector of extracted components
    * @param N Original FFT size
    * @param original_sr Original sample rate
    * @param target_sr Target sample rate
    * @param DC DC component
    * @param max_magnitude Maximum magnitude for normalization
    * @param use_parallel Enable parallel processing
    * @param num_threads Number of threads to use
    * @return Resampled signal
    */
   static std::vector<double> resample(const std::vector<Component> &components, size_t N,
                                     size_t original_sr, size_t target_sr,
                                     Complex DC, double max_magnitude,
                                     bool use_parallel, size_t num_threads);

   /**
    * @brief Finds peak bin in residual spectrum
    *
    * Helper function that locates the bin with maximum magnitude in the
    * current residual, used in iterative component extraction.
    *
    * @param residual Current residual spectrum
    * @param nyquist_bin Nyquist frequency bin index
    * @return Index of peak bin
    */
   static int findPeakBin(const std::vector<Complex> &residual, size_t nyquist_bin);

   /**
    * @brief Checks if deconvolution should terminate
    *
    * Evaluates whether the algorithm should stop based on peak location
    * and magnitude thresholds.
    *
    * @param peak_bin Index of current peak
    * @param nyquist_bin Nyquist frequency bin index
    * @param residual Current residual spectrum
    * @return True if extraction should stop
    */
   static bool isExitConditionMet(int peak_bin, size_t nyquist_bin,
                                 const std::vector<Complex> &residual);

private:
   /**
    * @brief Golden section search implementation
    *
    * Efficiently finds the maximum of a unimodal function without requiring
    * derivatives. Used for frequency and phase optimization.
    *
    * @tparam F Function type (must be callable with double)
    * @param objective Function to maximize
    * @param a Lower bound of search interval
    * @param b Upper bound of search interval
    * @return Value that maximizes the objective function
    */
   template<typename F>
   static double goldenSectionSearch(const F &objective, double a, double b) {
       double c = b - (b - a) / PHI;  // PHI is the golden ratio
       double d = a + (b - a) / PHI;

       double fc = objective(c);
       double fd = objective(d);

       // Iteratively narrow the search interval
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

       return (a + b) / 2.0;  // Return midpoint of final interval
   }
};

#endif //CLEANDFT_H