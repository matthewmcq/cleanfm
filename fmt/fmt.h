/**
 * @file fmt.h
 * @author Matthew McQuistion
 * @date 4/26/25
 * @brief Declares the FMT class and supporting structures for Frequency Modulation Transform analysis.
 *
 * This header file defines the structures used to hold the results of the
 * Frequency Modulation Transform (`FMTResult`) and the extracted FM operator
 * parameters (`FMOperatorParams`). It also declares the static `FMT` class
 * which provides the interface for performing FMT analysis, including
 * lookup table initialization, computation, parameter extraction, and result saving.
 */

#ifndef FMT_H // Include guard to prevent multiple inclusions
#define FMT_H

#pragma once // Alternative include guard, preferred by some compilers

#include "../constants.h" // Includes definition of Complex type and other constants
#include <vector>   // For std::vector
#include <string>   // For std::string
#include <map>      // For std::map (though not directly used in this header's definitions)

/**
 * @brief A static class providing utility functions for Frequency Modulation Transform (FMT) analysis.
 *
 * This class encapsulates the logic for computing the FMT of an audio spectrum,
 * extracting relevant FM parameters, and managing a lookup table for
 * modulation index estimation.
 */
class FMT {
public:
    /**
     * @brief Structure to hold the results of the Frequency Modulation Transform.
     *
     * Contains the third-order frequency spectrum, magnitudes of detected peaks,
     * indices of these peaks, and their estimated modulation indices.
     */
    struct FMTResult {
        std::vector<Complex> third_order_frequencies; ///< The complex spectrum resulting from the third DFT.
        std::vector<double> magnitudes;              ///< The magnitudes of the detected peaks in the third spectrum.
        std::vector<size_t> peak_indices;            ///< The indices (bins) of the detected peaks in the third spectrum.
        std::vector<double> modulation_indices;      ///< The estimated modulation indices corresponding to the peaks.
    };

    /**
     * @brief Structure to hold extracted Frequency Modulation (FM) operator parameters.
     *
     * Contains the carrier frequency (currently unused in implementation but part of struct),
     * and vectors of modulator frequencies and their corresponding modulation indices.
     */
    struct FMOperatorParams {
        double carrier;                         ///< The carrier frequency (Hz).
        std::vector<double> modulators;         ///< Vector of modulator frequencies (Hz).
        std::vector<double> modulation_indices; ///< Vector of modulation indices (beta values).
    };

    /**
     * @brief Initializes the internal lookup table from a CSV file.
     *
     * Reads beta and corresponding FMT values from the specified file for
     * use in estimating modulation indices from FMT amplitudes. Must be
     * called before `modulationIndexFromAmplitude` or `computeFMT`.
     *
     * @param lookup_table_path The path to the CSV file containing the lookup table data.
     * @return True if the lookup table was successfully loaded and initialized, false otherwise.
     */
    static bool initialize(const std::string& lookup_table_path);

    /**
     * @brief Computes the Frequency Modulation Transform (FMT) of an input spectrum.
     *
     * Performs the sequence of operations (breaking operator, DFT, breaking
     * operator, DFT) to compute the third-order spectrum. It then finds
     * peaks, calculates their magnitudes, and estimates modulation indices.
     *
     * @param signal A vector of Complex numbers representing the initial frequency spectrum (result of the first DFT).
     * @return An FMTResult struct containing the computed third-order spectrum and peak information.
     */
    static FMTResult computeFMT(const std::vector<Complex>& signal);

    /**
     * @brief Extracts Frequency Modulation (FM) operator parameters from FMT results.
     *
     * Analyzes the peak information from the FMT result to identify dominant
     * modulator frequencies and their corresponding modulation indices.
     *
     * @param fmt_result The result structure obtained from the `computeFMT` function.
     * @param sample_rate The sample rate of the original audio signal (Hz).
     * @param original_fft_size The size of the initial FFT performed on the audio signal.
     * @return An FMOperatorParams struct containing the extracted modulator frequencies and modulation indices.
     */
    static FMOperatorParams extractFMParameters(const FMTResult& fmt_result,
                                              double sample_rate,
                                              size_t original_fft_size);

    /**
     * @brief Saves the computed FMT results to a CSV file.
     *
     * Writes the peak information (frequency bin, frequency in Hz, magnitude,
     * and modulation index) from the FMTResult structure to a specified CSV file.
     *
     * @param result The FMTResult structure containing the results to save.
     * @param filename The path to the output CSV file.
     * @param sample_rate The sample rate of the original audio signal (Hz).
     * @param original_fft_size The size of the initial FFT performed on the audio signal.
     */
    static void saveFMTToCSV(const FMTResult& result, const std::string& filename,
                            double sample_rate, size_t original_fft_size);

private:
    /**
     * @brief Applies the breaking operator to a frequency spectrum.
     *
     * Computes the magnitude of each complex number and returns a vector of
     * these magnitudes for the first half of the spectrum.
     *
     * @param spectrum A vector of Complex numbers representing the frequency spectrum.
     * @return A vector of double-precision floating-point values representing the magnitudes.
     */
    static std::vector<double> applyBreakingOperator(const std::vector<Complex>& spectrum);

    /**
     * @brief Finds peak indices in a frequency spectrum.
     *
     * Identifies local maxima in the magnitude spectrum that are above a threshold.
     *
     * @param spectrum A vector of Complex numbers representing the frequency spectrum.
     * @return A vector of size_t containing the indices of the detected peaks.
     */
    static std::vector<size_t> findPeaks(const std::vector<Complex>& spectrum);

    /**
     * @brief Estimates the modulation index from a normalized FMT amplitude using the lookup table.
     *
     * Performs linear interpolation on the loaded lookup table to find the
     * corresponding modulation index for a given amplitude value. Requires
     * the lookup table to be initialized.
     *
     * @param fmt_amplitude The normalized amplitude from the FMT spectrum.
     * @return The estimated modulation index (beta). Returns 0.0 if the lookup table is not initialized.
     */
    static double modulationIndexFromAmplitude(double fmt_amplitude);

    /**
     * @brief Static member variable storing the FMT lookup table data.
     *
     * A vector of pairs, where each pair is (FMT value, beta). Sorted by FMT value.
     */
    static std::vector<std::pair<double, double>> lookup_table;

    /**
     * @brief Static member flag indicating whether the lookup table has been successfully initialized.
     */
    static bool initialized;
};


#endif //FMT_H
