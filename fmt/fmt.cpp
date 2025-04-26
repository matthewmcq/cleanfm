/**
 * @file fmt.cpp
 * @author Matthew McQuistion
 * @date 4/26/25
 * @brief Implements the FMT class for Frequency Modulation Transform analysis.
 *
 * This file contains the implementation of the FMT class, which provides
 * methods for performing the Frequency Modulation Transform on audio spectra.
 * It includes functions for initializing a lookup table, applying breaking
 * operators, finding peaks, calculating modulation indices, computing the
 * full FMT, extracting FM parameters, and saving results to a CSV file.
 */

#include "fmt.h" // Include the header file for the FMT class
// Include the implementation for dr_wav, typically done in one source file.
// The path assumes it's in the same directory as fmt.cpp.
// #include "fmt/fmt.cpp" // This line seems incorrect, likely a typo. Assuming it should be an include for dr_wav or similar if needed here.
// Corrected based on likely intent or removal if not needed here:
// #include "../libs/dr_wav.h" // Example if dr_wav implementation is needed here

#include "../fft/waveprocessor.h" // Include header for WaveProcessor (for computeFFT)
#include <fstream> // For file input/output operations (ifstream, ofstream)
#include <sstream> // For string stream operations (stringstream)
#include <algorithm> // For algorithms like std::sort, std::lower_bound, std::max
#include <iostream> // For standard input/output streams (cerr, cout)
#include <cmath> // For mathematical functions like std::abs

// Static member initialization
/**
 * @brief Static member variable to store the FMT lookup table.
 *
 * Stores pairs of (FMT value, beta) for interpolation.
 */
std::vector<std::pair<double, double>> FMT::lookup_table;

/**
 * @brief Static member variable indicating if the lookup table has been initialized.
 */
bool FMT::initialized = false;

/**
 * @brief Initializes the FMT lookup table from a CSV file.
 *
 * Reads beta and corresponding FMT values from a specified CSV file,
 * stores them in the lookup_table vector, and sorts the table by FMT value
 * for efficient lookup.
 *
 * @param lookup_table_path The path to the CSV file containing the lookup table data.
 * @return True if initialization is successful, false otherwise.
 */
bool FMT::initialize(const std::string& lookup_table_path) {
    std::ifstream file(lookup_table_path); // Open the lookup table file
    if (!file.is_open()) {
        std::cerr << "Failed to open FMT lookup table: " << lookup_table_path << std::endl;
        return false; // Return false if file opening fails
    }

    std::string line;
    // Skip header line in the CSV file
    std::getline(file, line);

    lookup_table.clear(); // Clear any existing data in the lookup table

    // Read data lines from the file
    while (std::getline(file, line)) {
        std::stringstream ss(line); // Use stringstream to parse the line
        std::string beta_str, fmt_str; // Strings to hold beta and FMT values

        // Extract beta and FMT values separated by a comma
        if (std::getline(ss, beta_str, ',') && std::getline(ss, fmt_str)) {
            double beta = std::stod(beta_str); // Convert beta string to double
            double fmt_value = std::stod(fmt_str); // Convert FMT string to double
            lookup_table.emplace_back(fmt_value, beta);  // Store as pair (FMT value, beta)
        }
    }

    // Sort the lookup table by the FMT value (first element of the pair)
    // This is necessary for efficient binary search later.
    std::sort(lookup_table.begin(), lookup_table.end());

    initialized = true; // Set initialized flag to true
    std::cout << "FMT lookup table loaded with " << lookup_table.size() << " entries" << std::endl;
    return true; // Return true indicating successful initialization
}

/**
 * @brief Applies the breaking operator to a frequency spectrum.
 *
 * Computes the magnitude of each complex number in the input spectrum
 * and returns a vector of these magnitudes. Only processes the first half
 * of the spectrum (up to Nyquist frequency).
 *
 * @param spectrum A vector of Complex numbers representing the frequency spectrum.
 * @return A vector of double-precision floating-point values representing the
 * magnitudes of the first half of the spectrum.
 */
std::vector<double> FMT::applyBreakingOperator(const std::vector<Complex>& spectrum) {
    size_t half_size = spectrum.size() / 2; // Calculate the size of the first half (including DC)
    std::vector<double> broken_signal(half_size); // Vector to store the magnitudes

    // Compute the magnitude for each complex number in the first half of the spectrum
    for (size_t i = 0; i < half_size; i++) {
        broken_signal[i] = std::abs(spectrum[i]); // Store the absolute value (magnitude)
    }

    return broken_signal; // Return the vector of magnitudes
}

/**
 * @brief Finds peak indices in a frequency spectrum.
 *
 * Identifies indices where the magnitude is greater than its immediate
 * neighbors (a local maximum) and above a small threshold to avoid noise.
 * Excludes the first and last bins due to boundary conditions.
 *
 * @param spectrum A vector of Complex numbers representing the frequency spectrum.
 * @return A vector of size_t containing the indices of the detected peaks.
 */
std::vector<size_t> FMT::findPeaks(const std::vector<Complex>& spectrum) {
    std::vector<size_t> peaks; // Vector to store peak indices

    // Iterate through the spectrum, excluding the first and last elements
    for (size_t i = 1; i < spectrum.size() - 1; i++) {
        double current = std::abs(spectrum[i]); // Magnitude of the current bin
        double prev = std::abs(spectrum[i-1]); // Magnitude of the previous bin
        double next = std::abs(spectrum[i+1]); // Magnitude of the next bin

        // Check if the current bin's magnitude is a local maximum and above a threshold
        if (current > prev && current > next && current > 1e-10) {
            peaks.push_back(i); // Add the index to the peaks vector
        }
    }

    return peaks; // Return the vector of peak indices
}

/**
 * @brief Estimates the modulation index from a normalized FMT amplitude using the lookup table.
 *
 * Performs linear interpolation on the loaded lookup table to find the
 * corresponding modulation index for a given normalized FMT amplitude.
 * Handles cases outside the range of the lookup table by returning the
 * minimum or maximum beta value.
 *
 * @param fmt_amplitude The normalized amplitude from the FMT spectrum (usually relative to the DC component).
 * @return The estimated modulation index (beta). Returns 0.0 if the lookup table is not initialized.
 */
double FMT::modulationIndexFromAmplitude(double fmt_amplitude) {
    if (!initialized) {
        std::cerr << "FMT lookup table not initialized!" << std::endl;
        return 0.0; // Return 0 if the lookup table hasn't been loaded
    }

    // Handle cases outside the lookup table range
    if (fmt_amplitude <= lookup_table.front().first) {
        return lookup_table.front().second; // Return the minimum beta
    }
    if (fmt_amplitude >= lookup_table.back().first) {
        return lookup_table.back().second; // Return the maximum beta
    }

    // Perform binary search to find the position in the sorted lookup table.
    // lower_bound finds the first element not less than (fmt_amplitude, 0.0).
    auto it = std::lower_bound(lookup_table.begin(), lookup_table.end(),
                               std::make_pair(fmt_amplitude, 0.0));

    // If fmt_amplitude is exactly the first value, return its beta
    if (it == lookup_table.begin()) {
        return it->second;
    }

    // Get the iterator to the element just before 'it'
    auto prev = std::prev(it);

    // Perform linear interpolation between the two points (prev and it)
    // t is the interpolation factor between 0 and 1
    double t = (fmt_amplitude - prev->first) / (it->first - prev->first);
    // Interpolate the beta value
    return prev->second + t * (it->second - prev->second);
}

/**
 * @brief Computes the Frequency Modulation Transform (FMT) of an input spectrum.
 *
 * Performs the sequence of operations: apply breaking operator, compute DFT,
 * apply breaking operator again, compute final DFT. It then finds peaks in
 * the final spectrum, calculates their magnitudes, normalizes them by the
 * DC component, and estimates modulation indices using the lookup table.
 *
 * @param signal A vector of Complex numbers representing the initial frequency spectrum (result of the first DFT).
 * @return An FMTResult struct containing the third-order spectrum, peak indices,
 * magnitudes, and estimated modulation indices.
 */
FMT::FMTResult FMT::computeFMT(const std::vector<Complex>& signal) {
    FMTResult result; // Structure to store the results

    // Step 1: First DFT (already done - input 'signal' is the DFT)

    // Step 2: Apply breaking operator to the first DFT result
    std::vector<double> broken_signal1 = applyBreakingOperator(signal);

    // Step 3: Compute the second DFT on the result of the first breaking operator.
    // WaveProcessor::computeFFT expects a vector of doubles.
    auto second_spectrum = WaveProcessor::computeFFT(broken_signal1);

    // Step 4: Apply breaking operator again to the second DFT result.
    std::vector<double> broken_signal2 = applyBreakingOperator(second_spectrum);

    // Step 5: Compute the third DFT on the result of the second breaking operator.
    // This is the final FMT spectrum.
    auto third_spectrum = WaveProcessor::computeFFT(broken_signal2);

    // Store the full third-order spectrum in the result structure.
    result.third_order_frequencies = third_spectrum;

    // Find peak indices in the third-order spectrum.
    result.peak_indices = findPeaks(third_spectrum);

    // Calculate the magnitudes of the peaks found.
    result.magnitudes.resize(result.peak_indices.size()); // Resize magnitudes vector
    for (size_t i = 0; i < result.peak_indices.size(); i++) {
        // Get the magnitude at each peak index
        result.magnitudes[i] = std::abs(third_spectrum[result.peak_indices[i]]);
    }

    // Normalize peak magnitudes and calculate modulation indices.
    double max_magnitude = 0.0; // Find the maximum magnitude (not used for normalization here, DC is used)
    for (double mag : result.magnitudes) {
        max_magnitude = std::max(max_magnitude, mag);
    }

    // Check if the DC component is non-zero to avoid division by zero.
    // The DC component is found at index 0 of the third spectrum.
    double dc_magnitude = std::abs(third_spectrum[0]);

    if (dc_magnitude > 1e-10) { // Use a small threshold for non-zero check
        result.modulation_indices.resize(result.magnitudes.size()); // Resize modulation indices vector

        // Calculate normalized magnitude and look up modulation index for each peak.
        for (size_t i = 0; i < result.magnitudes.size(); i++) {
            // Normalize the peak magnitude by the DC component magnitude.
            double normalized_mag = result.magnitudes[i] / dc_magnitude;
            // Estimate the modulation index using the lookup table.
            result.modulation_indices[i] = modulationIndexFromAmplitude(normalized_mag);
        }
    } else {
         // If DC is zero or near zero, modulation indices cannot be normalized this way.
         // Set modulation indices to 0 or handle as an error case.
         result.modulation_indices.assign(result.magnitudes.size(), 0.0);
         if (!result.magnitudes.empty()) {
             std::cerr << "Warning: DC component of third spectrum is zero or near zero. Cannot normalize magnitudes." << std::endl;
         }
    }


    return result; // Return the computed FMT results
}

/**
 * @brief Extracts Frequency Modulation (FM) operator parameters from FMT results.
 *
 * Processes the peak information from the FMT result to determine dominant
 * modulator frequencies and their corresponding modulation indices.
 * Filters out the DC component and frequencies above Nyquist. Sorts peaks
 * by magnitude to prioritize dominant modulators.
 *
 * @param fmt_result The result structure from the computeFMT function.
 * @param sample_rate The sample rate of the original audio signal.
 * @param original_fft_size The size of the initial FFT performed on the audio signal.
 * @return An FMOperatorParams struct containing vectors of modulator frequencies and modulation indices.
 */
FMT::FMOperatorParams FMT::extractFMParameters(const FMTResult& fmt_result,
                                             double sample_rate,
                                             size_t original_fft_size) {
    FMOperatorParams params; // Structure to store the extracted FM parameters

    // Vector to hold peak information for sorting: (bin, magnitude, modulation_index)
    std::vector<std::tuple<size_t, double, double>> sorted_peaks;

    // Iterate through the detected peaks. Only consider the first half of peaks
    // because the third spectrum is real and symmetric, and peaks in the second
    // half correspond to negative frequencies or are redundant.
    // Also, ensure the peak index is not the DC component (index 0).
    for (size_t i = 0; i < fmt_result.peak_indices.size() / 2; i++) { // Iterate through all peak indices
        size_t bin = fmt_result.peak_indices[i];
        if (bin > 0) {  // Exclude the DC component (bin 0)
            // Add peak info as a tuple (bin, magnitude, modulation_index)
            sorted_peaks.emplace_back(
                bin,
                fmt_result.magnitudes[i],
                fmt_result.modulation_indices[i]
            );
        }
    }

    // Sort the peaks by magnitude in descending order.
    std::sort(sorted_peaks.begin(), sorted_peaks.end(),
              [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });

    // Extract the modulator frequencies and modulation indices from the sorted peaks.
    for (const auto& peak : sorted_peaks) {
        // Convert the peak bin index from the third spectrum to a frequency in Hz.
        // The frequency resolution of the third spectrum is (sample_rate / original_fft_size) / (third_spectrum_size).
        // The bin index in the third spectrum corresponds to a frequency in the *original* signal's spectrum.
        // The scaling factor needs careful consideration based on the sizes of the FFTs.
        // Assuming the third spectrum size is related to original_fft_size / 2.
        // A bin 'k' in the third spectrum corresponds to a frequency k * (sample_rate / (original_fft_size / 2))? No, this is complex.
        // A bin 'k' in the third spectrum corresponds to a frequency k * (sample_rate / original_fft_size) in the original *time-domain* signal's spectrum.
        // Let's assume the bin-to-frequency mapping is relative to the original FFT size.
        double freq = static_cast<double>(std::get<0>(peak)) * sample_rate / static_cast<double>(original_fft_size);

        // Only keep frequencies below the Nyquist frequency of the original signal.
        if (freq < sample_rate / 2.0) {
             params.modulators.push_back(freq); // Add the modulator frequency
             params.modulation_indices.push_back(std::get<2>(peak)); // Add the corresponding modulation index
        }
        // Note: The number of modulators extracted might need a limit based on MAX_COMPONENTS or other criteria.
    }

    return params; // Return the extracted FM parameters
}

/**
 * @brief Saves the computed FMT results to a CSV file.
 *
 * Writes the peak information (frequency bin, frequency in Hz, magnitude,
 * and modulation index) to a specified CSV file. Includes a header row.
 *
 * @param result The FMTResult structure containing the results.
 * @param filename The path to the output CSV file.
 * @param sample_rate The sample rate of the original audio signal.
 * @param original_fft_size The size of the initial FFT performed on the audio signal.
 */
void FMT::saveFMTToCSV(const FMTResult& result, const std::string& filename,
                       double sample_rate, size_t original_fft_size) {
    std::ofstream file(filename); // Open the output file

    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return; // Return if file opening fails
    }

    // Write the header row to the CSV
    file << "frequency_bin,frequency_hz,magnitude,modulation_index\n";

    // Write the data for each detected peak
    for (size_t i = 0; i < result.peak_indices.size(); i++) {
        size_t bin = result.peak_indices[i]; // Get the peak bin index
        // Convert the peak bin index to frequency in Hz, relative to the original FFT size and sample rate.
        double freq_hz = static_cast<double>(bin) * sample_rate / static_cast<double>(original_fft_size);

        // Write the data fields for the current peak, separated by commas
        file << bin << ","
             << freq_hz << ","
             << result.magnitudes[i] << ","
             << result.modulation_indices[i] << "\n";
    }

    file.close(); // Close the output file
    std::cout << "FMT results written to " << filename << std::endl; // Log confirmation
}
