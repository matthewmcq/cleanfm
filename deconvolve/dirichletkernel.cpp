/**
* @file dirichletkernel.cpp
* @author Matthew McQuistion
* @date 4/25/25
* @brief Implementation of the Dirichlet Kernel for spectral analysis.
*
* The Dirichlet kernel represents the frequency response of a rectangular windowed signal in the DFT.
* It is used in the DKD algorithm to model and remove spectral leakage patterns.
*/

#include "dirichletkernel.h"
#include <cmath>

/**
* Constructor for DirichletKernel.
*
* @param frequency The continuous frequency parameter of the kernel
* @param amplitude The amplitude of the kernel (unused in current implementation)
* @param phase The phase of the kernel (unused in current implementation)
* @param N The FFT size (unused in current implementation)
*/
DirichletKernel::DirichletKernel(const double frequency, const double amplitude, const double phase, const int N) {
   // Constructor is currently empty as the class uses static methods
   // Future implementations may store these parameters as member variables
}

/**
* Calculates the complex value of the Dirichlet kernel at a specific frequency bin.
*
* This function evaluates the Dirichlet kernel at a given fractional bin offset from
* a true frequency, accounting for phase correction due to the rectangular window.
*
* @param difference The frequency difference (true_frequency - bin_index)
* @param phase The phase of the component at the true frequency
* @param N The FFT size
* @return Complex value of the kernel at the specified bin
*/
Complex DirichletKernel::getValueAtBin(const double difference, const double phase, const size_t N) {
   // Define imaginary unit
   constexpr std::complex<double> i(0, 1);

   // Compute the numerator and denominator of the Dirichlet kernel
   const double numerator = std::sin(M_PI * difference);
   const double denominator = std::sin(M_PI * difference / static_cast<double>(N));

   // Handle the case where difference is near zero to avoid division by zero
   if (std::abs(difference) < 1e-16) {
       return std::exp(-i * phase);
   }

    // If you want phases for sine, change COSINE_PHASES_ONLY to false in constants.h
    // For Non-iterative, sine basis seems to be a better approximation of phases
    const int sine_coefficient = USE_COSINE_BASIS ? 1 : -1;

   // Calculate the complex kernel value
   // The kernel includes both magnitude (sinc-like term) and phase components
   // Assume phase has been corrected for kernel torsion before it has been passed in
   return (numerator / denominator) * Complex(
       std::cos(M_PI * difference + phase),
       sine_coefficient*std::sin((M_PI * difference + phase))
   );

}

/**
* Calculates the true amplitude of a frequency component from its observed bin magnitude.
*
* This function corrects for the attenuation caused by spectral leakage when a frequency
* doesn't align perfectly with a DFT bin. The correction factor is derived from the
* Dirichlet kernel's magnitude at the fractional bin offset.
*
* @param frequency The true continuous frequency
* @param bin_magnitude The observed magnitude at the nearest bin
* @param N The FFT size
* @param nearest_bin The index of the nearest DFT bin
* @return The corrected amplitude of the frequency component
*/
double DirichletKernel::getAmplitudeAtBin(const double frequency, const double bin_magnitude, const size_t N,
                                         const int nearest_bin) {
   // Calculate the fractional bin offset
   const double delta = frequency - static_cast<double>(nearest_bin);

   // Compute the sinc correction factor
   // When delta is near zero, we avoid division by zero by returning 1
   const double sinc_correction =
           std::abs(delta) < 1e-16 ? 1 : std::sin(M_PI * delta) / std::sin(M_PI * delta / static_cast<double>(N));

   // Apply the correction to get the true amplitude
   return bin_magnitude / sinc_correction;
}

/**
* Generates a Dirichlet kernel response across multiple frequency bins.
*
* This function creates the kernel response pattern for a given frequency and phase
* across a specified set of frequency bins. It's used in the DKD algorithm to model
* how a single frequency component affects multiple DFT bins through spectral leakage.
*
* @param bins Vector of bin indices where the kernel should be evaluated
* @param frequency The true continuous frequency of the component
* @param phase The phase of the component
* @param N The FFT size
* @param bin_magnitude The magnitude at the peak bin (currently unused)
* @return Vector of complex values representing the kernel response
*/
std::vector<Complex> DirichletKernel::generateKernel(const std::vector<int> &bins, const double frequency,
                                                    const double phase, const size_t N,
                                                    const double bin_magnitude) {
   std::vector<Complex> result(bins.size());

   // Iterate through all specified bins
   for (int i = 0; i < bins.size(); i++) {
       const int bin = bins[i];

       // Calculate the frequency difference from the true frequency
       const double difference = frequency - static_cast<double>(bin);

       // Apply phase correction based on the frequency difference
       // This correction ensures the kernel maintains proper phase relationships
       const double corrected_phase = phase + M_PI * difference / static_cast<double>(N);

       // Calculate the kernel value at this bin
       result[i] = getValueAtBin(difference, corrected_phase, N);
   }

   return result;
}