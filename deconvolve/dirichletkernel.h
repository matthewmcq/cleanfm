/**
 * @file dirichletkernel.h
 * @author Matthew McQuistion
 * @date 04/25/25
 * @brief Defines the DirichletKernel class for calculations related to the DKD algorithm.
 *
 * This file contains the definition of the DirichletKernel class, which provides
 * static methods to calculate properties of the Dirichlet kernel (the Fourier
 * transform of a rectangular window). These calculations are essential for
 * modeling spectral leakage and accurately estimating component parameters (amplitude,
 * phase, frequency) in the Dirichlet Kernel Deconvolution (DKD) method.
 */
#pragma once

#include "../constants.h" // Include necessary constants (like Complex type definition)

// Standard include guard
#ifndef DIRICHLETKERNEL_H
#define DIRICHLETKERNEL_H

#include <vector> // For std::vector
#include <cmath>  // For mathematical functions like sin, cos, exp
#include <numeric> // Potentially needed for complex number operations if not in constants.h

/**
 * @brief Provides static methods for calculations involving the Dirichlet kernel.
 *
 * The Dirichlet kernel describes the spectral shape of a pure sinusoid within a
 * finite DFT window. This class offers tools to evaluate the kernel's value at
 * specific frequencies, infer true component amplitudes from peak bin magnitudes,
 * and generate the kernel's shape across multiple bins. These are core operations
 * for the DKD algorithm.
 */
class DirichletKernel {
public:
    /**
     * @brief Constructor for the DirichletKernel class.
     * @note This constructor is currently declared but may not be used if only static methods are employed.
     * If used, it would likely initialize properties for a specific kernel instance.
     *
     * @param frequency The true frequency of the component (in fractional bins).
     * @param amplitude The true amplitude of the component.
     * @param phase The true phase of the component (in radians).
     * @param N The size of the DFT window.
     */
    DirichletKernel(double frequency, double amplitude, double phase, int N);

    /**
     * @brief Calculates the complex value of the normalized Dirichlet kernel at a specific frequency offset.
     *
     * The formula implemented is typically:
     * D(diff, N) = (1/N) * sin(pi * diff) / sin(pi * diff / N) * exp(j * pi * diff * (N-1)/N) * exp(j * phase)
     * scaled by amplitude. The phase term provided might be combined internally.
     * This represents the contribution of a component at its true frequency to a specific DFT bin.
     *
     * @param difference The difference between the component's true frequency and the target bin's frequency (true_freq - bin_freq), in fractional bins.
     * @param phase The phase of the component (in radians) at its true frequency, potentially including phase correction terms.
     * @param N The size of the DFT window.
     * @return The complex value representing the kernel's contribution at the specified bin frequency relative to the true frequency.
     * @note Uses static_cast<double>(N) for calculations to ensure floating-point arithmetic. Handles potential division by zero near integer bins carefully (using limits or approximations).
     */
    [[nodiscard]] static Complex getValueAtBin(double difference, double phase, size_t N);


    /**
     * @brief Estimates the true amplitude of a component based on its magnitude at the nearest integer bin.
     *
     * The magnitude observed in an FFT bin is attenuated by the Dirichlet kernel relative to the component's
     * true amplitude, depending on how far the true frequency is from the bin center. This function inverts
     * that relationship. It calculates the magnitude of the Dirichlet kernel at the difference between the
     * true frequency and the nearest bin frequency, then divides the observed bin magnitude by this kernel
     * magnitude to estimate the true amplitude.
     * Amplitude_true = bin_magnitude / |D(frequency - nearest_bin, N)|
     *
     * @param frequency The estimated true frequency of the component (in fractional bins).
     * @param bin_magnitude The magnitude measured in the nearest integer DFT bin.
     * @param N The size of the DFT window.
     * @param nearest_bin The index of the integer DFT bin closest to the true frequency.
     * @return The estimated true amplitude of the frequency component.
     * @note Returns 0 if N is 0 to prevent division by zero.
     */
    [[nodiscard]] static double getAmplitudeAtBin(double frequency, double bin_magnitude, size_t N, int nearest_bin);


    /**
     * @brief Generates the complex values of the Dirichlet kernel across a specified set of bins.
     * @deprecated This function's signature seems less commonly used than direct evaluation via getValueAtBin.
     * It might be intended to generate a template kernel shape.
     * @param bins A vector of integer bin indices for which to calculate the kernel values.
     * @param frequency The true frequency of the component generating the kernel (in fractional bins).
     * @param phase The true phase of the component (in radians).
     * @param N The size of the DFT window.
     * @param bin_magnitude This parameter seems misplaced for generating a kernel shape based on true parameters. It might be intended as the *true* amplitude, or it might be unused.
     * @return A vector of complex numbers representing the kernel's values at the specified bins.
     * @note The role of `bin_magnitude` here is unclear based on standard kernel generation. It might represent the *true* amplitude to scale the kernel.
     */
    [[nodiscard]] static std::vector<Complex> generateKernel(const std::vector<int> &bins, double frequency, double phase,
                                                             size_t N, double bin_magnitude); // Revisit purpose of bin_magnitude
};


#endif // DIRICHLETKERNEL_H