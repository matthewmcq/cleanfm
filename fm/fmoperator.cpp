#include "fmoperator.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Initialize static Bessel function lookup table
BesselLookup FMOperator::besselLookup(20, 10.0, 0.01);

FMOperator::FMOperator(double carrierFreq, double modulatorFreq, double modulationIdx,
                     SidebandMode sbMode, SpectrumRange specRange, size_t maxSidebands)
    : carrierFrequency(carrierFreq),
      modulatorFrequency(modulatorFreq),
      modulationIndex(modulationIdx),
      sidebandMode(sbMode),
      spectrumRange(specRange),
      maxSidebandCount(maxSidebands) {

    updateSidebandAmplitudes();
}

// Use the lookup table for Bessel function calculation
double FMOperator::besselJ(int n, double x) {
    return besselLookup.besselJ(n, x);
}

void FMOperator::updateSidebandAmplitudes() {
    sidebandAmplitudes.clear();

    // Determine the range of sidebands to calculate
    int minSideband = (spectrumRange == SpectrumRange::FULL_SPECTRUM) ? -static_cast<int>(maxSidebandCount) : 0;
    int maxSideband = static_cast<int>(maxSidebandCount);

    // Calculate amplitudes for each sideband
    for (int i = minSideband; i <= maxSideband; ++i) {
        if (includeSideband(i)) {
            double amplitude = besselJ(std::abs(i), modulationIndex);

            // Apply sign alternation for negative-order Bessel functions with odd n
            if (i < 0 && i % 2 != 0) {
                amplitude = -amplitude;
            }

            sidebandAmplitudes.push_back(amplitude);
        }
    }
}

bool FMOperator::includeSideband(int sidebandNumber) {
    // Always include the carrier (sideband 0) regardless of mode
    if (sidebandNumber == 0) {
        return true;
    }

    int absN = std::abs(sidebandNumber);

    switch (sidebandMode) {
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

std::vector<std::pair<double, Complex>> FMOperator::generateSpectrum() {
    std::vector<std::pair<double, Complex>> spectrum;

    // Determine the range of sidebands to calculate
    int minSideband = (spectrumRange == SpectrumRange::FULL_SPECTRUM) ? -static_cast<int>(maxSidebandCount) : 0;
    int maxSideband = static_cast<int>(maxSidebandCount);

    // Add each sideband to the spectrum
    for (int i = minSideband; i <= maxSideband; ++i) {
        if (includeSideband(i)) {
            double frequency = carrierFrequency + i * modulatorFrequency;
            double amplitude = besselJ(std::abs(i), modulationIndex);

            // Apply sign alternation for negative-order Bessel functions with odd n
            if (i < 0 && i % 2 != 0) {
                amplitude = -amplitude;
            }

            // Only add sidebands with positive frequencies
            if (frequency >= 0) {
                // Calculate the phase for this sideband
                // For FM, each sideband has a phase related to the carrier
                // n*π/2 phase shift for each sideband (n = sideband number)
                double sidebandPhase = i * M_PI / 2.0;

                // Create complex amplitude using e^(i*θ) = cos(θ) + i*sin(θ)
                Complex complexAmp = amplitude * Complex(std::cos(sidebandPhase), std::sin(sidebandPhase));

                spectrum.push_back(std::make_pair(frequency, complexAmp));
            }
        }
    }

    return spectrum;
}

double FMOperator::getSidebandAmplitude(int sidebandNumber) {
    if (!includeSideband(sidebandNumber)) {
        return 0.0;
    }

    double amplitude = besselJ(std::abs(sidebandNumber), modulationIndex);

    // Apply sign alternation for negative-order Bessel functions with odd n
    if (sidebandNumber < 0 && sidebandNumber % 2 != 0) {
        amplitude = -amplitude;
    }

    return amplitude;
}

double FMOperator::getSidebandFrequency(int sidebandNumber) {
    return carrierFrequency + sidebandNumber * modulatorFrequency;
}

double FMOperator::correlateWithSpectrum(const std::vector<Complex>& targetSpectrum,
                                       size_t sampleRate, size_t fftSize) {
    // Generate this operator's spectrum with complex amplitudes
    auto fmSpectrum = generateSpectrum();

    // Create a vector of Complex values representing the FM spectrum
    std::vector<Complex> fmComplexSpectrum(fftSize, Complex(0, 0));

    // Populate the FM spectrum vector
    for (const auto& [freq, complexAmp] : fmSpectrum) {
        // Convert frequency to bin number
        size_t bin = static_cast<size_t>(freq * fftSize / sampleRate);

        // Ensure bin is within bounds
        if (bin < fftSize) {
            // Use the full complex amplitude
            fmComplexSpectrum[bin] = complexAmp;
        }
    }

    // Calculate correlation between the two spectra
    Complex numerator = 0;
    double normTarget = 0;
    double normFM = 0;

    // Only consider the first half of the spectrum (up to Nyquist)
    for (size_t i = 0; i < fftSize / 2; ++i) {
        // Use complex correlation to account for both magnitude and phase
        numerator += targetSpectrum[i] * std::conj(fmComplexSpectrum[i]);
        normTarget += std::norm(targetSpectrum[i]); // norm is |z|²
        normFM += std::norm(fmComplexSpectrum[i]);
    }

    // Return complex correlation coefficient (magnitude only)

    double ret = std::abs(numerator) / (std::sqrt(normTarget) * std::sqrt(normFM));
    std::cout << ret << std::endl;
    return ret;
}