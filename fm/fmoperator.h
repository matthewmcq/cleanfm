#ifndef FM_OPERATOR_H
#define FM_OPERATOR_H

#include <vector>
#include <cmath>
#include <complex>
#include "../constants.h"
#include "bessellookup.h"

// Enum for sideband selection modes
enum class SidebandMode {
    ALL_SIDEBANDS,       // All sidebands
    EVEN_SIDEBANDS,      // Only even-numbered sidebands
    ODD_SIDEBANDS        // Only odd-numbered sidebands
};

// Enum for spectrum range
enum class SpectrumRange {
    FULL_SPECTRUM,       // Both negative and positive sidebands
    POSITIVE_ONLY        // Only positive sidebands (above carrier)
};

class FMOperator {
public:
    // Constructor with carrier, modulator frequency and modulation index
    FMOperator(double carrierFreq, double modulatorFreq, double modulationIdx,
               SidebandMode sbMode = SidebandMode::ALL_SIDEBANDS,
               SpectrumRange specRange = SpectrumRange::POSITIVE_ONLY,
               size_t maxSidebands = 10);

    // Generate spectrum for this operator with complex amplitudes
    std::vector<std::pair<double, Complex>> generateSpectrum();

    // Get the amplitude of a specific sideband
    double getSidebandAmplitude(int sidebandNumber);

    // Get the frequency of a specific sideband
    double getSidebandFrequency(int sidebandNumber);

    // Calculate correlation between this operator's spectrum and a target spectrum
    double correlateWithSpectrum(const std::vector<Complex>& targetSpectrum,
                                size_t sampleRate, size_t fftSize);

    // Setters for properties
    void setCarrierFrequency(double freq) { carrierFrequency = freq; }
    void setModulatorFrequency(double freq) { modulatorFrequency = freq; }
    void setModulationIndex(double idx) { modulationIndex = idx; updateSidebandAmplitudes(); }
    void setSidebandMode(SidebandMode mode) { sidebandMode = mode; updateSidebandAmplitudes(); }
    void setSpectrumRange(SpectrumRange range) { spectrumRange = range; updateSidebandAmplitudes(); }

    // Getters for properties
    double getCarrierFrequency() const { return carrierFrequency; }
    double getModulatorFrequency() const { return modulatorFrequency; }
    double getModulationIndex() const { return modulationIndex; }
    SidebandMode getSidebandMode() const { return sidebandMode; }
    SpectrumRange getSpectrumRange() const { return spectrumRange; }

private:
    double carrierFrequency;     // Carrier frequency
    double modulatorFrequency;   // Modulator frequency
    double modulationIndex;      // Modulation index (beta)
    SidebandMode sidebandMode;   // Which sidebands to include
    SpectrumRange spectrumRange; // Full spectrum or positive only
    size_t maxSidebandCount;     // Maximum number of sidebands to calculate

    std::vector<double> sidebandAmplitudes; // Cached sideband amplitudes

    // Bessel function lookup table (shared among all instances)
    static BesselLookup besselLookup;

    // Calculate Bessel function of first kind, order n
    double besselJ(int n, double x);
    
    // Update sideband amplitudes based on current parameters
    void updateSidebandAmplitudes();
    
    // Check if a sideband should be included based on the mode
    bool includeSideband(int sidebandNumber);
};

#endif // FM_OPERATOR_H