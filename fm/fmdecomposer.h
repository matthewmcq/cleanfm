#ifndef PHASE_COHERENT_FM_DECOMPOSER_H
#define PHASE_COHERENT_FM_DECOMPOSER_H

#include <vector>
#include <map>
#include <complex>
#include <memory>
#include "fmoperator.h"
#include "../deconvolve/cleandft.h"

// Represents a group of components with phase coherence
struct PhaseCoherentGroup {
    std::vector<CleanDFT::Component> components;
    double baseFrequency;    // Potential fundamental frequency
    double basePhase;        // Reference phase
    double totalEnergy;

    // Calculate how well the components align with harmonic series of the baseFrequency
    double harmonicCoherence;
};

// Represents an FM operator with complex amplitudes
struct ComplexFMLayer {
    std::shared_ptr<FMOperator> fmOperator;
    Complex baseAmplitude;   // Complex amplitude with magnitude and phase
    double correlationScore;

    // Generate complex spectrum at specific sample rate and FFT size
    std::vector<Complex> generateComplexSpectrum(size_t sampleRate, size_t fftSize) const;
};

class PhaseCoherentFMDecomposer {
public:
    PhaseCoherentFMDecomposer(const std::vector<CleanDFT::Component>& components,
                            size_t sampleRate, size_t fftSize);

    // Run the decomposition process
    std::vector<ComplexFMLayer> decompose(double energyThreshold = 0.95,
                                        double phaseToleranceRadians = 0.3,
                                        size_t maxOperators = 20);

    // Get the residual components after decomposition
    std::vector<CleanDFT::Component> getResidualComponents() const;

    // Calculate the energy captured by the current FM operators
    double getCapturedEnergyRatio() const;

    // Reconstruct audio from the FM layers
    std::vector<double> reconstructAudio(size_t numSamples) const;

private:
    std::vector<CleanDFT::Component> originalComponents;
    std::vector<CleanDFT::Component> residualComponents;
    std::vector<ComplexFMLayer> fmLayers;

    size_t sampleRate;
    size_t fftSize;
    double totalEnergy;

    // Find phase coherent groups using harmonic relationships
    std::vector<PhaseCoherentGroup> findPhaseCoherentGroups(double phaseToleranceRadians);

    // Identify potential harmonic series within each group
    void identifyHarmonicSeries(PhaseCoherentGroup& group);

    // Find the optimal FM operator for a coherent group
    ComplexFMLayer findOptimalFMOperator(const PhaseCoherentGroup& group);

    // Calculate complex correlation between two spectra
    double calculateComplexCorrelation(const std::vector<Complex>& spectrum1,
                                     const std::vector<Complex>& spectrum2);

    // Subtract FM layer contribution from residual components
    void subtractFMLayer(const ComplexFMLayer& layer);

    // Convert components to complex spectrum
    std::vector<Complex> componentsToSpectrum(const std::vector<CleanDFT::Component>& components);

    // Calculate total energy of components
    double calculateTotalEnergy(const std::vector<CleanDFT::Component>& components) const;

    // Search for optimal modulation index
    double findOptimalModulationIndex(double carrierFreq, double modulatorFreq,
                                    const PhaseCoherentGroup& group);

    // Search for optimal modulator frequency
    double findOptimalModulatorFrequency(double carrierFreq,
                                       const PhaseCoherentGroup& group,
                                       double minRatio = 0.5,
                                       double maxRatio = 2.0);
};

#endif // PHASE_COHERENT_FM_DECOMPOSER_H