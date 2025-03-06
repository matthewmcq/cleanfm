#include "fmdecomposer.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <numeric>

// Helper function to generate complex spectrum from an FM layer
std::vector<Complex> ComplexFMLayer::generateComplexSpectrum(size_t sampleRate, size_t fftSize) const {
    std::vector<Complex> spectrum(fftSize, Complex(0, 0));

    // Get the raw FM operator spectrum with complex sidebands
    auto fmSpectrum = fmOperator->generateSpectrum();

    // Apply base amplitude (magnitude and phase) to each sideband
    for (const auto& [freq, complexAmp] : fmSpectrum) {
        // Convert frequency to bin
        size_t bin = static_cast<size_t>(freq * fftSize / sampleRate);

        // Ensure bin is within bounds
        if (bin < fftSize) {
            // Combine operator sideband with base amplitude
            spectrum[bin] = baseAmplitude * complexAmp;
        }
    }

    return spectrum;
}

// Constructor
PhaseCoherentFMDecomposer::PhaseCoherentFMDecomposer(const std::vector<CleanDFT::Component>& components,
                                                   size_t sampleRate, size_t fftSize)
    : originalComponents(components),
      residualComponents(components),
      sampleRate(sampleRate),
      fftSize(fftSize) {

    totalEnergy = calculateTotalEnergy(components);
    std::cout << "Total energy of original components: " << totalEnergy << std::endl;
}

// Main decomposition function
std::vector<ComplexFMLayer> PhaseCoherentFMDecomposer::decompose(
    double energyThreshold, double phaseToleranceRadians, size_t maxOperators) {

    double currentEnergyRatio = 0.0;
    size_t operatorCount = 0;

    while (currentEnergyRatio < energyThreshold &&
           operatorCount < maxOperators &&
           !residualComponents.empty()) {

        // Find phase coherent groups
        auto groups = findPhaseCoherentGroups(phaseToleranceRadians);

        for (const auto& group : groups) {
            std::cout << "base freq: " << group.baseFrequency << std::endl;
            std::cout << "cohesion: " << group.harmonicCoherence << std::endl;
        }
        if (groups.empty()) {
            break;
        }

        // Sort groups by total energy
        std::sort(groups.begin(), groups.end(),
                 [](const PhaseCoherentGroup& a, const PhaseCoherentGroup& b) {
                     return a.totalEnergy > b.totalEnergy;
                 });

        // Try to find an FM operator for the highest energy group
        auto layer = findOptimalFMOperator(groups[0]);
        std::cout << "Layer correlation score: " << layer.correlationScore << std::endl;

        if (layer.correlationScore > 0.7) {  // Threshold can be adjusted
            fmLayers.push_back(layer);

            // Remove the contribution of this FM layer
            subtractFMLayer(layer);

            operatorCount++;

            // Update energy ratio
            currentEnergyRatio = getCapturedEnergyRatio();

            std::cout << "Added FM operator " << operatorCount
                     << " with carrier " << layer.fmOperator->getCarrierFrequency()
                     << " Hz, modulator " << layer.fmOperator->getModulatorFrequency()
                     << " Hz, index " << layer.fmOperator->getModulationIndex()
                     << ", current energy ratio: " << currentEnergyRatio * 100 << "%" << std::endl;
        } else {
            std::cout << "No good FM operator fit found for the highest energy group" << std::endl;
            break;
        }
    }

    return fmLayers;
}

// Find groups of components with phase coherence
std::vector<PhaseCoherentGroup> PhaseCoherentFMDecomposer::findPhaseCoherentGroups(double phaseToleranceRadians) {
    std::vector<PhaseCoherentGroup> groups;
    std::vector<bool> assigned(residualComponents.size(), false);

    // First, sort components by amplitude (largest first)
    auto sortedComponents = residualComponents;
    std::sort(sortedComponents.begin(), sortedComponents.end(),
             [](const CleanDFT::Component& a, const CleanDFT::Component& b) {
                 return a.amplitude > b.amplitude;
             });

    // Potential fundamental frequencies to check
    std::unordered_set<double> potentialFundamentals;

    // Add the frequencies of the top components as potential fundamentals
    for (size_t i = 0; i < std::min(size_t(10), sortedComponents.size()); ++i) {
        double freqHz = sortedComponents[i].true_frequency * sampleRate / fftSize;
        potentialFundamentals.insert(freqHz);

        // Also add potential sub-harmonics (fundamental might be lower)
        for (int div = 2; div <= 8; ++div) {
            double subFreq = freqHz / div;
            if (subFreq > 20.0) {  // Above 20 Hz (audible range)
                potentialFundamentals.insert(subFreq);
            }
        }
    }

    // For each potential fundamental, try to form a group
    for (double fundFreq : potentialFundamentals) {
        PhaseCoherentGroup group;
        group.baseFrequency = fundFreq;
        group.totalEnergy = 0.0;

        // Find components that could belong to this harmonic series
        for (size_t i = 0; i < residualComponents.size(); ++i) {
            if (assigned[i]) continue;

            const auto& comp = residualComponents[i];
            double compFreqHz = comp.true_frequency * sampleRate / fftSize;

            // Check if this component is close to a harmonic of the fundamental
            for (int harmonic = 2; harmonic <= 20; ++harmonic) {
                double harmonicFreq = fundFreq * harmonic;

                // If frequency is close to a harmonic (within 2%)
                if (std::abs(compFreqHz - harmonicFreq) < 0.02 * harmonicFreq) {
                    // If this is the first component or the phase is coherent
                    if (group.components.empty()) {
                        // First component sets the reference phase
                        group.basePhase = comp.true_phase;
                        group.components.push_back(comp);
                        group.totalEnergy += comp.amplitude * comp.amplitude;
                        assigned[i] = true;
                    } else {
                        // Calculate expected phase for this harmonic
                        // In a phase-coherent FM, phases of harmonics have specific relationships
                        // For simplicity, we'll check if phase is within tolerance

                        double phaseDiff = std::fmod(comp.true_phase - group.basePhase + M_PI, 2 * M_PI) - M_PI;

                        if (std::abs(phaseDiff) <= phaseToleranceRadians) {
                            group.components.push_back(comp);
                            group.totalEnergy += comp.amplitude * comp.amplitude;
                            assigned[i] = true;
                        }
                    }

                    break; // Found a match for this component
                }
            }
        }

        // Only keep groups with at least 3 components
        if (group.components.size() >= 3) {
            // Calculate harmonic coherence score
            identifyHarmonicSeries(group);
            groups.push_back(group);
        }
    }

    return groups;
}

// Identify harmonic series within a group
void PhaseCoherentFMDecomposer::identifyHarmonicSeries(PhaseCoherentGroup& group) {
    double freqTolerance = 0.01 * group.baseFrequency; // 2% tolerance

    // Count how many harmonics are present
    std::vector<bool> harmonicPresent(21, false); // Up to 20th harmonic

    for (const auto& comp : group.components) {
        double compFreqHz = comp.true_frequency * sampleRate / fftSize;

        for (int h = 1; h <= 20; ++h) {
            double harmonicFreq = group.baseFrequency * h;

            if (std::abs(compFreqHz - harmonicFreq) < freqTolerance) {
                harmonicPresent[h] = true;
                break;
            }
        }
    }

    // Count consecutive harmonics
    int consecutiveCount = 0;
    int maxConsecutive = 0;

    for (int h = 1; h <= 20; ++h) {
        if (harmonicPresent[h]) {
            consecutiveCount++;
            maxConsecutive = std::max(maxConsecutive, consecutiveCount);
        } else {
            consecutiveCount = 0;
        }
    }

    // Calculate coherence score based on consecutive harmonics and total count
    int totalHarmonics = std::count(harmonicPresent.begin(), harmonicPresent.end(), true);
    group.harmonicCoherence = (0.7 * maxConsecutive + 0.3 * totalHarmonics) / 20.0;
}

// Find optimal FM operator for a phase coherent group
ComplexFMLayer PhaseCoherentFMDecomposer::findOptimalFMOperator(const PhaseCoherentGroup& group) {
    ComplexFMLayer bestLayer;
    bestLayer.correlationScore = 0.0;

    // Target spectrum for this group
    std::vector<Complex> targetSpectrum = componentsToSpectrum(group.components);

    // Use the base frequency as carrier
    double carrierFreq = group.baseFrequency;

    // Try different modulator frequencies
    double bestModFreq = carrierFreq; //findOptimalModulatorFrequency(carrierFreq, group);

    // Find optimal modulation index for this carrier/modulator pair
    double bestModIndex = findOptimalModulationIndex(carrierFreq, bestModFreq, group);

    // Create FM operator with optimal parameters
    auto fmOperator = std::make_shared<FMOperator>(
        carrierFreq,
        bestModFreq,
        bestModIndex,
        SidebandMode::ALL_SIDEBANDS,
        SpectrumRange::POSITIVE_ONLY,
        20  // Max sidebands
    );

    // Calculate correlation with target spectrum
    std::vector<Complex> fmSpectrum(fftSize, Complex(0, 0));
    auto rawFmSpectrum = fmOperator->generateSpectrum();

    for (const auto& [freq, complexAmp] : rawFmSpectrum) {
        size_t bin = static_cast<size_t>(freq * fftSize / sampleRate);
        if (bin < fftSize) {
            fmSpectrum[bin] = complexAmp;
        }
    }

    double correlation = calculateComplexCorrelation(targetSpectrum, fmSpectrum);

    if (correlation > bestLayer.correlationScore) {
        bestLayer.fmOperator = fmOperator;
        bestLayer.correlationScore = correlation;

        // Find the component at the carrier frequency to determine base amplitude
        const auto& carrierComp = *std::find_if(
            group.components.begin(),
            group.components.end(),
            [this, carrierFreq](const CleanDFT::Component& comp) {
                double compFreqHz = comp.true_frequency * sampleRate / fftSize;
                return std::abs(compFreqHz - carrierFreq) < 0.02 * carrierFreq;
            }
        );

        // Base amplitude includes magnitude and phase
        bestLayer.baseAmplitude = Complex(
            carrierComp.amplitude * std::cos(carrierComp.true_phase),
            carrierComp.amplitude * std::sin(carrierComp.true_phase)
        );
    }

    return bestLayer;
}

// Find optimal modulator frequency for a given carrier
double PhaseCoherentFMDecomposer::findOptimalModulatorFrequency(
    double carrierFreq, const PhaseCoherentGroup& group, double minRatio, double maxRatio) {

    // Target spectrum
    std::vector<Complex> targetSpectrum = componentsToSpectrum(group.components);

    // Try different modulator frequencies (as ratios of carrier)
    double bestModFreq = carrierFreq; // Default is 1:1 ratio
    double bestCorrelation = 0.0;

    // Common FM ratios to check (including 1:1)
    std::vector<double> ratios = {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0};

    for (double ratio : ratios) {
        if (ratio < minRatio || ratio > maxRatio) continue;

        double modFreq = carrierFreq * ratio;

        // For each modulator frequency, find the best modulation index
        double modIndex = findOptimalModulationIndex(carrierFreq, modFreq, group);

        // Create test FM operator
        FMOperator testOperator(
            carrierFreq,
            modFreq,
            modIndex,
            SidebandMode::ALL_SIDEBANDS,
            SpectrumRange::POSITIVE_ONLY
        );

        // Generate spectrum and calculate correlation
        std::vector<Complex> fmSpectrum(fftSize, Complex(0, 0));
        auto rawFmSpectrum = testOperator.generateSpectrum();

        for (const auto& [freq, complexAmp] : rawFmSpectrum) {
            size_t bin = static_cast<size_t>(freq * fftSize / sampleRate);
            if (bin < fftSize) {
                fmSpectrum[bin] = complexAmp;
            }
        }

        double correlation = calculateComplexCorrelation(targetSpectrum, fmSpectrum);

        if (correlation > bestCorrelation) {
            bestCorrelation = correlation;
            bestModFreq = modFreq;
        }
    }

    std::cout << "Best modulator frequency for carrier " << carrierFreq
             << " Hz: " << bestModFreq << " Hz (ratio: " << bestModFreq / carrierFreq
             << "), correlation: " << bestCorrelation << std::endl;

    return bestModFreq;
}

// Find optimal modulation index
double PhaseCoherentFMDecomposer::findOptimalModulationIndex(
    double carrierFreq, double modulatorFreq, const PhaseCoherentGroup& group) {

    // Target spectrum
    std::vector<Complex> targetSpectrum = componentsToSpectrum(group.components);

    // Try different modulation indices
    double bestIndex = 0.0;
    double bestCorrelation = 0.0;

    // Test a range of indices from 0.1 to 10.0
    for (double idx = 0.1; idx <= 20.0; idx += 0.01) {
        // Create test FM operator
        FMOperator testOperator(
            carrierFreq,
            modulatorFreq,
            idx,
            SidebandMode::ALL_SIDEBANDS,
            SpectrumRange::POSITIVE_ONLY
        );

        // Generate spectrum
        std::vector<Complex> fmSpectrum(fftSize, Complex(0, 0));
        auto rawFmSpectrum = testOperator.generateSpectrum();

        for (const auto& [freq, complexAmp] : rawFmSpectrum) {
            size_t bin = static_cast<size_t>(freq * fftSize / sampleRate);
            if (bin < fftSize) {
                fmSpectrum[bin] = complexAmp;
            }
        }

        // Calculate correlation
        double correlation = calculateComplexCorrelation(targetSpectrum, fmSpectrum);

        if (correlation > bestCorrelation) {
            std::cout << "best correlation MI: " << correlation << std::endl;
            std::cout << "best index MI: " << idx << std::endl;
            bestCorrelation = correlation;
            bestIndex = idx;
        }
    }

    return bestIndex;
}

// Calculate complex correlation between two spectra
double PhaseCoherentFMDecomposer::calculateComplexCorrelation(
    const std::vector<Complex>& spectrum1, const std::vector<Complex>& spectrum2) {

    Complex numerator = 0;
    double norm1 = 0;
    double norm2 = 0;

    // Only consider first half of spectrum (up to Nyquist)
    size_t len = std::min(spectrum1.size(), spectrum2.size()) / 2;

    for (size_t i = 0; i < len; ++i) {
        numerator += spectrum1[i] * std::conj(spectrum2[i]);
        norm1 += std::norm(spectrum1[i]);
        norm2 += std::norm(spectrum2[i]);
    }

    if (norm1 == 0 || norm2 == 0) {
        return 0.0;
    }

    double ret = std::abs(numerator) / (std::sqrt(norm1) * std::sqrt(norm2));
    // std::cout << ret << std::endl;
    return ret;
}

// Convert components to complex spectrum
std::vector<Complex> PhaseCoherentFMDecomposer::componentsToSpectrum(
    const std::vector<CleanDFT::Component>& components) {

    std::vector<Complex> spectrum(fftSize, Complex(0, 0));

    for (const auto& comp : components) {
        size_t bin = static_cast<size_t>(comp.true_frequency);

        if (bin < fftSize) {
            // Convert amplitude and phase to complex form
            spectrum[bin] = Complex(
                comp.amplitude * std::cos(comp.true_phase),
                comp.amplitude * std::sin(comp.true_phase)
            );
        }
    }

    return spectrum;
}

// Subtract FM layer from residual components
void PhaseCoherentFMDecomposer::subtractFMLayer(const ComplexFMLayer& layer) {
    // Generate complex spectrum from the FM layer
    std::vector<Complex> layerSpectrum = layer.generateComplexSpectrum(sampleRate, fftSize);

    // New set of residual components after subtraction
    std::vector<CleanDFT::Component> newResiduals;

    // For each residual component
    for (const auto& comp : residualComponents) {
        size_t bin = static_cast<size_t>(comp.true_frequency);

        if (bin < fftSize) {
            // Convert component to complex form
            Complex compComplex(
                comp.amplitude * std::cos(comp.true_phase),
                comp.amplitude * std::sin(comp.true_phase)
            );

            // Subtract FM contribution
            Complex residual = compComplex - layerSpectrum[bin];

            // If significant residual remains
            if (std::abs(residual) > 0.01 * comp.amplitude) {
                // Convert back to amplitude/phase form
                CleanDFT::Component newComp;
                newComp.true_frequency = comp.true_frequency;
                newComp.amplitude = std::abs(residual);
                newComp.true_phase = std::arg(residual);

                newResiduals.push_back(newComp);
            }
        } else {
            // Keep components outside FFT size untouched
            newResiduals.push_back(comp);
        }
    }

    // Update residual components
    residualComponents = newResiduals;
}

// Calculate total energy
double PhaseCoherentFMDecomposer::calculateTotalEnergy(
    const std::vector<CleanDFT::Component>& components) const {
    
    return std::accumulate(components.begin(), components.end(), 0.0,
        [](double sum, const CleanDFT::Component& comp) {
            return sum + comp.amplitude * comp.amplitude;
        });
}

// Get energy capture ratio
double PhaseCoherentFMDecomposer::getCapturedEnergyRatio() const {
    double residualEnergy = calculateTotalEnergy(residualComponents);
    return 1.0 - (residualEnergy / totalEnergy);
}

// Get residual components
std::vector<CleanDFT::Component> PhaseCoherentFMDecomposer::getResidualComponents() const {
    return residualComponents;
}

// Reconstruct audio from the FM layers
std::vector<double> PhaseCoherentFMDecomposer::reconstructAudio(size_t numSamples) const {
    std::vector<double> output(numSamples, 0.0);
    
    for (const auto& layer : fmLayers) {
        double carrierFreq = layer.fmOperator->getCarrierFrequency();
        double modulatorFreq = layer.fmOperator->getModulatorFrequency();
        double modulationIndex = layer.fmOperator->getModulationIndex();
        double amplitude = std::abs(layer.baseAmplitude);
        double initialPhase = std::arg(layer.baseAmplitude);
        
        // Synthesize this FM operator
        for (size_t i = 0; i < numSamples; ++i) {
            double t = static_cast<double>(i) / sampleRate;
            
            // FM synthesis formula with phase coherence
            double modulation = modulationIndex * std::sin(2.0 * M_PI * modulatorFreq * t);
            double signal = amplitude * std::sin(2.0 * M_PI * carrierFreq * t + modulation + initialPhase);
            
            output[i] += signal;
        }
    }
    
    return output;
}