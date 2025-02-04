#include <iostream>
#include "fft/waveprocessor.h"
#include "deconvolve/cleandft.h"
#include <chrono>

int main() {
    const auto start = std::chrono::high_resolution_clock::now();
    auto [leftChannel, rightChannel, sampleRate, channels]
            = WaveProcessor::readWav("examples/TEST_INTRO_SHORT.wav");
    std::cout << "Loaded audio: "
            << channels << " channels, "
            << leftChannel.size() << " samples at "
            << sampleRate << " Hz\n";

    // Compute FFT for left channel only
    const std::vector<Complex> spectrum = WaveProcessor::computeFFT(rightChannel);
    const Complex DC = spectrum[0];


    const std::vector<CleanDFT::Component> components = CleanDFT::deconvolveDirichletKernel(spectrum, sampleRate);

    std::cout << "FFT: " << spectrum.size() << " spectrum\n";

    constexpr size_t target_sr = 22050; //44100;
    const auto resampled = CleanDFT::resample(components, spectrum.size(), sampleRate, target_sr, DC);
    WaveProcessor::writeWav("resampled.wav", resampled, target_sr);

    const std::vector<double> reconstructed = CleanDFT::decompressComponents(
        components, spectrum.size(), DC
    );

    WaveProcessor::writeWav("reconstructed_Nyquist_real0.wav", reconstructed, sampleRate);

    const auto stop = std::chrono::high_resolution_clock::now();

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << duration.count() << " ms\n";
}
