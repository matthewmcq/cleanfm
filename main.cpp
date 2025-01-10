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


    const std::vector<CleanDFT::Component> components = CleanDFT::deconvolveDirichletKernel(spectrum, sampleRate);

    std::cout << "FFT: " << spectrum.size() << " spectrum\n";

    const auto stop = std::chrono::high_resolution_clock::now();

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << duration.count() << " ms\n";
}
