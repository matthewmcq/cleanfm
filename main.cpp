#include <iostream>
#include "fft/waveprocessor.h"
#include "deconvolve/cleandft.h"
#include <chrono>
#include <fstream>

void writeComponentsToCSV(const std::vector<CleanDFT::Component>& components,
                          const size_t N,
                          const size_t sample_rate,
                          const std::string& filename) {
    std::ofstream file(filename);
    file << "frequency_hz,phase,amplitude\n";  // CSV header

    for (const auto& comp : components) {
        // Convert frequency from bins to Hz
        double freq_hz = comp.true_frequency * sample_rate / N;
        file << freq_hz << ","
             << comp.true_phase << ","
             << comp.amplitude << "\n";
    }
    file.close();
}

int main() {
    const auto start = std::chrono::high_resolution_clock::now();

    // Load audio file
    const char *input_file = "examples/TEST_INTRO_SHORT.wav";
    // std::string input_file = "examples/TEST_INTRO_SHORT.wav";
    bool use_parallel = true;
    int num_threads = std::thread::hardware_concurrency();

    // Print configuration
    std::cout << "Configuration:\n"
            << "  Input file: " << input_file << "\n"
            << "  Processing mode: " << (use_parallel ? "Parallel" : "Sequential") << "\n"
            << "  Threads: " << num_threads << "\n"
            << std::endl;


    auto [leftChannel, rightChannel, sampleRate, channels]
            = WaveProcessor::readWav(input_file);
    std::cout << "Loaded audio: "
            << channels << " channels, "
            << leftChannel.size() << " samples at "
            << sampleRate << " Hz\n";

    // Compute FFT for left channel
    const std::vector<Complex> spectrum = WaveProcessor::computeFFT(leftChannel);

    std::vector<CleanDFT::Component> components = std::vector<CleanDFT::Component>();


    if (use_parallel) {
        std::cout << "Using parallel decomposition with " << num_threads << " threads" << std::endl;
        components = CleanDFT::deconvolveParallelDirichlet(spectrum, sampleRate, num_threads);
    } else {
        std::cout << "Using sequential decomposition" << std::endl;
        components = CleanDFT::deconvolveDirichletKernel(spectrum, sampleRate);
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << " ms\n";
    writeComponentsToCSV(components, spectrum.size(), sampleRate, "components.csv");

    const std::vector<double> reconstructed = CleanDFT::decompressComponents(components, spectrum.size());
    WaveProcessor::writeWav("reconstructed.wav", reconstructed, sampleRate);

    return 0;
    // Use the dual-modulator FM-Dirichlet deconvolution
    // const std::vector<CleanDFT::FMComponent> fm_components =
    //     CleanDFT::deconvolveFMDirichletKernel(spectrum, sampleRate);
    //
    // // Save FM components to CSV for analysis
    // CleanDFT::writeFMComponentsToCSV(fm_components, spectrum.size(), sampleRate, "fm_components.csv");
    //
    // // Generate audio from FM components
    // const std::vector<double> fm_reconstructed =
    //     CleanDFT::decompressFMComponents(fm_components, spectrum.size());
    //
    // // Write FM reconstructed audio to file
    // WaveProcessor::writeWav("fm_reconstructed.wav", fm_reconstructed, sampleRate);
    //
    // // Print summary
    // std::cout << "Number of FM components: " << fm_components.size() << std::endl;
    //
    // // For comparison, also run the original Dirichlet deconvolution
    // // const std::vector<CleanDFT::Component> components =
    // //     CleanDFT::deconvolveDirichletKernel(spectrum, sampleRate);
    //
    // std::cout << "Number of original components: " << components.size() << std::endl;
    //
    // // Calculate compression ratio
    // double compression_ratio = static_cast<double>(components.size()) / fm_components.size();
    // std::cout << "Compression ratio: " << compression_ratio << ":1" << std::endl;
    //
    // const auto stop = std::chrono::high_resolution_clock::now();
    // const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    // std::cout << duration.count() << " ms\n";
    //
    // return 0;
}
