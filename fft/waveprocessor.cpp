/**
 * @file waveprocessor.cpp
 * @author Matthew McQuistion
 * @date 12/17/24
 * @brief Implements functions for reading, processing (FFT/IFFT), and writing WAV audio files.
 *
 * This file provides the implementation for the WaveProcessor class, utilizing
 * the dr_wav library for WAV file I/O and the FFTW library for Fast Fourier
 * Transform and Inverse Fast Fourier Transform operations. It includes functions
 * to read WAV files into a structured AudioData format, perform FFT on audio
 * samples, perform IFFT on frequency spectra, and write audio data back to a
 * WAV file.
 */

#include "waveprocessor.h" // Include the header file for the WaveProcessor class

#include <iostream> // For standard input/output operations (e.g., cerr, cout)
#include <vector>   // For using std::vector to store audio samples and spectrum data
#include <stdexcept> // For throwing std::runtime_error

// Include the dr_wav library implementation. This should typically be done in
// one source file in your project.
#define DR_WAV_IMPLEMENTATION
#include "../libs/dr_wav.h"

/**
 * @brief Reads a WAV audio file from the specified filename.
 *
 * Initializes the dr_wav library to open and read the WAV file. It extracts
 * the sample rate, number of channels, and total PCM frame count. The raw
 * sample data is read, converted to double-precision floating-point values
 * normalized between -1.0 and 1.0, and then separated into left and right
 * channels. Supports mono (1 channel) and stereo (2 channels) WAV files.
 *
 * @param filename The path to the WAV file to read.
 * @return An AudioData struct containing the sample rate, number of channels,
 * and the audio samples separated into left and right channels.
 * @throws std::runtime_error if the WAV file cannot be opened or if the
 * number of channels is not 1 or 2.
 */
AudioData WaveProcessor::readWav(const char *filename) {
    drwav wav; // dr_wav structure to hold WAV file information

    // Initialize dr_wav for reading the specified file.
    // NULL is passed for the optional allocation callbacks.
    if (!drwav_init_file(&wav, filename, NULL)) {
        // If initialization fails, throw a runtime error.
        throw std::runtime_error("Failed to open WAV file");
    }

    AudioData audio; // Structure to hold the processed audio data
    audio.sampleRate = wav.sampleRate; // Store the sample rate
    audio.channels = wav.channels;     // Store the number of channels

    // Read samples. drwav_read_pcm_frames_s16 reads samples as 16-bit integers.
    // rawSamples will store interleaved samples if stereo (L R L R ...).
    std::vector<int16_t> rawSamples(wav.totalPCMFrameCount * wav.channels);
    drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, rawSamples.data());

    // Separate channels and convert to double (normalized -1.0 to 1.0).
    const size_t frames = wav.totalPCMFrameCount; // Number of audio frames (samples per channel)
    audio.leftChannel.resize(frames);             // Resize left channel vector
    audio.rightChannel.resize(frames);            // Resize right channel vector

    if (wav.channels == 2) {
        // Stereo data: samples are interleaved (Left, Right, Left, Right, ...)
        for (size_t i = 0; i < frames; i++) {
            // Convert 16-bit integer to double (-1.0 to 1.0) by dividing by max value (32768.0)
            audio.leftChannel[i] = static_cast<double>(rawSamples[i * 2]) / 32768.0;
            audio.rightChannel[i] = static_cast<double>(rawSamples[i * 2 + 1]) / 32768.0;
        }
    } else if (wav.channels == 1) {
        // Mono data: copy the single channel to both left and right channels.
        for (size_t i = 0; i < frames; i++) {
            // Convert 16-bit integer to double (-1.0 to 1.0)
            audio.leftChannel[i] = audio.rightChannel[i] = static_cast<double>(rawSamples[i]) / 32768.0;
        }
    } else {
        // Throw an error for unsupported channel counts.
        drwav_uninit(&wav); // Clean up dr_wav resources before throwing
        throw std::runtime_error("Unsupported number of channels");
    }

    // Clean up dr_wav resources.
    drwav_uninit(&wav);

    // Return the populated AudioData struct.
    return audio;
}

/**
 * @brief Computes the Fast Fourier Transform (FFT) of a vector of real samples.
 *
 * Uses the FFTW library to perform a 1D forward DFT. The input samples are
 * treated as the real part of complex numbers with zero imaginary parts.
 * The output is a vector of complex numbers representing the frequency spectrum.
 * Note: This function assumes the input size is suitable for FFTW (e.g., power of 2).
 *
 * @param samples A vector of double-precision floating-point values representing
 * the time-domain audio samples.
 * @return A vector of Complex numbers representing the frequency spectrum.
 * The size of the output vector is the same as the input vector.
 */
std::vector<Complex> WaveProcessor::computeFFT(const std::vector<double> &samples) {
    size_t N = samples.size(); // Get the number of samples (size of the transform)

    // Allocate memory for FFTW input and output arrays.
    // fftw_complex is a double[2] where index 0 is real and index 1 is imaginary.
    fftw_complex *in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);

    // Copy samples to the FFTW input array, treating them as real numbers.
    for (size_t i = 0; i < N; i++) {
        in[i][0] = samples[i]; // Real part is the sample value
        in[i][1] = 0.0;       // Imaginary part is zero
    }

    // Create an FFTW plan for a 1D forward DFT.
    // FFTW_ESTIMATE is a flag that tells FFTW to choose an algorithm quickly
    // without spending much time on initialization. Other flags like FFTW_MEASURE
    // can find potentially faster algorithms but take longer to plan.
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the planned FFT.
    fftw_execute(plan);

    // Copy the results from the FFTW output array to a vector of Complex objects.
    std::vector<Complex> spectrum(N); // Vector to store the resulting spectrum
    for (size_t i = 0; i < N; i++) {
        spectrum[i] = Complex(out[i][0], out[i][1]); // Create Complex object from real and imaginary parts
    }

    // Clean up FFTW resources.
    fftw_destroy_plan(plan); // Destroy the plan
    fftw_free(in);           // Free input memory
    fftw_free(out);          // Free output memory

    // Return the computed frequency spectrum.
    return spectrum;
}

/**
 * @brief Computes the Inverse Fast Fourier Transform (IFFT) of a frequency spectrum.
 *
 * Uses the FFTW library to perform a 1D backward DFT. The input is a vector
 * of Complex numbers representing the frequency spectrum. The output is a
 * vector of double-precision floating-point values representing the time-domain
 * samples. The result is normalized by the size of the transform (N).
 *
 * @param spectrum A vector of Complex numbers representing the frequency spectrum.
 * @return A vector of double-precision floating-point values representing the
 * reconstructed time-domain audio samples. The size is the same as
 * the input spectrum vector.
 */
std::vector<double> WaveProcessor::computeIFFT(const std::vector<Complex> &spectrum) {
    size_t N = spectrum.size(); // Get the size of the spectrum (size of the transform)
    // std::cout << "N = " << N << std::endl; // Debug output

    // Allocate memory for FFTW input and output arrays.
    fftw_complex *in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);

    // Copy the input spectrum (Complex objects) to the FFTW input array.
    for (size_t i = 0; i < N; i++) {
        in[i][0] = spectrum[i].real(); // Copy the real part
        in[i][1] = spectrum[i].imag(); // Copy the imaginary part
    }

    // Create an FFTW plan for a 1D backward DFT.
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute the planned IFFT.
    fftw_execute(plan);

    // Convert the results from the FFTW output array to a vector of doubles
    // and normalize by N. The IFFT result needs to be scaled by 1/N.
    std::vector<double> result(N); // Vector to store the resulting samples
    for (size_t i = 0; i < N; i++) {
        result[i] = out[i][0] / static_cast<double>(N); // Get the real part and normalize
        // The imaginary part (out[i][1]) should theoretically be zero for real input,
        // but might have small floating-point errors. We only take the real part.
    }

    // Clean up FFTW resources.
    fftw_destroy_plan(plan); // Destroy the plan
    fftw_free(in);           // Free input memory
    fftw_free(out);          // Free output memory

    // Return the reconstructed time-domain samples.
    return result;
}

/**
 * @brief Writes audio samples to a WAV file.
 *
 * Initializes the dr_wav library for writing a WAV file with the specified
 * format (PCM, 16 bits per sample). The input samples (double-precision,
 * normalized -1.0 to 1.0) are converted back to 16-bit integers. If the
 * output is stereo but the input is mono, the mono samples are duplicated
 * for both left and right channels.
 *
 * @param filename The path where the WAV file will be written.
 * @param samples A vector of double-precision floating-point values representing
 * the time-domain audio samples (assumed to be a single channel
 * if channels is 1, or interleaved if channels is 2, although
 * the current implementation assumes mono input for stereo output).
 * @param sampleRate The sample rate for the output WAV file.
 * @param channels The number of channels for the output WAV file (1 for mono, 2 for stereo).
 * @throws std::runtime_error if the WAV file cannot be opened for writing.
 */
void WaveProcessor::writeWav(const char *filename, const std::vector<double> &samples,
                           int sampleRate, int channels) {
    drwav_data_format format; // Structure to define the output WAV format
    format.container = drwav_container_riff; // Use the standard RIFF container
    format.format = DR_WAVE_FORMAT_PCM;     // Use PCM format
    format.channels = channels;             // Set the number of channels
    format.sampleRate = sampleRate;         // Set the sample rate
    format.bitsPerSample = 16;              // Set bits per sample to 16

    std::cout << "Writing WAV file " << filename << std::endl; // Log output
    std::cout << "Sample rate: " << sampleRate << std::endl; // Log output
    std::cout << "Channels: " << channels << std::endl;     // Log output

    drwav wav; // dr_wav structure for writing

    // Initialize dr_wav for writing the specified file with the given format.
    // NULL is passed for the optional allocation callbacks.
    if (!drwav_init_file_write(&wav, filename, &format, NULL)) {
        // If initialization fails, throw a runtime error.
        throw std::runtime_error("Failed to open WAV file for writing");
    }

    // Convert double samples (-1.0 to 1.0) to 16-bit integers (-32768 to 32767).
    // Account for channels: if stereo output is requested from mono input,
    // duplicate the mono samples.
    const size_t frames = samples.size();  // Number of frames (samples per channel in the input vector)
    std::vector<int16_t> pcm(frames * channels);  // Vector to hold the 16-bit PCM samples (interleaved if stereo)

    if (channels == 1) {
        // Mono output: Convert each double sample directly to int16.
        for (size_t i = 0; i < frames; i++) {
            // Scale by max 16-bit value (32767.0) and cast to int16.
            // Use 32767.0 for positive scaling, int16_t range is -32768 to 32767.
            pcm[i] = static_cast<int16_t>(samples[i] * 32767.0);
        }
    } else if (channels == 2) {
        // Stereo output requested. Assumes input 'samples' is a single mono channel.
        // Duplicate the mono sample for both left and right channels.
        for (size_t i = 0; i < frames; i++) {
            // Scale and cast for Left channel
            pcm[i * 2] = static_cast<int16_t>(samples[i] * 32767.0);
            // Scale and cast for Right channel (same value as Left)
            pcm[i * 2 + 1] = static_cast<int16_t>(samples[i] * 32767.0);
        }
    }
    // Note: If the input 'samples' vector already contained interleaved stereo data,
    // the logic here would need to be different to handle that structure.
    // The current implementation assumes 'samples' is always a single channel's data.

    // Write the PCM frames to the WAV file.
    // The number of frames to write is the number of samples in the input vector.
    drwav_write_pcm_frames(&wav, frames, pcm.data());

    // Clean up dr_wav resources.
    drwav_uninit(&wav);
}
