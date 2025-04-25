/**
 * @file waveprocessor.h
 * @author Matthew McQuistion
 * @date 12/17/24
 * @brief Declares the WaveProcessor class and supporting structures for WAV audio processing.
 *
 * This file defines the AudioData structure used to hold audio samples and
 * metadata, and declares the WaveProcessor class which provides static methods
 * for reading WAV files, performing Fast Fourier Transforms (FFT), Inverse
 * Fast Fourier Transforms (IFFT), and writing audio data to WAV files.
 */

#ifndef WAVEPROCESSOR_H
#define WAVEPROCESSOR_H

#pragma once // Ensures this header file is included only once during compilation

#include "../constants.h" // Assumes constants.h defines the Complex type (e.g., std::complex<double>)
#include <vector>   // For std::vector
#include <stdexcept> // For std::runtime_error
// Include the FFTW library header. The path might need adjustment based on your system.
// The commented-out line is a more standard way if FFTW is in your system's include path.
#include "/opt/homebrew/Cellar/fftw/3.3.10_2/include/fftw3.h"
// #include <fftw3.h>


/**
 * @brief Structure to hold audio data read from a WAV file.
 *
 * Contains the audio samples for the left and right channels, the sample rate,
 * and the number of channels. Samples are typically stored as double-precision
 * floating-point values normalized between -1.0 and 1.0.
 */
struct AudioData {
    std::vector<double> leftChannel;  ///< Vector containing the audio samples for the left channel.
    std::vector<double> rightChannel; ///< Vector containing the audio samples for the right channel.
    int sampleRate;                   ///< The sample rate of the audio in Hz.
    int channels;                     ///< The number of channels (e.g., 1 for mono, 2 for stereo).
};

/**
 * @brief A static class providing utility functions for processing WAV audio files.
 *
 * This class offers methods to read audio data from WAV files, perform FFT
 * to transform time-domain samples to the frequency domain, perform IFFT
 * to transform frequency-domain data back to the time domain, and write
 * audio data to WAV files. It relies on external libraries like dr_wav for
 * file I/O and FFTW for FFT/IFFT computations.
 */
class WaveProcessor
{
public:
    /**
     * @brief Reads a WAV audio file from the specified filename.
     *
     * Uses the dr_wav library to read the file, extract metadata, and load
     * the audio samples into an AudioData structure. Handles both mono and
     * stereo files, converting samples to double-precision and separating
     * channels.
     *
     * @param filename The path to the WAV file to read.
     * @return An AudioData struct containing the loaded audio samples and metadata.
     * @throws std::runtime_error if the file cannot be opened or the format is unsupported.
     */
    static AudioData readWav(const char *filename);

    /**
     * @brief Computes the Fast Fourier Transform (FFT) of a vector of samples.
     *
     * Transforms time-domain audio samples into the frequency domain using
     * the FFTW library. The input is a vector of real-valued samples, and
     * the output is a vector of Complex numbers representing the spectrum.
     *
     * @param samples A vector of double-precision floating-point values (time-domain samples).
     * @return A vector of Complex numbers representing the frequency spectrum.
     */
    static std::vector<Complex> computeFFT(const std::vector<double> &samples);

    /**
     * @brief Computes the Inverse Fast Fourier Transform (IFFT) of a frequency spectrum.
     *
     * Transforms frequency-domain data (spectrum) back into the time domain
     * using the FFTW library. The input is a vector of Complex numbers, and
     * the output is a vector of double-precision floating-point values
     * (time-domain samples). The result is normalized.
     *
     * @param spectrum A vector of Complex numbers representing the frequency spectrum.
     * @return A vector of double-precision floating-point values (reconstructed time-domain samples).
     */
    static std::vector<double> computeIFFT(const std::vector<Complex>& spectrum);

    /**
     * @brief Writes audio samples to a WAV file.
     *
     * Uses the dr_wav library to write the provided audio samples to a WAV file
     * with the specified sample rate and number of channels. The input samples
     * (double-precision, -1.0 to 1.0) are converted to 16-bit integers for the
     * PCM format.
     *
     * @param filename The path where the WAV file will be written.
     * @param samples A vector of double-precision floating-point values (time-domain samples).
     * Assumed to be mono if channels is 1, or a single channel
     * to be duplicated if channels is 2 (stereo output from mono input).
     * @param sampleRate The sample rate for the output WAV file.
     * @param channels The number of channels for the output WAV file (default is 1 for mono).
     * @throws std::runtime_error if the file cannot be opened for writing.
     */
    static void writeWav(const char* filename, const std::vector<double>& samples,
                        int sampleRate, int channels = 1);
};


#endif //WAVEPROCESSOR_H
