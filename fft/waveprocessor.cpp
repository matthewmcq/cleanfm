//
// Created by Matthew McQuistion on 12/17/24.
//

#include "waveprocessor.h"
#include "../libs/dr_wav.h"

#define DR_WAV_IMPLEMENTATION


AudioData WaveProcessor::readWav(const char *filename) {
    drwav wav;
    if (!drwav_init_file(&wav, filename, NULL)) {
        throw std::runtime_error("Failed to open WAV file");
    }

    AudioData audio;
    audio.sampleRate = wav.sampleRate;
    audio.channels = wav.channels;

    // Read samples
    std::vector<int16_t> rawSamples(wav.totalPCMFrameCount * wav.channels);
    drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, rawSamples.data());

    // Separate channels
    const size_t frames = wav.totalPCMFrameCount;
    audio.leftChannel.resize(frames);
    audio.rightChannel.resize(frames);

    if (wav.channels == 2) {
        // Stereo
        for (size_t i = 0; i < frames; i++) {
            audio.leftChannel[i] = rawSamples[i * 2] / 32768.0;
            audio.rightChannel[i] = rawSamples[i * 2 + 1] / 32768.0;
        }
    } else if (wav.channels == 1) {
        // Mono - copy to both channels
        for (size_t i = 0; i < frames; i++) {
            audio.leftChannel[i] = audio.rightChannel[i] = rawSamples[i] / 32768.0;
        }
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }

    drwav_uninit(&wav);
    return audio;
}

std::vector<Complex> WaveProcessor::computeFFT(const std::vector<double> &samples) {
    size_t N = samples.size();

    // FFTW plan and arrays
    fftw_complex *in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);

    // Copy samples to complex array
    for (size_t i = 0; i < N; i++) {
        in[i][0] = samples[i]; // Real part
        in[i][1] = 0.0; // Imaginary part
    }

    // Create and execute FFT plan
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Copy results to vector
    std::vector<Complex> spectrum(N);
    for (size_t i = 0; i < N; i++) {
        spectrum[i] = Complex(out[i][0], out[i][1]);
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return spectrum;
}

std::vector<double> WaveProcessor::computeIFFT(const std::vector<Complex> &spectrum) {
    size_t N = spectrum.size();

    // FFTW plan and arrays
    fftw_complex *in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);

    // Copy spectrum to FFTW array
    for (size_t i = 0; i < N; i++) {
        in[i][0] = spectrum[i].real();
        in[i][1] = spectrum[i].imag();
    }

    // Create and execute inverse FFT plan
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Convert to real and normalize
    std::vector<double> result(N);
    for (size_t i = 0; i < N; i++) {
        result[i] = out[i][0] / N; // Normalize by N for IFFT
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
}

void WaveProcessor::writeWav(const char *filename, const std::vector<double> &samples,
                           int sampleRate, int channels) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = channels;
    format.sampleRate = sampleRate;
    format.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, filename, &format, NULL)) {
        throw std::runtime_error("Failed to open WAV file for writing");
    }

    // Convert doubles to int16, accounting for channels
    const size_t frames = samples.size();  // Number of frames per channel
    std::vector<int16_t> pcm(frames * channels);  // Total number of samples

    if (channels == 1) {
        // Mono: straight conversion
        for (size_t i = 0; i < frames; i++) {
            pcm[i] = static_cast<int16_t>(samples[i] * 32767.0);
        }
    } else if (channels == 2) {
        // Stereo: duplicate mono channel to both left and right
        for (size_t i = 0; i < frames; i++) {
            pcm[i * 2] = static_cast<int16_t>(samples[i] * 32767.0);     // Left
            pcm[i * 2 + 1] = static_cast<int16_t>(samples[i] * 32767.0); // Right
        }
    }

    // Write frames (note: samples.size() is number of frames, not total samples)
    drwav_write_pcm_frames(&wav, frames, pcm.data());
    drwav_uninit(&wav);
}

