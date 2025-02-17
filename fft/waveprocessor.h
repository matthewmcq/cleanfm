//
// Created by Matthew McQuistion on 12/17/24.
//

#ifndef WAVEPROCESSOR_H
#define WAVEPROCESSOR_H

#pragma once

#include "../constants.h"
#include <stdexcept>
#include "/opt/homebrew/Cellar/fftw/3.3.10_2/include/fftw3.h"
// #include <fftw3.h>


struct AudioData {
    std::vector<double> leftChannel;
    std::vector<double> rightChannel;
    int sampleRate;
    int channels;
};

class WaveProcessor
{
public:
    static AudioData readWav(const char *filename);
    static std::vector<Complex> computeFFT(const std::vector<double> &samples);
    static std::vector<double> computeIFFT(const std::vector<Complex>& spectrum);
    static void writeWav(const char* filename, const std::vector<double>& samples,
                        int sampleRate, int channels = 1);
};


#endif //WAVEPROCESSOR_H
