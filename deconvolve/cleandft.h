//
// Created by Matthew McQuistion on 12/17/24.
//

#ifndef CLEANDFT_H
#define CLEANDFT_H
#include "../constants.h"
#include <cmath>
#include <vector>
#include "dirichletkernel.h"

class CleanDFT {
public:

    struct Component {
        double true_frequency;
        double true_phase;
        double amplitude;
    };

    explicit CleanDFT();


    static double computeCorrelation(const std::vector<Complex> &data_slice, const std::vector<Complex> &kernel,
                                     bool isPhase);

    static double findOptimalPhase(const std::vector<Complex> &fft, int center_bin, double frequency);

    static double findOptimalFrequency(const std::vector<Complex> &fft, int center_bin, double test_frequency);

    static std::vector<Component> deconvolveDirichletKernel(const std::vector<Complex> &fft, size_t sample_rate);

};


#endif //CLEANDFT_H
