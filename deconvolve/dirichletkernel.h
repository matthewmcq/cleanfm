//
// Created by Matthew McQuistion on 12/17/24.
//
#pragma once
#include "../constants.h"


#ifndef DIRICHLETKERNEL_H
#define DIRICHLETKERNEL_H

class DirichletKernel {
public:

    DirichletKernel(double frequency, double amplitude, double phase, int N);

    [[nodiscard]] static Complex getValueAtBin(double difference, double phase, size_t N) ;

    [[nodiscard]] static double getAmplitudeAtBin(double frequency, double bin_magnitude, size_t N, int nearest_bin) ;

    [[nodiscard]] static std::vector<Complex> generateKernel(const std::vector<int> &bins, double frequency, double phase,
                                                             size_t N, double bin_magnitude) ;
};


#endif //DIRICHLETKERNEL_H
