//
// Created by Matthew McQuistion on 12/17/24.
//

#include "dirichletkernel.h"
#include <cmath>

DirichletKernel::DirichletKernel(const double frequency, const double amplitude, const double phase, const int N) {
}

Complex DirichletKernel::getValueAtBin(const double difference, const double phase, const size_t N) {
    constexpr std::complex<double> i(0, 1);
    const double numerator = std::sin(M_PI * difference);
    const double denominator = std::sin(M_PI * difference / static_cast<double>(N));
    if (std::abs(difference) < 1e-16) {
        return std::exp(-i * phase);
    }
    return numerator / denominator * std::exp(-i * (M_PI * difference + phase));
}

double DirichletKernel::getAmplitudeAtBin(const double frequency, const double bin_magnitude, const size_t N,
                                          const int nearest_bin) {
    const double delta = frequency - static_cast<double>(nearest_bin);

    const double sinc_correction =
            std::abs(delta) < 1e-16 ? 1 : std::sin(M_PI * delta) / std::sin(M_PI * delta / static_cast<double>(N));

    return bin_magnitude / sinc_correction;
}

std::vector<Complex> DirichletKernel::
generateKernel(const std::vector<int> &bins, const double frequency, const double phase, const size_t N,
               const double bin_magnitude) {
    std::vector<Complex> result(bins.size());
    // const double amplitude = getAmplitudeAtBin(frequency, bin_magnitude, N);

    for (int i = 0; i < bins.size(); i++) {
        const int bin = bins[i];
        const double difference = frequency - static_cast<double>(bin);
        result[i] = getValueAtBin(difference, phase, N);
    }

    return result;
}
