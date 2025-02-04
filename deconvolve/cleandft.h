//
// Created by Matthew McQuistion on 12/17/24.
//

#ifndef CLEANDFT_H
#define CLEANDFT_H
#include "../constants.h"
#include <cmath>
#include <vector>
#include "../fft/waveprocessor.h"
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
    static std::vector<double> decompressComponents(const std::vector<Component> &components,
                                                    size_t N, Complex DC);

    static std::vector<double> resample(const std::vector<Component> &components, size_t N,
                                        size_t original_sr, size_t target_sr, Complex DC);

private:
    template<typename F>
    static double goldenSectionSearch(const F& objective, double a, double b) {
        double c = b - (b - a) / PHI;
        double d = a + (b - a) / PHI;

        double fc = objective(c);
        double fd = objective(d);

        while (std::abs(b - a) > GSS_TOLERANCE) {
            if (fc > fd) {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) / PHI;
                fc = objective(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) / PHI;
                fd = objective(d);
            }
        }

        return (a + b) / 2.0;
    }
};


#endif //CLEANDFT_H
