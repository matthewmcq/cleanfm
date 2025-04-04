//
// Created by Matthew McQuistion on 12/17/24.
//


#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <complex>
#include <cmath>
#include <vector>

using Complex = std::complex<double>;
enum class SpectralDivisionMethod {
    UNIFORM,              // Original uniform division
    EQUAL_ENERGY,         // Equal energy content per band
    PERCEPTUAL_WEIGHTING, // Weighted energy with high-frequency boost
    MAGNITUDE_THRESHOLD,   // Division based on magnitude thresholds
    HYBRID
};


inline int PHASE_WINDOW_SIZE = 3; // 9
inline int FREQUENCY_WINDOW_SIZE = 4; // 10

inline int PHASE_NUM_STEPS = 256;
inline int FREQUENCY_NUM_STEPS = 512;
inline int MAX_COMPONENTS = 40000;

// inline int ITER_WINDOW_SIZE = 10000;
inline double PHI = 1.618033988749895;
inline double GSS_TOLERANCE = 1e-8;//1e-11;
inline double CONVERGENCE = 1e-6;
inline int MAX_FM_COMPONENTS = 1;
inline int MAX_FM_ITERATIONS = 1;
inline bool RAISED_BESSEL = false;
inline double RAISED_BESSEL_CONSTANT = 0.402759;

// Threads
inline int BATCH_SIZE = 1024;

// FM
inline double DEFAULT_HARMONIC_MOD_INDEX = 0.000001;
inline double DEFAULT_LFO_INDEX = 1.0;
inline double DEFAULT_LFO_FREQ = 1.0;

inline double MIN_LFO_FREQ = 0.01;
inline double MAX_LFO_FREQ = 2.0;
inline double MIN_LFO_INDEX = 0.01;
inline double MAX_LFO_INDEX = 1.4347;
inline bool VERBOSE_OUTPUT = true; // for debugging DKD parallel

inline SpectralDivisionMethod SPECTRAL_DIVISION_METHOD = SpectralDivisionMethod::UNIFORM;



#endif //CONSTANTS_H

