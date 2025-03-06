//
// Created by Matthew McQuistion on 12/17/24.
//


#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <complex>
#include <cmath>
#include <vector>

using Complex = std::complex<double>;

inline int PHASE_WINDOW_SIZE = 9;
inline int FREQUENCY_WINDOW_SIZE = 10;

inline int PHASE_NUM_STEPS = 256;
inline int FREQUENCY_NUM_STEPS = 512;
inline int MAX_COMPONENTS = 30000;
inline int ITER_WINDOW_SIZE = 30000;
inline double PHI = 1.618033988749895;
inline double GSS_TOLERANCE = 1e-6;//1e-11;
inline double CONVERGENCE = 1e-6;
inline int MAX_FM_COMPONENTS = 1;
inline int MAX_FM_ITERATIONS = 1;
inline bool RAISED_BESSEL = false;
inline double RAISED_BESSEL_CONSTANT = 0.402759;


#endif //CONSTANTS_H

