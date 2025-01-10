//
// Created by Matthew McQuistion on 12/17/24.
//


#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <complex>
#include <cmath>
#include <vector>

using Complex = std::complex<double>;

inline int PHASE_WINDOW_SIZE = 1;
inline int FREQUENCY_WINDOW_SIZE = 5;

inline int PHASE_NUM_STEPS = 256;
inline int FREQUENCY_NUM_STEPS = 512;
inline int MAX_COMPONENTS = 1000;
inline int ITER_WINDOW_SIZE = 20000;
inline double PHI = 1.618033988749895;
inline double GSS_TOLERANCE = 1e-2;


#endif //CONSTANTS_H
