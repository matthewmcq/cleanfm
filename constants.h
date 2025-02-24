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
inline int MAX_COMPONENTS = 10000;
inline int ITER_WINDOW_SIZE = 10000;
inline double PHI = 1.618033988749895;
inline double GSS_TOLERANCE = 1e-11;
inline double CONVERGENCE = 1e-6;


#endif //CONSTANTS_H
// Component 6463: true bin=3073.52: old bin=6403, phase=-2.40836
// % L2 norm retained: 0.0724548
