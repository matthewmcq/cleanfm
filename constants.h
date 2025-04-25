/**
 * @file constants.h
 * @author Matthew McQuistion
 * @date 12/17/24
 * @brief Defines various inline constants used throughout the project, particularly for audio processing and algorithms.
 *
 * This header file centralizes the definition of numerical constants and
 * parameters used in different parts of the application, including those
 * related to audio processing window sizes, iteration steps, limits,
 * and algorithm tolerances.
 */

#ifndef CONSTANTS_H // Include guard to prevent multiple inclusions
#define CONSTANTS_H

#pragma once // Alternative include guard, preferred by some compilers

#include <complex> // Required for std::complex
#include <cmath>   // Required for mathematical functions like std::sqrt (though not directly used here, often needed with complex numbers)
#include <vector>  // Required for std::vector

/**
 * @brief Alias for std::complex<double> for convenience.
 *
 * Defines 'Complex' as a synonym for the standard complex number type
 * using double-precision floating-point values.
 */
using Complex = std::complex<double>;

/**
 * @brief Defines the size of the window used for phase estimation in processing.
 *
 * This constant determines the number of frequency bins considered around a peak
 * for accurate phase calculation.
 */
inline int PHASE_WINDOW_SIZE = 3; // 9

/**
 * @brief Defines the size of the window used for frequency estimation in processing.
 *
 * This constant determines the number of frequency bins considered around a peak
 * for accurate frequency calculation.
 */
inline int FREQUENCY_WINDOW_SIZE = 4; // 10


/**
 * @brief The maximum number of components (e.g., peaks) expected or allowed.
 *
 * This constant sets an upper limit on the number of individual frequency
 * components that the algorithm will attempt to identify or process.
 */
inline int MAX_COMPONENTS = 40000;

/**
 * @brief The golden ratio constant.
 *
 * Used in algorithms that employ golden section search or related optimization techniques.
 */
inline double PHI = 1.618033988749895;

/**
 * @brief Tolerance level for Golden Section Search (GSS).
 *
 * This constant defines the desired accuracy for the golden section search
 * algorithm, determining when the search should terminate.
 */
inline double GSS_TOLERANCE = 1e-8;//1e-11;

/**
 * @brief Convergence threshold for iterative algorithms.
 *
 * This constant defines the level of change between iterations below which
 * an algorithm is considered to have converged.
 */
inline double CONVERGENCE = 1e-6;

// Threads

/**
 * @brief The size of batches for parallel processing tasks.
 *
 * This constant is used to divide larger processing tasks into smaller batches
 * that can be distributed among threads in a thread pool.
 */
inline int BATCH_SIZE = 256;

/**
 * @brief Flag to enable or disable verbose output for debugging parallel processing.
 *
 * When set to true, enables detailed output messages related to the parallel
 * execution of algorithms (e.g., DKD).
 */
inline bool VERBOSE_OUTPUT = true; // for debugging DKD parallel


#endif //CONSTANTS_H
