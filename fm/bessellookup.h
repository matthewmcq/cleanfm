//
// Created by Matthew McQuistion on 3/2/25.
//
#ifndef BESSEL_LOOKUP_H
#define BESSEL_LOOKUP_H

#include <vector>
#include <unordered_map>
#include <cmath>

class BesselLookup {
public:
    BesselLookup(int maxOrder = 10, double maxX = 15.0, double stepSize = 0.1)
        : maxOrder_(maxOrder), maxX_(maxX), stepSize_(stepSize) {
        generateTable();
    }

    // Get Bessel function of first kind, order n, argument x
    double besselJ(int n, double x) {
        // Handle negative orders
        if (n < 0) {
            n = std::abs(n);
            return (n % 2 == 0) ? besselJ(n, x) : -besselJ(n, x);
        }

        // Handle out-of-range values
        if (n > maxOrder_ || x > maxX_) {
            return computeBesselJ(n, x);
        }

        // Look up in table or interpolate
        return lookupOrInterpolate(n, x);
    }

private:
    int maxOrder_;
    double maxX_;
    double stepSize_;

    // 2D table: [order][x_index]
    std::vector<std::vector<double>> table_;

    void generateTable() {
        const int numSteps = static_cast<int>(maxX_ / stepSize_) + 1;

        // Initialize table
        table_.resize(maxOrder_ + 1);
        for (int n = 0; n <= maxOrder_; ++n) {
            table_[n].resize(numSteps);

            for (int i = 0; i < numSteps; ++i) {
                double x = i * stepSize_;
                table_[n][i] = computeBesselJ(n, x);
            }
        }
    }

    // Compute Bessel function accurately without table
    double computeBesselJ(int n, double x) {
        // Special cases
        if (x == 0.0) {
            return (n == 0) ? 1.0 : 0.0;
        }

        if (n == 0) {
            // Series approximation for J₀
            if (std::abs(x) <= 8.0) {
                double x2 = x * x;
                double num = 1.0;
                double den = 1.0;
                double sum = 1.0;
                double term = 1.0;

                for (int k = 1; k <= 15; ++k) {
                    num = -num * x2;
                    den = den * 4.0 * k * k;
                    term = num / den;
                    sum += term;

                    if (std::abs(term) < 1e-15 * std::abs(sum)) break;
                }

                return sum;
            }
            else {
                // Asymptotic form for large x
                double z = 8.0 / x;
                double z2 = z * z;
                double p0 = 1.0;
                double p1 = -0.0703125 * z2;
                double p2 = 0.112152099609375 * z2 * z2;
                double p3 = -0.5725014209747314 * z2 * z2 * z2;
                double q0 = -0.125 * z;
                double q1 = 0.0732421875 * z * z2;
                double q2 = -0.2271080017089844 * z * z2 * z2;
                double q3 = 1.727727502584457 * z * z2 * z2 * z2;

                double p = p0 + p1 + p2 + p3;
                double q = q0 + q1 + q2 + q3;

                double theta = x - M_PI / 4.0;
                return sqrt(2.0 / (M_PI * x)) * (p * cos(theta) - q * sin(theta));
            }
        }

        if (n == 1) {
            // Series approximation for J₁
            if (std::abs(x) <= 8.0) {
                double x2 = x * x;
                double num = x / 2.0;
                double den = 1.0;
                double sum = num;
                double term = num;

                for (int k = 1; k <= 15; ++k) {
                    num = -num * x2;
                    den = den * 4.0 * k * (k + 1);
                    term = num / den;
                    sum += term;

                    if (std::abs(term) < 1e-15 * std::abs(sum)) break;
                }

                return sum;
            }
            else {
                // Asymptotic form for large x
                double z = 8.0 / x;
                double z2 = z * z;
                double p0 = 1.0;
                double p1 = 0.1171875 * z2;
                double p2 = -0.144195556640625 * z2 * z2;
                double p3 = 0.6765925884246826 * z2 * z2 * z2;
                double q0 = 0.375 * z;
                double q1 = -0.1025390625 * z * z2;
                double q2 = 0.2775764465332031 * z * z2 * z2;
                double q3 = -1.993531733751297 * z * z2 * z2 * z2;

                double p = p0 + p1 + p2 + p3;
                double q = q0 + q1 + q2 + q3;

                double theta = x - 3.0 * M_PI / 4.0;
                return sqrt(2.0 / (M_PI * x)) * (p * cos(theta) - q * sin(theta));
            }
        }

        // For n > 1, use forward recurrence relation
        double jnm1 = computeBesselJ(0, x);  // J_{n-1}(x)
        double jn = computeBesselJ(1, x);    // J_n(x)
        double jnp1;                      // J_{n+1}(x)

        for (int i = 1; i < n; ++i) {
            jnp1 = (2.0 * i / x) * jn - jnm1;
            jnm1 = jn;
            jn = jnp1;
        }

        return jn;
    }

    // Linear interpolation between table values
    double lookupOrInterpolate(int n, double x) {
        if (x == 0.0) {
            return (n == 0) ? 1.0 : 0.0;
        }

        // Calculate indices
        double xIndex = x / stepSize_;
        int lowerIndex = static_cast<int>(xIndex);
        int upperIndex = lowerIndex + 1;

        // Ensure indices are within bounds
        if (upperIndex >= static_cast<int>(table_[n].size())) {
            return computeBesselJ(n, x);
        }

        // Get values at lower and upper indices
        double lowerValue = table_[n][lowerIndex];
        double upperValue = table_[n][upperIndex];

        // Interpolate
        double fraction = xIndex - lowerIndex;
        return lowerValue + fraction * (upperValue - lowerValue);
    }
};

#endif // BESSEL_LOOKUP_H