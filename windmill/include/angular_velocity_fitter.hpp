#ifndef ANGULAR_VELOCITY_FITTER_HPP
#define ANGULAR_VELOCITY_FITTER_HPP
#pragma once

#include <vector>
#include <array>
#include <utility> // For std::pair

namespace AngleFitter {

/**
 * @brief Represents the convexity of the angle-time data.
 * Used internally by the fitting process, especially with sparse data.
 */
enum class Convexity {
    UNKNOWN,
    CONCAVE,
    CONVEX
};

/**
 * @brief Fits the angular motion model y(t) = -a*cos(w*(t + t0)) + b*t + c to the provided data.
 *
 * Uses RANSAC and Ceres Solver for robust non-linear least squares estimation.
 *
 * @param fitData A vector of pairs, where each pair is (time_in_seconds, absolute_angle_in_radians).
 *                Time should be relative to a consistent start point.
 * @param initialParams An initial guess for the parameters [a, w, t0, b, c].
 *                      If unsure, a default like {0.8, 1.9, 0.0, 0.0, 0.0} can be tried,
 *                      based loosely on typical rune behavior. The original code used {0.470, 1.942, 0, 1.178, 0}.
 * @param minDataSize The minimum number of data points required to attempt fitting.
 * @param outParams Reference to an array where the fitted parameters [a, w, t0, b, c] will be stored.
 * @return true if fitting was successful (e.g., enough data, solver converged), false otherwise.
 *
 * @note The parameters have the following meaning in the model:
 *       a: Amplitude of the sinusoidal component (params[0]).
 *       w: Angular frequency of the sinusoidal component (rad/s) (params[1]).
 *       t0: Phase shift of the sinusoidal component (s) (params[2]).
 *       b: Linear angular velocity component (rad/s) (params[3]).
 *       c: Constant angle offset (rad) (params[4]).
 * @note Requires linking against the Ceres Solver library.
 */
bool fitAngularVelocityParameters(
    const std::vector<std::pair<double, double>>& fitData,
    const std::array<double, 5>& initialParams,
    size_t minDataSize,
    std::array<double, 5>& outParams
);

/**
 * @brief Calculates the predicted angle at a given time using the fitted parameters.
 *        y(t) = -a*cos(w*(t + t0)) + b*t + c
 *
 * @param time Time in seconds (relative to the same start point as the fitting data).
 * @param params The fitted parameters [a, w, t0, b, c].
 * @return Predicted absolute angle in radians.
 */
double getAngleFromParameters(double time, const std::array<double, 5>& params);

} // namespace AngleFitter
#endif // TRADITIONAL_DETECTION_HPP