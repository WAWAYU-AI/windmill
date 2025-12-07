#include <angular_velocity_fitter.hpp>

#include <algorithm>
#include <ceres/ceres.h>
#include <chrono>
#include <cmath>
#include <iostream> // For potential debugging output
#include <numeric>
#include <random>

namespace AngleFitter {

// Forward declarations of internal helper functions
Convexity getConvexity(const std::vector<std::pair<double, double>> &data);
std::array<double, 5>
ransacFitting(const std::vector<std::pair<double, double>> &data,
              const std::array<double, 5> &initialParams, Convexity convexity,
              size_t minDataSize); // Added initialParams and minDataSize
std::array<double, 5>
leastSquareEstimate(const std::vector<std::pair<double, double>> &points,
                    const std::array<double, 5> &params, Convexity convexity);

// --- Ceres Cost Functors (Copied and adapted) ---

// Model: y_pred = -p[0]*cos(p[1]*(t + p[2])) + p[3]*t + p[4]
// Residual: y_pred - y_observed
class CostFunctorData : public ceres::SizedCostFunction<1, 5> {
public:
  CostFunctorData(double t_, double y_) : t(t_), y(y_) {}
  virtual ~CostFunctorData() {}
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {
    double a = parameters[0][0];
    double w = parameters[0][1];
    double t0 = parameters[0][2];
    double b = parameters[0][3];
    double c = parameters[0][4];
    double wt_t0 = w * (t + t0);
    double cs = cos(wt_t0);
    double sn = sin(wt_t0);

    residuals[0] = (-a * cs + b * t + c) - y; // Model prediction - observation

    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        jacobians[0][0] = -cs;               // dResidual/da
        jacobians[0][1] = a * (t + t0) * sn; // dResidual/dw
        jacobians[0][2] = a * w * sn;        // dResidual/dt0
        jacobians[0][3] = t;                 // dResidual/db
        jacobians[0][4] = 1.0;               // dResidual/dc
      }
    }
    return true;
  }

private:
  double t; // Time
  double y; // Observed Angle
};

// Regularization: penalize deviation from initial guess
class CostFunctorPrior : public ceres::SizedCostFunction<1, 5> {
public:
  CostFunctorPrior(double initialValue_, int paramIndex_)
      : initialValue(initialValue_), paramIndex(paramIndex_) {}
  virtual ~CostFunctorPrior() {}
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {
    residuals[0] = parameters[0][paramIndex] - initialValue;

    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        for (int i = 0; i < 5; ++i) {
          jacobians[0][i] = (i == paramIndex) ? 1.0 : 0.0;
        }
      }
    }
    return true;
  }

private:
  double initialValue;
  int paramIndex; // Index of the parameter (0=a, 1=w, 2=t0, 3=b, 4=c)
};

// --- Helper Functions (Copied and adapted) ---

double getAngleFromParameters(double time,
                              const std::array<double, 5> &params) {
  // Model: y(t) = -a*cos(w*(t + t0)) + b*t + c
  return -params[0] * std::cos(params[1] * (time + params[2])) +
         params[3] * time + params[4];
}

Convexity getConvexity(const std::vector<std::pair<double, double>> &data) {
  if (data.size() < 3) {
    return Convexity::UNKNOWN;
  }
  auto first = data.begin();
  auto last = data.end() - 1;
  if (std::abs(last->first - first->first) < 1e-6) {
    return Convexity::UNKNOWN;
  }
  double slope = (last->second - first->second) / (last->first - first->first);
  double offset = (first->second * last->first - last->second * first->first) /
                  (last->first - first->first);
  int concave = 0, convex = 0;
  for (const auto &i : data) {
    if (slope * i.first + offset > i.second) {
      concave++;
    } else {
      convex++;
    }
  }
  const int standard = static_cast<int>(data.size() * 0.75);
  return concave > standard  ? Convexity::CONCAVE
         : convex > standard ? Convexity::CONVEX
                             : Convexity::UNKNOWN;
}

std::array<double, 5>
ransacFitting(const std::vector<std::pair<double, double>> &data,
              const std::array<double, 5> &initialParams, Convexity convexity,
              size_t minDataSize) {
  if (data.size() < minDataSize) {
    return initialParams;
  }

  std::vector<std::pair<double, double>> inliers = data;
  std::vector<std::pair<double, double>> outliers;
  std::array<double, 5> bestParams = initialParams;

  const int iterTimes = data.size() < 400 ? 200 : 20;
  const size_t sampleSize = std::min((size_t)200, data.size() / 2);
  const double errorThresholdFactor = 1.5;
  const bool refineUsingInliers = true;

  std::default_random_engine rng(
      std::chrono::system_clock::now().time_since_epoch().count());
  int maxInliers = -1;

  for (int i = 0; i < iterTimes; ++i) {
    std::vector<std::pair<double, double>> currentSample;
    std::vector<std::pair<double, double>> currentInliers;
    std::array<double, 5> currentParams;

    if (inliers.size() <= sampleSize) {
      currentSample = inliers;
    } else {
      std::shuffle(inliers.begin(), inliers.end(), rng);
      currentSample.assign(inliers.begin(), inliers.begin() + sampleSize);
    }

    if (currentSample.size() < 5)
      continue;

    currentParams = leastSquareEstimate(currentSample, bestParams, convexity);

    std::vector<double> errors;
    for (const auto &point : data) {
      errors.push_back(std::abs(
          point.second - getAngleFromParameters(point.first, currentParams)));
    }
    std::vector<double> sortedErrors = errors;
    std::sort(sortedErrors.begin(), sortedErrors.end());
    double medianError = sortedErrors[sortedErrors.size() / 2];
    double threshold = std::max(
        0.01, medianError *
                  errorThresholdFactor); // Avoid zero threshold, add minimum

    currentInliers.clear();
    for (size_t k = 0; k < data.size(); ++k) {
      if (errors[k] < threshold) {
        currentInliers.push_back(data[k]);
      }
    }

    if ((int)currentInliers.size() > maxInliers) {
      maxInliers = currentInliers.size();
      bestParams = currentParams;
    }
  }

  if (refineUsingInliers && maxInliers > (int)minDataSize) {
    std::vector<std::pair<double, double>> finalInliers;
    std::vector<double> errors;
    for (const auto &point : data) {
      errors.push_back(std::abs(
          point.second - getAngleFromParameters(point.first, bestParams)));
    }
    std::vector<double> sortedErrors = errors;
    std::sort(sortedErrors.begin(), sortedErrors.end());
    double medianError = sortedErrors[sortedErrors.size() / 2];
    double threshold = std::max(0.01, medianError * errorThresholdFactor);

    for (size_t k = 0; k < data.size(); ++k) {
      if (errors[k] < threshold) {
        finalInliers.push_back(data[k]);
      }
    }
    if (finalInliers.size() >= minDataSize) {
      bestParams = leastSquareEstimate(finalInliers, bestParams, convexity);
    }
  }

  return bestParams;
}

std::array<double, 5>
leastSquareEstimate(const std::vector<std::pair<double, double>> &points,
                    const std::array<double, 5>
                        &initialParams, // Changed from params to initialParams
                    Convexity convexity) {
  if (points.empty()) {
    return initialParams;
  }

  std::array<double, 5> params = initialParams; // Work with a mutable copy
  ceres::Problem problem;

  for (size_t i = 0; i < points.size(); i++) {
    ceres::CostFunction *costFunction =
        new CostFunctorData(points[i].first, points[i].second);
    ceres::LossFunction *lossFunction =
        new ceres::CauchyLoss(0.1); // Or HuberLoss(1.0), SoftLOneLoss(0.1)
    problem.AddResidualBlock(costFunction, lossFunction, params.data());
  }

  std::array<double, 3> omega = {
      1.0, 0.1, 0.1}; // Weights for a, w, b priors respectively
  if (points.size() > 100) {
    omega = {10.0, 1.0,
             1.0}; // Increase regularization with more data? Or decrease?
                   // Original code logic was different. Let's try decreasing.
  }
  if (points.size() > 400) {
    omega = {1.0, 0.1, 0.1};
  }

  ceres::CostFunction *costPriorA = new CostFunctorPrior(initialParams[0], 0);
  ceres::LossFunction *lossPriorA = new ceres::ScaledLoss(
      nullptr, omega[0],
      ceres::DO_NOT_TAKE_OWNERSHIP); // No robust loss for prior? Or Huber?
  problem.AddResidualBlock(costPriorA, lossPriorA, params.data());

  ceres::CostFunction *costPriorW = new CostFunctorPrior(initialParams[1], 1);
  ceres::LossFunction *lossPriorW =
      new ceres::ScaledLoss(nullptr, omega[1], ceres::DO_NOT_TAKE_OWNERSHIP);
  problem.AddResidualBlock(costPriorW, lossPriorW, params.data());

  ceres::CostFunction *costPriorB = new CostFunctorPrior(initialParams[3], 3);
  ceres::LossFunction *lossPriorB =
      new ceres::ScaledLoss(nullptr, omega[2], ceres::DO_NOT_TAKE_OWNERSHIP);
  problem.AddResidualBlock(costPriorB, lossPriorB, params.data());

  // Apply bounds based on convexity if data is sparse
  if (points.size() < 100) {
    // Original code set bounds on t0 (index 2). Let's replicate that logic.
    // Note: The physical meaning of these bounds on t0 isn't immediately
    // obvious without more context.
    if (convexity == Convexity::CONCAVE) {
      problem.SetParameterUpperBound(params.data(), 2, -2.8);
      problem.SetParameterLowerBound(params.data(), 2, -4.0);
    } else if (convexity == Convexity::CONVEX) {
      problem.SetParameterUpperBound(params.data(), 2, -1.1);
      problem.SetParameterLowerBound(params.data(), 2, -2.3);
    }
  }
  // Consider adding bounds for w (param 1) to be positive, e.g., > 0.1
  problem.SetParameterLowerBound(params.data(), 1, 0.1);
  // Consider adding bounds for a (param 0) to be non-negative? Or depends on
  // definition. problem.SetParameterLowerBound(params.data(), 0, 0.0);

  // Configure and run the solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 50;
  options.minimizer_progress_to_stdout = false; // Set to true for debugging
  options.check_gradients = false; // Enable for debugging gradients if needed
  options.gradient_check_relative_precision = 1e-4;
  // options.function_tolerance = 1e-6;
  // options.parameter_tolerance = 1e-8;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return params;
}

// --- Public Interface Function ---

bool fitAngularVelocityParameters(
    const std::vector<std::pair<double, double>> &fitData,
    const std::array<double, 5> &initialParams, size_t minDataSize,
    std::array<double, 5> &outParams) {
  if (fitData.size() < minDataSize) {
    outParams = initialParams;
    return false;
  }

  Convexity convexity = getConvexity(fitData);

  outParams = ransacFitting(fitData, initialParams, convexity, minDataSize);

  if (outParams[1] <= 0) { // Check if omega is non-positive
  }

  return true;
}

} // namespace AngleFitter
