#pragma once

#include "ceres/ceres.h"

namespace mpl {
struct PhotometricBAConfig {
    PhotometricBAConfig();

    // Ceres-Problem options
    ceres::Problem::Options problemOptions;

    // Ceres-Solver options.
    ceres::Solver::Options solverOptions;
};
}  // namespace mpl