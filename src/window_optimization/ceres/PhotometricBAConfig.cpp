
#include "PhotometricBAConfig.h"

namespace mpl {
PhotometricBAConfig::PhotometricBAConfig() {
    // problem options

    // Do not enable ceres to take ownership of any object
    this->problemOptions.cost_function_ownership =
        ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    this->problemOptions.local_parameterization_ownership =
        ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    this->problemOptions.loss_function_ownership =
        ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

    // If true, trades memory for faster RemoveResidualBlock() and
    // RemoveParameterBlock() operations
    this->problemOptions.enable_fast_removal = false;

    // By default, Ceres performs a variety of safety checks when constructing
    // the problem. There is a small but measurable performance penalty to
    // these checks, typically around 5% of construction time. If you are sure
    // your problem construction is correct, and 5% of the problem construction
    // time is truly an overhead you want to avoid, then you can set
    // disable_all_safety_checks to true.
    //
    // WARNING: Do not set this to true, unless you are absolutely sure of what
    // you are doing.
    this->problemOptions.disable_all_safety_checks = true;

    // solver options
    this->solverOptions.logging_type = ceres::PER_MINIMIZER_ITERATION;
    this->solverOptions.minimizer_progress_to_stdout = true;

    this->solverOptions.linear_solver_type = ceres::DENSE_SCHUR;
    this->solverOptions.minimizer_type = ceres::TRUST_REGION;
    this->solverOptions.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    this->solverOptions.use_nonmonotonic_steps = true;
    this->solverOptions.max_consecutive_nonmonotonic_steps = 2;

    this->solverOptions.jacobi_scaling = true;

    // required for ceres::IterationCallback
    this->solverOptions.update_state_every_iteration = true;

    this->solverOptions.max_num_iterations = 50;

    // ignore ceres termination criteria
    this->solverOptions.function_tolerance = 0.0;
    this->solverOptions.gradient_tolerance = 0.0;
    this->solverOptions.parameter_tolerance = 0.0;

    // number of parallel threads
    this->solverOptions.num_threads = 8;
}
}  // namespace mpl