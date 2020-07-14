
#include "bundle_adjustment.h"

namespace mpl {
BundleAdjustment::BundleAdjustment(
    const ceres::Problem::Options& problemOptions) {
    // ceres problem creation
    this->problem_ = std::make_unique<ceres::Problem>(problemOptions);
}

BundleAdjustment::~BundleAdjustment() {
}

void BundleAdjustment::solve(const ceres::Solver::Options& solverOptions) {
    ceres::Solve(solverOptions, this->problem_.get(), &this->summary_);
}

const ceres::Solver::Summary& BundleAdjustment::summary() const {
    return this->summary_;
}

void BundleAdjustment::removeResidualBlock(
    ceres::ResidualBlockId residualBlockId) {
    this->problem_->RemoveResidualBlock(residualBlockId);
}

int BundleAdjustment::numParameterBlocks() const {
    return this->problem_->NumParameterBlocks();
}

int BundleAdjustment::numParameters() const {
    return this->problem_->NumParameters();
}

int BundleAdjustment::numResidualBlocks() const {
    return this->problem_->NumResidualBlocks();
}

int BundleAdjustment::numResiduals() const {
    return this->problem_->NumResiduals();
}
}  // namespace mpl
