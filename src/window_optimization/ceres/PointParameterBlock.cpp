
#include "PointParameterBlock.h"
#include "config.h"

namespace mpl {
// Parameterization
PointParameterization::PointParameterization() {
}

PointParameterization::~PointParameterization() {
}

bool PointParameterization::Plus(const double* x, const double* delta, double* x_plus_delta) const {
    const auto& config = Config::getInstance();

    // variable scaling
    // "Numerical Optimization" Nocedal et al. 2006, page 95
    x_plus_delta[0] = x[0] + (delta[0] * config.OPTIMIZATION_IDEPTH_SCALE);

    return true;
}

bool PointParameterization::ComputeJacobian(const double* x, double* jacobian) const {
    jacobian[0] = 1.0;
    return true;
}

int PointParameterization::GlobalSize() const {
    return 1;
}

int PointParameterization::LocalSize() const {
    return 1;
}

// Parameter Block
const std::unique_ptr<PointParameterization> PointParameterBlock::pointParameterization =
    std::make_unique<PointParameterization>();

PointParameterBlock::PointParameterBlock() : BAParameterBlock<1>() {
}

PointParameterBlock::PointParameterBlock(double iDepth) : BAParameterBlock<1>() {
    this->setIDepth(iDepth);
    this->backup();
    this->setLocalParameterization(PointParameterBlock::pointParameterization.get());
}

PointParameterBlock::~PointParameterBlock() {
}

void PointParameterBlock::setIDepth(double iDepth) {
    this->parameters_[0] = iDepth;
}

double PointParameterBlock::getIDepth() const {
    return this->parameters_[0];
}

double PointParameterBlock::getIDepthBackup() const {
    return this->parameters_backup_[0];
}

void PointParameterBlock::step(double& delta) const {
    // linear in the euclidean space
    // currentState = backupState + delta
    delta = this->parameters_[0] - this->parameters_backup_[0];
}

void PointParameterBlock::scaledStep(double& delta) const {
    const auto& config = Config::getInstance();

    // linear in the euclidean space
    // currentState = backupState + delta
    delta = this->parameters_[0] - this->parameters_backup_[0];

    // scale
    delta /= config.OPTIMIZATION_IDEPTH_SCALE;
}

}  // namespace mpl