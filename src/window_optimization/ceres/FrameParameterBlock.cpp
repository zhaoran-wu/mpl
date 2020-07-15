#include "FrameParameterBlock.h"
#include "config.h"

namespace mpl {
// Parameterization
FrameParameterization::FrameParameterization() {
}

FrameParameterization::~FrameParameterization() {
}

bool FrameParameterization::Plus(const double* x, const double* delta, double* x_plus_delta) const {
    const auto& config = Config::getInstance();

    // x: 9 number of parameters
    // delta: 8 number of parameters
    // x_plus_delta: 9 number of parameters

    // map buffers
    Eigen::Map<Sophus::SE3d const> const T(x);
    Eigen::Map<Sophus::Vector6d const> const d(delta);
    Eigen::Map<Sophus::SE3d> T_plus_delta(x_plus_delta);

    // variable scaling
    // "Numerical Optimization" Nocedal et al. 2006, page 95
    Sophus::Vector6d poseDelta;
    poseDelta.segment<3>(0) = d.segment<3>(0) * config.OPTIMIZATION_TRANS_SCALE;
    poseDelta.segment<3>(3) = d.segment<3>(3) * config.OPTIMIZATION_ROTATION_SCALE;

    // pose left composition : exp(delta) * T
    T_plus_delta = Sophus::SE3d::exp(poseDelta) * T;

    // affine light in euclidean space
    x_plus_delta[7] = x[7] + (delta[6] * config.OPTIMIZATION_ALPHA_SCALE);
    x_plus_delta[8] = x[8] + (delta[7] * config.OPTIMIZATION_BETA_SCALE);

    return true;
}

bool FrameParameterization::ComputeJacobian(const double* x, double* jacobian) const {
    // trick to work directly in the tangent space
    // compute jacobians relative to the tangent space
    // and let this jacobian be the identity
    // J = [ I(8x8); 0 ]

    Eigen::Map<Eigen::Matrix<double, 9, 8, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.block<8, 8>(0, 0).setIdentity();

    return true;
}

int FrameParameterization::GlobalSize() const {
    return (Sophus::SE3f::num_parameters + AffineLight::num_parameters);
}

int FrameParameterization::LocalSize() const {
    return (Sophus::SE3f::DoF + AffineLight::DoF);
}

// Parameter Block
const std::unique_ptr<FrameParameterization> FrameParameterBlock::frameParameterization =
    std::make_unique<FrameParameterization>();

FrameParameterBlock::FrameParameterBlock() : BAParameterBlock<9>() {
}

FrameParameterBlock::FrameParameterBlock(const Sophus::SE3d& camToWorld, const AffineLight& affineLight)
    : BAParameterBlock<9>() {
    this->setPose(camToWorld);
    this->setAffineLight(affineLight);
    this->backup();
    this->setLocalParameterization(FrameParameterBlock::frameParameterization.get());
}

FrameParameterBlock::~FrameParameterBlock() {
}

void FrameParameterBlock::setPose(const Sophus::SE3d& camToWorld) {
    // copy values
    // the internal order is: qx, qy, qz, qw, tx, ty, tz
    // the same as in Eigen and Sophus
    std::copy(camToWorld.data(), camToWorld.data() + Sophus::SE3f::num_parameters, this->parameters_.data());
}

void FrameParameterBlock::setAffineLight(const AffineLight& affineLight) {
    // copy values
    // the internal order is: alpha, beta
    this->parameters_[7] = affineLight.alpha();
    this->parameters_[8] = affineLight.beta();
}

Sophus::SE3d FrameParameterBlock::getPose() const {
    // map internal data
    Eigen::Map<const Sophus::SE3d> const state(this->parameters_.data());

    // return a copy
    return state;
}

AffineLight FrameParameterBlock::getAffineLight() const {
    // return a copy
    return AffineLight((float)this->parameters_[7], (float)this->parameters_[8]);
}

Sophus::SE3d FrameParameterBlock::getPoseBackup() const {
    // map internal data
    Eigen::Map<const Sophus::SE3d> const state_backup(this->parameters_backup_.data());

    // return a copy
    return state_backup;
}

AffineLight FrameParameterBlock::getAffineLightBackup() const {
    // return a copy
    return AffineLight((float)this->parameters_backup_[7], (float)this->parameters_backup_[8]);
}

void FrameParameterBlock::step(Eigen::Matrix<double, 8, 1>& delta) const {
    // map buffers
    Eigen::Map<Sophus::SE3d const> const T(this->parameters_backup_.data());
    Eigen::Map<Sophus::SE3d const> const T_plus_delta(this->parameters_.data());

    // pose left composition: T_plus_delta = exp(delta) * T
    //						  delta = log(T_plus_delta *
    // T^-1)
    delta.segment<6>(0) = (T_plus_delta * T.inverse()).log();

    // affine light linear in the euclidean space
    // currentState = backupState + delta
    delta[6] = this->parameters_[7] - this->parameters_backup_[7];
    delta[7] = this->parameters_[8] - this->parameters_backup_[8];
}

void FrameParameterBlock::scaledStep(Eigen::Matrix<double, 8, 1>& delta) const {
    const auto& config = Config::getInstance();

    // map buffers
    Eigen::Map<Sophus::SE3d const> const T(this->parameters_backup_.data());
    Eigen::Map<Sophus::SE3d const> const T_plus_delta(this->parameters_.data());

    // pose left composition: T_plus_delta = exp(delta) * T
    //						  delta = log(T_plus_delta *
    // T^-1)
    delta.segment<6>(0) = (T_plus_delta * T.inverse()).log();

    // affine light linear in the euclidean space
    // currentState = backupState + delta
    delta[6] = this->parameters_[7] - this->parameters_backup_[7];
    delta[7] = this->parameters_[8] - this->parameters_backup_[8];

    // scale
    delta.segment<3>(0) /= config.OPTIMIZATION_TRANS_SCALE;
    delta.segment<3>(3) /= config.OPTIMIZATION_ROTATION_SCALE;
    delta[6] /= config.OPTIMIZATION_ALPHA_SCALE;
    delta[7] /= config.OPTIMIZATION_BETA_SCALE;
}
}  // namespace mpl