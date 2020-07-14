
#pragma once

#include "affine_light.h"
#include "bundle_adjustment.h"

#include "sophus/se3.hpp"

#include "ceres/local_parameterization.h"

#include <memory>

namespace mpl {
// Local parameterization of frame parameters
// In this case pose SE3 and affine light
// Use only this parameterization if you are
// estimating jacobians respect to the tangent space se3
class FrameParameterization : public ceres::LocalParameterization {
   public:
    FrameParameterization();
    virtual ~FrameParameterization();

    virtual bool Plus(const double* x, const double* delta,
                      double* x_plus_delta) const;

    virtual bool ComputeJacobian(const double* x, double* jacobian) const;

    virtual int GlobalSize() const;

    virtual int LocalSize() const;
};

// Frame parameter block
// First 7 components are the pose ( 4 quaternion + 3 translation)
// Last 2 components are the affine light (alpha, beta)
class FrameParameterBlock : public BAParameterBlock<9> {
   public:
    // variable ordering
    static const int Group = 1;

    // default constructor
    FrameParameterBlock();

    // constructor with a pose
    FrameParameterBlock(const Sophus::SE3d& camToWorld,
                        const AffineLight& affineLight);

    // destructor
    ~FrameParameterBlock();

    void setPose(const Sophus::SE3d& camToWorld);
    void setAffineLight(const AffineLight& affineLight);

    Sophus::SE3d getPose() const;
    AffineLight getAffineLight() const;

    Sophus::SE3d getPoseBackup() const;
    AffineLight getAffineLightBackup() const;

    void step(Eigen::Matrix<double, 8, 1>& delta) const;
    void scaledStep(Eigen::Matrix<double, 8, 1>& delta) const;

   private:
    static const std::unique_ptr<FrameParameterization> frameParameterization;
};
}  // namespace mpl