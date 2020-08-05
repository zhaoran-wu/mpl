
#pragma once

#include "affine_light.h"
#include "bundle_adjustment.h"

#include "sophus/se3.hpp"

#include "ceres/local_parameterization.h"

#include <memory>

namespace mpl {
/**
 * @brief Frame parameterization: ceres interface to update state , since for some parameter the local
 * paramter is not the same with the global parameter(num are equal to num the of freedom) and compute Jacobian(chain
 * rule)
 *
 */
class FrameParameterization : public ceres::LocalParameterization {
   public:
    FrameParameterization();
    virtual ~FrameParameterization();

    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const;

    virtual bool ComputeJacobian(const double* x, double* jacobian) const;

    virtual int GlobalSize() const;

    virtual int LocalSize() const;
};

/**
 * @brief class to pack up the frame param(and it's back up) and it's parameterization
 *
 */
class FrameParameterBlock : public BAParameterBlock<9> {
   public:
    // variable ordering
    static const int Group = 1;

    FrameParameterBlock();

    FrameParameterBlock(const Sophus::SE3d& T_w_c, const AffineLight& Aff_w_c);

    ~FrameParameterBlock();

    void setPose(const Sophus::SE3d& T_w_c);
    void setAffineLight(const AffineLight& Aff_w_c);

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