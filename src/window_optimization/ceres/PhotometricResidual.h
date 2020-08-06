#pragma once

#include "ceres/cost_function.h"

#include "bundle_adjustment.h"
#include "frame.h"
#include "visibility.h"

namespace mpl {
class PhotometricBA;
class PhotometricResidual;

class Candidate;

// Cost function
class PhotometricCostFunction : public ceres::CostFunction {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    PhotometricCostFunction(PhotometricResidual* const residual);

    virtual ~PhotometricCostFunction();

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

   private:
    void discardOutlier(double** jacobians) const;
    void discardOutlier(double** jacobians, int idx) const;
    void discardOOB(double* residuals, double** jacobians) const;

   private:
    // reference to residual
    PhotometricResidual* const residual_;
};

// Residual block
class PhotometricResidual : public BAResidualBlock {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    friend class PhotometricCostFunction;

    PhotometricResidual(Candidate* point, Frame* targetFrame) noexcept;
    virtual ~PhotometricResidual();

    // avoid copying
    PhotometricResidual(const PhotometricResidual&) = delete;
    PhotometricResidual& operator=(const PhotometricResidual&) = delete;

    // get current state residuals
    bool evaluate(int lvl, Eigen::VectorXf& residuals) const;

    // number of residual dimensions
    int dimension() const;

    // number of parameter blocks for this residual
    int numParameterBlocks() const;

    // returns the cost function
    ceres::CostFunction* getCostFunction() const;

    // point
    Candidate* point() const;

    // frames
    Frame* host_frame() const;
    Frame* obs_frame() const;

    // optimization state
    Visibility state() const;

    // point inverse depth hessian
    double iDepthHessian() const;

    // final energy
    double energy() const;
    const Eigen::VectorXd& pixelEnergy() const;

    // huber loss per pixel
    double lossWeight() const;

   private:
    // reference to parameters(1 obsevation)
    Frame* host_frame_;
    Frame* target_frame_;
    Candidate* point_;

    // status
    Visibility state_;

    // inverse depth hessian
    double iDepthHessian_;

    // residual
    double energy_;
    Eigen::VectorXd pixelEnergy_;

    // weight
    double lossWeight_;

    // cost function
    std::unique_ptr<ceres::CostFunction> costFunction_;
};
}  // namespace mpl