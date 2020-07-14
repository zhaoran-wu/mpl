
#pragma once

#include "bundle_adjustment.h"

#include "ceres/local_parameterization.h"

#include <memory>

namespace mpl {
// Local parameterization of point parameters
// In this case, only the inverse depth
class PointParameterization : public ceres::LocalParameterization {
   public:
    PointParameterization();
    virtual ~PointParameterization();

    virtual bool Plus(const double* x, const double* delta,
                      double* x_plus_delta) const;

    virtual bool ComputeJacobian(const double* x, double* jacobian) const;

    virtual int GlobalSize() const;

    virtual int LocalSize() const;
};

// Parameter block of points
// Only onde dimensional parameter
// This class is not needed but makes all the
// optimization easier to understand
class PointParameterBlock : public BAParameterBlock<1> {
   public:
    // variable ordering
    static const int Group = 0;

    // default constructor
    PointParameterBlock();

    // constructor with an inverse depth
    PointParameterBlock(double iDepth);

    // destructor
    ~PointParameterBlock();

    void setIDepth(double iDepth);

    double getIDepth() const;

    double getIDepthBackup() const;

    void step(double& delta) const;
    void scaledStep(double& delta) const;

   private:
    static const std::unique_ptr<PointParameterization> pointParameterization;
};
}  // namespace mpl