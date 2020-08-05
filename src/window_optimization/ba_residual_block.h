#pragma once

#include "ceres/ceres.h"

namespace mpl {
/**
 * @brief class to pack up residual's id and it's cost & loss function
 * drived class should has a point to CostFunction used as a memeber in the CostFunction class
 * to provide nesessary info to evaluate residual and jacobi
 *
 */
class BAResidualBlock {
   public:
    BAResidualBlock();

    virtual ~BAResidualBlock();

    // unique parameter block identifier
    void setID(ceres::ResidualBlockId parameterBlockID);

    ceres::ResidualBlockId getID() const;

    // loss function
    void setLossFunction(ceres::LossFunction* lossFunction);

    ceres::LossFunction* getLossFunction() const;

    // virtual functions to implement in each derived class

    virtual int dimension() const = 0;

    virtual int numParameterBlocks() const = 0;

    virtual ceres::CostFunction* getCostFunction() const = 0;

   protected:
    // unique identifier
    // it is a pointer to ceres residual block
    ceres::ResidualBlockId id_;

    // loss function
    // memory is not managed by this class
    ceres::LossFunction* lossFunction_;
};
}  // namespace mpl