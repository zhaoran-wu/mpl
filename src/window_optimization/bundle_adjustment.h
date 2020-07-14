#pragma once

#include <memory>

#include "ceres/ceres.h"

#include "ba_parameter_block.h"
#include "ba_residual_block.h"

namespace mpl {
// Wrapper of ceres problem
class BundleAdjustment {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BundleAdjustment(const ceres::Problem::Options& problemOptions);
    ~BundleAdjustment();

    // main function to solve the least squares problem
    void solve(const ceres::Solver::Options& solverOptions);

    // optimization summary
    // solve() must be called before
    const ceres::Solver::Summary& summary() const;

    // ParameterBlock management
    template <int Dim>
    void addParameterBlock(BAParameterBlock<Dim>* parameterBlock);

    template <int Dim>
    void removeParameterBlock(BAParameterBlock<Dim>* parameterBlock);

    template <int Dim>
    void setParameterBlockConstant(BAParameterBlock<Dim>* parameterBlock);

    template <int Dim>
    void setParameterBlockVariable(BAParameterBlock<Dim>* parameterBlock);

    template <int Dim>
    void setParameterLowerBound(BAParameterBlock<Dim>* parameterBlock,
                                int index, double lowerBound);

    template <int Dim>
    void setParameterUpperBound(BAParameterBlock<Dim>* parameterBlock,
                                int index, double upperBound);

    // ResidualBlock management
    template <int Dim0>
    void addResidualBlock(BAResidualBlock* residualBlock,
                          BAParameterBlock<Dim0>* x0);

    template <int Dim0, int Dim1>
    void addResidualBlock(BAResidualBlock* residualBlock,
                          BAParameterBlock<Dim0>* x0,
                          BAParameterBlock<Dim1>* x1);

    template <int Dim0, int Dim1, int Dim2>
    void addResidualBlock(BAResidualBlock* residualBlock,
                          BAParameterBlock<Dim0>* x0,
                          BAParameterBlock<Dim1>* x1,
                          BAParameterBlock<Dim2>* x2);

    template <int Dim0, int Dim1, int Dim2, int Dim3>
    void addResidualBlock(BAResidualBlock* residualBlock,
                          BAParameterBlock<Dim0>* x0,
                          BAParameterBlock<Dim1>* x1,
                          BAParameterBlock<Dim2>* x2,
                          BAParameterBlock<Dim3>* x3);

    template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4>
    void addResidualBlock(BAResidualBlock* residualBlock,
                          BAParameterBlock<Dim0>* x0,
                          BAParameterBlock<Dim1>* x1,
                          BAParameterBlock<Dim2>* x2,
                          BAParameterBlock<Dim3>* x3,
                          BAParameterBlock<Dim4>* x4);

    template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5>
    void addResidualBlock(BAResidualBlock* residualBlock,
                          BAParameterBlock<Dim0>* x0,
                          BAParameterBlock<Dim1>* x1,
                          BAParameterBlock<Dim2>* x2,
                          BAParameterBlock<Dim3>* x3,
                          BAParameterBlock<Dim4>* x4,
                          BAParameterBlock<Dim5>* x5);

    template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5,
              int Dim6>
    void addResidualBlock(
        BAResidualBlock* residualBlock, BAParameterBlock<Dim0>* x0,
        BAParameterBlock<Dim1>* x1, BAParameterBlock<Dim2>* x2,
        BAParameterBlock<Dim3>* x3, BAParameterBlock<Dim4>* x4,
        BAParameterBlock<Dim5>* x5, BAParameterBlock<Dim6>* x6);

    template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5,
              int Dim6, int Dim7>
    void addResidualBlock(
        BAResidualBlock* residualBlock, BAParameterBlock<Dim0>* x0,
        BAParameterBlock<Dim1>* x1, BAParameterBlock<Dim2>* x2,
        BAParameterBlock<Dim3>* x3, BAParameterBlock<Dim4>* x4,
        BAParameterBlock<Dim5>* x5, BAParameterBlock<Dim6>* x6,
        BAParameterBlock<Dim7>* x7);

    template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5,
              int Dim6, int Dim7, int Dim8>
    void addResidualBlock(
        BAResidualBlock* residualBlock, BAParameterBlock<Dim0>* x0,
        BAParameterBlock<Dim1>* x1, BAParameterBlock<Dim2>* x2,
        BAParameterBlock<Dim3>* x3, BAParameterBlock<Dim4>* x4,
        BAParameterBlock<Dim5>* x5, BAParameterBlock<Dim6>* x6,
        BAParameterBlock<Dim7>* x7, BAParameterBlock<Dim8>* x8);

    template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5,
              int Dim6, int Dim7, int Dim8, int Dim9>
    void addResidualBlock(
        BAResidualBlock* residualBlock, BAParameterBlock<Dim0>* x0,
        BAParameterBlock<Dim1>* x1, BAParameterBlock<Dim2>* x2,
        BAParameterBlock<Dim3>* x3, BAParameterBlock<Dim4>* x4,
        BAParameterBlock<Dim5>* x5, BAParameterBlock<Dim6>* x6,
        BAParameterBlock<Dim7>* x7, BAParameterBlock<Dim8>* x8,
        BAParameterBlock<Dim9>* x9);

    void removeResidualBlock(ceres::ResidualBlockId residualBlockId);

    // Statistics
    int numParameterBlocks() const;

    int numParameters() const;

    int numResidualBlocks() const;

    int numResiduals() const;

   private:
    // Ceres problem
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
};

// Implementation

template <int Dim>
inline void BundleAdjustment::addParameterBlock(
    BAParameterBlock<Dim>* parameterBlock) {
    parameterBlock->setFixed(false);
    this->problem_->AddParameterBlock(
        parameterBlock->getParameters(), parameterBlock->dimension(),
        parameterBlock->getLocalParameterization());
}

template <int Dim>
inline void BundleAdjustment::removeParameterBlock(
    BAParameterBlock<Dim>* parameterBlock) {
    this->problem_->RemoveParameterBlock(parameterBlock->getParameters());
}

template <int Dim>
inline void BundleAdjustment::setParameterBlockConstant(
    BAParameterBlock<Dim>* parameterBlock) {
    parameterBlock->setFixed(true);
    this->problem_->SetParameterBlockConstant(parameterBlock->getParameters());
}

template <int Dim>
inline void BundleAdjustment::setParameterBlockVariable(
    BAParameterBlock<Dim>* parameterBlock) {
    parameterBlock->setFixed(false);
    this->problem_->SetParameterBlockVariable(parameterBlock->getParameters());
}

template <int Dim>
inline void BundleAdjustment::setParameterLowerBound(
    BAParameterBlock<Dim>* parameterBlock, int index, double lowerBound) {
    this->problem_->SetParameterLowerBound(parameterBlock->getParameters(),
                                           index, lowerBound);
}

template <int Dim>
inline void BundleAdjustment::setParameterUpperBound(
    BAParameterBlock<Dim>* parameterBlock, int index, double upperBound) {
    this->problem_->SetParameterUpperBound(parameterBlock->getParameters(),
                                           index, upperBound);
}

template <int Dim0>
inline void BundleAdjustment::addResidualBlock(BAResidualBlock* residualBlock,
                                               BAParameterBlock<Dim0>* x0) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1>
inline void BundleAdjustment::addResidualBlock(BAResidualBlock* residualBlock,
                                               BAParameterBlock<Dim0>* x0,
                                               BAParameterBlock<Dim1>* x1) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1, int Dim2>
inline void BundleAdjustment::addResidualBlock(BAResidualBlock* residualBlock,
                                               BAParameterBlock<Dim0>* x0,
                                               BAParameterBlock<Dim1>* x1,
                                               BAParameterBlock<Dim2>* x2) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters(), x2->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1, int Dim2, int Dim3>
inline void BundleAdjustment::addResidualBlock(BAResidualBlock* residualBlock,
                                               BAParameterBlock<Dim0>* x0,
                                               BAParameterBlock<Dim1>* x1,
                                               BAParameterBlock<Dim2>* x2,
                                               BAParameterBlock<Dim3>* x3) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters(), x2->getParameters(),
        x3->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4>
inline void BundleAdjustment::addResidualBlock(BAResidualBlock* residualBlock,
                                               BAParameterBlock<Dim0>* x0,
                                               BAParameterBlock<Dim1>* x1,
                                               BAParameterBlock<Dim2>* x2,
                                               BAParameterBlock<Dim3>* x3,
                                               BAParameterBlock<Dim4>* x4) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters(), x2->getParameters(),
        x3->getParameters(), x4->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5>
inline void BundleAdjustment::addResidualBlock(BAResidualBlock* residualBlock,
                                               BAParameterBlock<Dim0>* x0,
                                               BAParameterBlock<Dim1>* x1,
                                               BAParameterBlock<Dim2>* x2,
                                               BAParameterBlock<Dim3>* x3,
                                               BAParameterBlock<Dim4>* x4,
                                               BAParameterBlock<Dim5>* x5) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters(), x2->getParameters(),
        x3->getParameters(), x4->getParameters(), x5->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5, int Dim6>
inline void BundleAdjustment::addResidualBlock(
    BAResidualBlock* residualBlock, BAParameterBlock<Dim0>* x0,
    BAParameterBlock<Dim1>* x1, BAParameterBlock<Dim2>* x2,
    BAParameterBlock<Dim3>* x3, BAParameterBlock<Dim4>* x4,
    BAParameterBlock<Dim5>* x5, BAParameterBlock<Dim6>* x6) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters(), x2->getParameters(),
        x3->getParameters(), x4->getParameters(), x5->getParameters(),
        x6->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5, int Dim6,
          int Dim7>
inline void BundleAdjustment::addResidualBlock(
    BAResidualBlock* residualBlock, BAParameterBlock<Dim0>* x0,
    BAParameterBlock<Dim1>* x1, BAParameterBlock<Dim2>* x2,
    BAParameterBlock<Dim3>* x3, BAParameterBlock<Dim4>* x4,
    BAParameterBlock<Dim5>* x5, BAParameterBlock<Dim6>* x6,
    BAParameterBlock<Dim7>* x7) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters(), x2->getParameters(),
        x3->getParameters(), x4->getParameters(), x5->getParameters(),
        x6->getParameters(), x7->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5, int Dim6,
          int Dim7, int Dim8>
inline void BundleAdjustment::addResidualBlock(
    BAResidualBlock* residualBlock, BAParameterBlock<Dim0>* x0,
    BAParameterBlock<Dim1>* x1, BAParameterBlock<Dim2>* x2,
    BAParameterBlock<Dim3>* x3, BAParameterBlock<Dim4>* x4,
    BAParameterBlock<Dim5>* x5, BAParameterBlock<Dim6>* x6,
    BAParameterBlock<Dim7>* x7, BAParameterBlock<Dim8>* x8) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters(), x2->getParameters(),
        x3->getParameters(), x4->getParameters(), x5->getParameters(),
        x6->getParameters(), x7->getParameters(), x8->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}

template <int Dim0, int Dim1, int Dim2, int Dim3, int Dim4, int Dim5, int Dim6,
          int Dim7, int Dim8, int Dim9>
inline void BundleAdjustment::addResidualBlock(
    BAResidualBlock* residualBlock, BAParameterBlock<Dim0>* x0,
    BAParameterBlock<Dim1>* x1, BAParameterBlock<Dim2>* x2,
    BAParameterBlock<Dim3>* x3, BAParameterBlock<Dim4>* x4,
    BAParameterBlock<Dim5>* x5, BAParameterBlock<Dim6>* x6,
    BAParameterBlock<Dim7>* x7, BAParameterBlock<Dim8>* x8,
    BAParameterBlock<Dim9>* x9) {
    // insert into problem
    ceres::ResidualBlockId residualId = this->problem_->AddResidualBlock(
        residualBlock->getCostFunction(), residualBlock->getLossFunction(),
        x0->getParameters(), x1->getParameters(), x2->getParameters(),
        x3->getParameters(), x4->getParameters(), x5->getParameters(),
        x6->getParameters(), x7->getParameters(), x8->getParameters(),
        x9->getParameters());

    // set identifier
    residualBlock->setID(residualId);
}
}  // namespace mpl
