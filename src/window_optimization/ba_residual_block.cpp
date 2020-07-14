
#include "ba_residual_block.h"

namespace mpl {
BAResidualBlock::BAResidualBlock() : id_(nullptr), lossFunction_(nullptr) {
}

BAResidualBlock::~BAResidualBlock() {
}

void BAResidualBlock::setID(ceres::ResidualBlockId parameterBlockID) {
    this->id_ = parameterBlockID;
}

ceres::ResidualBlockId BAResidualBlock::getID() const {
    return this->id_;
}

void BAResidualBlock::setLossFunction(ceres::LossFunction* lossFunction) {
    this->lossFunction_ = lossFunction;
}

ceres::LossFunction* BAResidualBlock::getLossFunction() const {
    return this->lossFunction_;
}
}  // namespace mpl