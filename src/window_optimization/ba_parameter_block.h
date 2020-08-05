
#pragma once
#include <array>

namespace ceres {
class LocalParameterization;
}

namespace mpl {
// Dim: dimension of parameter block
template <int Dim>
/**
 * @brief
 * class to pack up the param(and it's back up) and it's parameterization
 *
 */

class BAParameterBlock {
   public:
    inline BAParameterBlock() : fixed_(false), localParameterization_(nullptr) {
        this->parameters_.fill(0);
        this->parameters_backup_.fill(0);
    }

    inline virtual ~BAParameterBlock() {
    }

    // dimensions of the parameter block
    inline int dimension() const {
        return Dim;
    }

    // manage if this block should be optimized
    inline void setFixed(bool fixed) {
        this->fixed_ = fixed;
    }

    inline bool getFixed() const {
        return this->fixed_;
    }

    // parameterization
    inline void setLocalParameterization(ceres::LocalParameterization* localParameterization) {
        this->localParameterization_ = localParameterization;
    }

    inline ceres::LocalParameterization* getLocalParameterization() const {
        return this->localParameterization_;
    }

    // parameters
    inline void setParameters(double* params) {
        std::copy(params, params + Dim, this->parameters_.data());
    }

    inline double* getParameters() {
        return this->parameters_.data();
    }

    inline void backup() {
        this->parameters_backup_ = this->parameters_;
    }

    inline double* getBackupParameters() {
        return this->parameters_backup_.data();
    }

   protected:
    // flag to control if optimized or not
    bool fixed_;

    // parameters
    std::array<double, Dim> parameters_;

    // back up of the parameters
    std::array<double, Dim> parameters_backup_;

    // local parameterization to use
    ceres::LocalParameterization* localParameterization_;
};
}  // namespace mpl
