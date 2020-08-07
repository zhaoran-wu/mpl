#include "PhotometricBAIterationCallback.h"
#include "FrameParameterBlock.h"
#include "PhotometricResidual.h"
#include "PointParameterBlock.h"
#include "candidate_manager.h"
#include "config.h"
#include "frame.h"
#include "pattern.h"

namespace mpl {
PhotometricBAIterationCallback::PhotometricBAIterationCallback(const PhotometricBA& bundleAdjustment)
    : bundle(bundleAdjustment) {
}

PhotometricBAIterationCallback::~PhotometricBAIterationCallback() {
}

ceres::CallbackReturnType PhotometricBAIterationCallback::operator()(const ceres::IterationSummary& summary) {
    bool canBreak = false;

    // termination criteria
    if (summary.iteration > 0 && summary.step_is_successful) {
        canBreak = this->checkTerminationCriteria();
    }

    if (summary.step_is_successful) {
        // state backup

        this->backup();
    }

    // converged?
    if (canBreak && summary.iteration >= 50) {
        return ceres::CallbackReturnType::SOLVER_TERMINATE_SUCCESSFULLY;
    }

    return ceres::CallbackReturnType::SOLVER_CONTINUE;
}

void PhotometricBAIterationCallback::backup() const {
    for (const Frame::ptr frame : this->bundle.activeKeyframes) {
        frame->get_frame_block()->backup();
        for (const auto& point : bundle.candidate_map_ptr->operator[](frame)) {
            point->get_point_block()->backup();
        }
    }
}

bool PhotometricBAIterationCallback::checkTerminationCriteria() const {
    double deltaAlpha = 0;
    double deltaBeta = 0;
    double deltaRot = 0;
    double deltaTrans = 0;
    double meanIDepth = 0;
    double numPoints = 0;

    Eigen::Matrix<double, 8, 1> frameStep;

    for (auto frame : this->bundle.activeKeyframes) {
        // obtain last iteration steps
        frame->get_frame_block()->scaledStep(frameStep);

        // sum
        deltaTrans += frameStep.segment<3>(0).squaredNorm();
        deltaRot += frameStep.segment<3>(3).squaredNorm();
        deltaAlpha += frameStep[6] * frameStep[6];
        deltaBeta += frameStep[7] * frameStep[7];

        for (const auto& point : bundle.candidate_map_ptr->operator[](frame)) {
            meanIDepth += fabs(point->get_point_block()->getIDepthBackup());
            numPoints++;
        }
    }

    int numKeyframes = (int)this->bundle.activeKeyframes.size();

    deltaAlpha /= numKeyframes;
    deltaBeta /= numKeyframes;
    deltaTrans /= numKeyframes;
    deltaRot /= numKeyframes;
    meanIDepth /= numPoints;

    bool converged = sqrt(deltaAlpha) < 0.0006 &&              // affine light a change
                     sqrt(deltaBeta) < 0.00006 &&              // affine light b change
                     sqrt(deltaRot) < 0.00006 &&               // transformation R change
                     sqrt(deltaTrans) * meanIDepth < 0.00006;  // transformation T change respet to inverse depths

    return converged;
}
}  // namespace mpl
