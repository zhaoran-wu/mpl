#include "PhotometricBA.h"
#include "cam_data.h"
#include "ceres/FrameParameterBlock.h"
#include "ceres/PhotometricBAIterationCallback.h"
#include "ceres/PhotometricResidual.h"
#include "ceres/PointParameterBlock.h"
#include "config.h"
#include "frame.h"
#include "pattern.h"
#include "visibility.h"

#include <fstream>
#include <iomanip>

namespace mpl {
PhotometricBA::PhotometricBA(const PhotometricBAConfig& config)
    : options(config), numFrames(0), numPoints(0), numResiduals(0) {
    // iteration callback
    this->iterCallback =
        std::make_unique<PhotometricBAIterationCallback>(*this);
    this->options.solverOptions.callbacks.push_back(this->iterCallback.get());
}

PhotometricBA::~PhotometricBA() {
}

void PhotometricBA::reset() {
    this->activeKeyframes.clear();
    this->activeObservations.clear();
}

void PhotometricBA::solve(CandidateManager& cm) {
    const auto& config = Config::getInstance();
    candidate_map_ptr = &cm.get_candidate_map();

    // reset internal data
    this->reset();

    // create least squares problem
    BundleAdjustment problem(this->options.problemOptions);

    // prepare data and observations -> very slow!!

    this->prepareOptimization(cm, problem);

    // solve
    problem.solve(this->options.solverOptions);

    // stats
    const ceres::Solver::Summary& summary = problem.summary();

    std::cout << summary.FullReport() << '\n';

    // merge solution
    std::vector<PhotometricResidual*> obsToRemove;
    this->mergeOptimization(cm, obsToRemove);

    // remove bad observations
    // this->removeBadObservations(obsToRemove);
}

void PhotometricBA::prepareOptimization(CandidateManager& cm,
                                        BundleAdjustment& problem) {
    const auto& Config = Config::getInstance();
    auto& candidate_map = cm.get_candidate_map();

    // ordering
    ceres::ParameterBlockOrdering* ordering =
        new ceres::ParameterBlockOrdering();

    // add all cameras, points and observations
    for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
        // add camera
        auto kf = it->first;
        problem.addParameterBlock(kf->get_frame_block().get());
        ordering->AddElementToGroup(kf->get_frame_block()->getParameters(),
                                    FrameParameterBlock::Group);
        this->activeKeyframes.push_back(kf);

        // fix first camera, this fixes unobservable gauge freedoms
        if (kf == cm.get_key_frames().front()) {
            problem.setParameterBlockConstant(kf->get_frame_block().get());
        }

        for (auto& point : candidate_map[kf]) {
            // add point
            problem.addParameterBlock(point.get_point_block().get());
            ordering->AddElementToGroup(
                point.get_point_block()->getParameters(),
                PointParameterBlock::Group);

            // add residuals
            for (const auto& obs : point.observations) {
                problem.addParameterBlock(obs.first->get_frame_block().get());
                ordering->AddElementToGroup(
                    obs.first->get_frame_block()->getParameters(),
                    FrameParameterBlock::Group);

                problem.addResidualBlock(obs.second.get(),
                                         kf->get_frame_block().get(),
                                         obs.first->get_frame_block().get(),
                                         point.get_point_block().get());

                this->activeObservations.push_back(obs.second.get());
            }
        }
    }

    this->options.solverOptions.linear_solver_ordering.reset(ordering);
}

void PhotometricBA::mergeOptimization(
    CandidateManager& cm,
    std::vector<PhotometricResidual*>& obsToRemove) const {
    // merge all keyframes first
    for (const std::shared_ptr<Frame>& kf : cm.get_key_frames()) {
        // pose and affine light
        kf->merge_optimization_result();
    }

    const std::shared_ptr<Frame>& lastKeyfame = cm.get_key_frames().back();
    const int lastIdx = lastKeyfame->get_id();

    // merge all points - THIS CAN BE PARALLELIZED - TODO
    for (const std::shared_ptr<Frame>& kf : cm.get_key_frames()) {
        for (auto& point : cm.get_candidate_map()[kf]) {
            const Sophus::SE3f& refToWorld =
                point.host_frame->get_pose<Sophus::SE3f>().inverse();

            // inverse depth
            point.merge_optimization_result();

            // obtain the optimizatio hessian and
            // maximum parallax between observations
            float iDepthHessian = 0.01f;
            float maxParallax = 1.f;

            // const Eigen::Vector3f pt3d =
            // point.host_frame->unproject(point);//point->pt3d();

            for (const auto& observation : point.observations) {
                // const int obsIdx = observation.first->keyframeID();
                const Visibility state = observation.second->state();

                // visibility
                // point->setVisibility(obsIdx, state);

                // remove or compute useful information based on good
                // observations
                if (state == Visibility::OOB || state == Visibility::OUTLIER) {
                    obsToRemove.push_back(observation.second.get());
                } /* else {
                    const Sophus::SE3f refToTarget =
                        observation.first->camToWorld().inverse() * refToWorld;

                    // hessian
                    iDepthHessian += (float)observation.second->iDepthHessian();

                    // relative parallax
                    const Eigen::Vector3f pt3d_cam =
                        pt3d + refToTarget.rotationMatrix().transpose() *
                                   refToTarget.translation();
                    const float dist = pt3d_cam.norm();

                    const float parallax =
                        pt3d_cam.dot(pt3d) / (dist * pt3d.norm());  // cosine

                    if (parallax < maxParallax)  // if the cosine is lower, the
                                                 // parallax is bigger
                    {
                        maxParallax = parallax;
                    }
                }
            }

            point->setIDepthHessian(iDepthHessian);
            point->setParallax(maxParallax); */
            }
        }
    }

    // TODO remove bad point
/*     void PhotometricBA::removeBadObservations(
        const std::vector<PhotometricResidual*>& obsToRemove) const {
        for (int i = 0; i < obsToRemove.size(); ++i) {
            Candidate* const point = obsToRemove[i]->point();
            point->eraseObservation(obsToRemove[i]->targetFrame());
        }
     */}

}  // namespace mpl