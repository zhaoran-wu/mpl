#include "PhotometricBA.h"
#include "cam_data.h"
#include "ceres/FrameParameterBlock.h"
#include "ceres/PhotometricBAIterationCallback.h"
#include "ceres/PhotometricResidual.h"
#include "ceres/PointParameterBlock.h"
#include "config.h"
#include "debug.h"
#include "frame.h"
#include "pattern.h"
#include "visibility.h"

#include <fstream>
#include <iomanip>

namespace mpl {
PhotometricBA::PhotometricBA(const PhotometricBAConfig& config)
    : options(config), numFrames(0), numPoints(0), numResiduals(0) {
    // iteration callback
    this->iterCallback = std::make_unique<PhotometricBAIterationCallback>(*this);
    //! this->options.solverOptions.callbacks.push_back(this->iterCallback.get());
    // todo use callback
}

PhotometricBA::~PhotometricBA() {
}

void PhotometricBA::reset() {
    this->activeKeyframes.clear();
    this->activeObservations.clear();
}

void PhotometricBA::solve(CandidateManager& cm) {
    auto& config = Config::getInstance();
    candidate_map_ptr = &cm.get_candidate_map();
    if (cm.get_key_frames().size() < 3) return;

    // reset internal data
    this->reset();

    // create bundle adjust problem
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

    //! this->removeBadObservations(obsToRemove);

    debug::execute_mem_according_to_config(config.DEBUG_SLIDING_WINDOW, config.debug_sliding_window_mutex,
                                           &PhotometricBA::draw_result, this, &cm);
}

void PhotometricBA::draw_result(CandidateManager* cm) const {
    CamData& cam = CamData::getInstance();
    const auto& kf_vec = cm->get_key_frames();
    auto& candidate_map = cm->get_candidate_map();
    // prepare image to draw
    std::unordered_map<Frame*, cv::Mat> imgs_to_draw;
    for (const auto& kf : kf_vec) {
        cv::Mat gray_im =
            cv::Mat(cv::Size(cam.width[0], cam.height[0]), CV_8UC1, kf->get_image_pyramid()->data(0).get());
        cv::Mat color_im;
        cv::cvtColor(gray_im, color_im, CV_GRAY2BGR);
        imgs_to_draw[kf.get()] = color_im;
    }

    // draw every observation on target frame
    for (const auto& kf : kf_vec) {
        const auto& can_vec = candidate_map[kf];
        for (const auto& can : can_vec) {
            for (const auto& it : can->observations) {
                const auto& obs = it.second;
                const auto& obs_frame = obs->obs_frame();

                const Eigen::Vector2f projection = unproject_trans_project(can.get(), kf.get(), obs_frame);

                cv::circle(imgs_to_draw[obs_frame], cv::Point(projection(0), projection(1)), 2, cv::Scalar(0, 255, 0),
                           2);
            }
        }
    }

    // stiching and show all img
    const int imgs_size = imgs_to_draw.size();
    const int num_img_x = 2;
    const int num_img_y = imgs_size / 2 + (imgs_size % 2);
    int imgs_width = num_img_x * cam.width[0];
    int imgs_height = num_img_y * cam.height[0];

    int interval = 15;

    cv::Mat result = cv::Mat::zeros(
        cv::Size(imgs_width + (num_img_x + 1) * interval, imgs_height + (num_img_y + 1) * interval), CV_8UC3);

    int id = 0;
    for (auto it = kf_vec.begin(); it != kf_vec.end(); ++it) {
        cv::Mat im = imgs_to_draw[it->get()];
        const int row = id / 2;
        const int col = id % 2;

        int ul_x = interval * (col + 1) + cam.width[0] * col;
        int ul_y = interval * (row + 1) + cam.height[0] * row;

        cv::putText(im,                                               // target image
                    "ID : " + std::to_string(id),                     // text
                    cv::Point(interval * 3, interval * 3),            // top-left position
                    cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0),  // font color
                    2);

        im.copyTo(result(cv::Rect(cv::Point(ul_x, ul_y), cv::Size(cam.width[0], cam.height[0]))));

        id++;
    }

    const int max_rows = 750;

    float ratio = static_cast<float>(max_rows) / result.rows;

    cv::resize(result, result, cv::Size(result.cols * ratio, result.rows * ratio));
    cv::imshow("sliding window result", result);
    cv::waitKey(0);
}

void PhotometricBA::prepareOptimization(CandidateManager& cm, BundleAdjustment& problem) {
    const auto& Config = Config::getInstance();
    auto& candidate_map = cm.get_candidate_map();

    // ordering
    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering();

    // add all cameras, points and observations
    for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
        // add host frame to param
        auto kf = it->first;

        problem.addParameterBlock(kf->get_frame_block().get());
        ordering->AddElementToGroup(kf->get_frame_block()->getParameters(), FrameParameterBlock::Group);
        this->activeKeyframes.push_back(kf);

        // add point->param/ obs residual to problem
        if (kf == cm.get_key_frames().back()) continue;  // all candidate on newst kf is non active
        for (const auto& point : candidate_map[kf]) {
            if (point->status == CandidateStatus::NOT_ACTIVE || point->status == CandidateStatus::OUTLIER) continue;
            //! add point->param, try at fix the depth
            problem.addParameterBlock(point->get_point_block().get());
            problem.setParameterBlockConstant(point->get_point_block().get());

            ordering->AddElementToGroup(point->get_point_block()->getParameters(), PointParameterBlock::Group);

            // add obs frame to param and add residuals
            for (const auto& obs_per_frame : point->observations) {
                problem.addParameterBlock(obs_per_frame.first->get_frame_block().get());
                ordering->AddElementToGroup(obs_per_frame.first->get_frame_block()->getParameters(),
                                            FrameParameterBlock::Group);

                problem.addResidualBlock(obs_per_frame.second.get(), kf->get_frame_block().get(),
                                         obs_per_frame.first->get_frame_block().get(), point->get_point_block().get());

                this->activeObservations.push_back(obs_per_frame.second.get());
            }
        }
    }

    this->options.solverOptions.linear_solver_ordering.reset(ordering);
}

void PhotometricBA::mergeOptimization(CandidateManager& cm, std::vector<PhotometricResidual*>& obsToRemove) const {
    // merge all keyframes first
    for (const std::shared_ptr<Frame>& kf : cm.get_key_frames()) {
        // pose and affine light
        kf->merge_optimization_result();
    }

    // merge depth
    for (const std::shared_ptr<Frame>& kf : cm.get_key_frames()) {
        if (kf == cm.get_key_frames().back()) continue;
        for (auto& point : cm.get_candidate_map()[kf]) {
            const Sophus::SE3f& refToWorld = point->host_frame->get_pose();

            if (point->status == CandidateStatus::ACTIVE || point->status == CandidateStatus::OOB) {
                point->merge_optimization_result();
            }

            // const Eigen::Vector3f pt3d =
            // point->host_frame->unproject(point);//point->pt3d();

            for (const auto& observation : point->observations) {
                // const int obsIdx = observation.first->keyframeID();
                const Visibility state = observation.second->state();

                // visibility
                // point->setVisibility(obsIdx, state);

                // remove or compute useful information based on good
                // observations
                if (state == Visibility::OUTLIER) {
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
}

// TODO remove bad point
void PhotometricBA::removeBadObservations(const std::vector<PhotometricResidual*>& obsToRemove) const {
    for (int i = 0; i < obsToRemove.size(); ++i) {
        Candidate* const point = obsToRemove[i]->point();
        // point->observations.erase(obsToRemove[i]->obs_frame());
        if (point->observations.size() < 1) {
            point->status = CandidateStatus::OUTLIER;
        }
    }
}

}  // namespace mpl