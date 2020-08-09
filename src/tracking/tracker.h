#pragma once
#include "affine_light.h"
#include "frame.h"
#include "point_cloud_pyramid.h"
#include "sophus/se3.hpp"
#include "tracking_optimizer.h"
#include <glog/logging.h>

namespace mpl {

class Tracker {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracker() = default;
    /**
     * @brief pass pre-computed reference point cloud for tracking
     *
     * @param point_cloud_pyramid
     */

    void set_tracking_ref(const Frame::ptr ref_frame, const PointCloudPyramid::ptr point_cloud_pyramid);

    /**
     * @brief tracking new frame relative pose to reference frame
     *
     * @param to_track_frame new frame
     * @param pose_in_out  T from ref frame to new frame
     * @param affine_in_out A map I of ref to I of new frame
     * @return if the tracking convergence
     */
    bool tracking(Frame::ptr to_track_frame);

    float get_per_pixel_energy() const;

    /**
     * @brief tracking the frame itself
     *
     * @param to_refine_frame
     * @param point_cloud_pyramid: point cloud in current synetic image coordinate
     */
    void refine_pose(Frame::ptr to_refine_frame);

   private:
    // generate pediction T_curr_last
    void generate_movement_predictions();
    void draw_result(PointCloudPyramid::ptr pcp, const std::vector<Sophus::SE3f>& T_vec, Frame::ptr frame,
                     const float time_cost, const std::vector<float> energy_vec) const;

    // we update our 10 movement predection each times a new frame is tracked
    // those movement prediction are designed for autonomous car
    // 1x movement
    // 0.5x movement
    // 2x movement
    // 2* (0.5x movement + turn left , 0.5x movement + turn right)
    // no movement
    // forward movement
    Sophus::SE3f movement_prediction[25];
    PointCloudPyramid::ptr point_cloud_pyramid_;  // in synetic image coordinate
    TrackingOptimizer optimizer;

    Frame::ptr curr_ref_frame;

    // movement pridiction part
    Sophus::SE3f T_last_lastKF = Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 0, 0));
    AffineLight aff_last_map = AffineLight(0, 0);

    Frame::ptr last_frame = nullptr;
    Frame::ptr over_last_frame = nullptr;

    float per_pixel_energy;

    int failaure_cnt = 0;
};

inline float Tracker::get_per_pixel_energy() const {
    return per_pixel_energy;
}

}  // namespace mpl