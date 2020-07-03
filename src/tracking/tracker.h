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
    void set_tracking_ref(const Frame::ptr ref_frame,
                          const PointCloudPyramid::ptr point_cloud_pyramid);

    /**
     * @brief tracking new frame relative pose to reference frame
     *
     * @param to_track_frame new frame
     * @param pose_in_out  T from ref frame to new frame
     * @param affine_in_out A map I of ref to I of new frame
     * @return if the tracking convergence
     */
    bool tracking(Frame::ptr to_track_frame);

   private:
    void generate_movement_predictions();

    // we update our 10 movement predection each times a new frame is tracked
    // those movement prediction are designed for autonomous car
    // 1x movement
    // 0.5x movement
    // 2x movement
    // 2* (0.5x movement + turn left , 0.5x movement + turn right)
    // no movement
    // forward movement
    Sophus::SE3f movement_prediction[10];
    PointCloudPyramid::ptr point_cloud_pyramid_;
    TrackingOptimizer optimizer;

    Frame::ptr curr_ref_frame;

    // movement pridiction part
    Sophus::SE3f T_last_lastKF =
        Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 0, 0));
    AffineLight aff_last_lastKF = AffineLight(0, 0);
};
}  // namespace mpl