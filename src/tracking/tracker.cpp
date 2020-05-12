#include "tracker.h"

namespace mpl {

void Tracker::set_tracking_ref(
    const PointCloudPyramid::ptr point_cloud_pyramid) {
    assert(point_cloud_pyramid != nullptr);
    this->point_cloud_pyramid_ = point_cloud_pyramid;
}

bool Tracker::tracking(Frame::ptr to_track_frame, Sophus::SE3f& pose_in_out,
                       AffineLight& affine_in_out) {
    generate_movement_predictions(pose_in_out);

    problem->computeResidual() problem->computeJacobian() problem->solve(20);
}

void Tracker::generate_movement_predictions(const Sophus::SE3f& pose_in_out) {
    // todo test movement prediction part
    // 1x movement
    movement_prediction[0] = pose_in_out;

    auto se3_priori = pose_in_out.log();
    // 0.5x movement
    auto half_movement = 0.5f * se3_priori;
    movement_prediction[1] = Sophus::SE3f::exp(half_movement);
    // 2x movement
    movement_prediction[2] = Sophus::SE3f::exp(2 * se3_priori);
    // 2* (0.5x movement + turn left , 0.5x movement + turn right)
    Sophus::SE3f turn_right_small = Sophus::SE3f::rotY(0.03f);
    Sophus::SE3f turn_right_large = Sophus::SE3f::rotY(0.06f);
    Sophus::SE3f turn_left_small = Sophus::SE3f::rotY(-0.03f);
    Sophus::SE3f turn_left_large = Sophus::SE3f::rotY(-0.06f);

    movement_prediction[3] =
        turn_right_small * Sophus::SE3f::exp(half_movement);
    movement_prediction[4] =
        turn_right_large * Sophus::SE3f::exp(half_movement);
    movement_prediction[5] = turn_left_small * Sophus::SE3f::exp(half_movement);
    movement_prediction[6] = turn_left_large * Sophus::SE3f::exp(half_movement);
    // no movement
    movement_prediction[7] =
        Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero());
    // forward movement
    //* notice the movement is T_new_ref, so z negative here
    movement_prediction[8] = Sophus::SE3f::transZ(-0.1);
    movement_prediction[9] = Sophus::SE3f::transZ(-0.3);
}

}  // namespace mpl