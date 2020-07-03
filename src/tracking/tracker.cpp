#include "tracker.h"
#include "../util/tictoc.h"

namespace mpl {

void Tracker::set_tracking_ref(
    const Frame::ptr ref_frame,
    const PointCloudPyramid::ptr point_cloud_pyramid) {
    assert(point_cloud_pyramid != nullptr);
    this->point_cloud_pyramid_ = point_cloud_pyramid;
    curr_ref_frame = ref_frame;
}

bool Tracker::tracking(Frame::ptr to_track_frame, Sophus::SE3f& pose_in_out,
                       AffineLight& affine_in_out) {
    // todo use movement prdiction mode, and pick best inital value
    // generate_movement_predictions(pose_in_out);
    to_track_frame->set_ref_frame(curr_ref_frame);

    Sophus::SE3f init_pose(pose_in_out);
    AffineLight init_aff_light(affine_in_out);

    optimizer.init(init_pose, init_aff_light, point_cloud_pyramid_,
                   to_track_frame);

    int lvls = point_cloud_pyramid_->lvls();

    tictoc::tic();
    for (int lvl = lvls - 1; lvl >= 0; --lvl) {
        optimizer.set_lvl(lvl);
        int iterations;
        float lamda_init;
        int huber_radius;
        float lamda_min;
        if (lvl > 4) {
            iterations = 8;
            lamda_init = 1e-2;
            huber_radius = 100;
            lamda_min = 1e-5;
        } else {
            auto& config = Config::getInstance();
            iterations = config.max_iteration_each_lvl[lvl];
            lamda_init = config.lamda_init_each_lvl[lvl];
            lamda_min = config.lamda_min_eahc_lvl[lvl];
            huber_radius = config.huber_residual_each_lvl[lvl];
        }
        optimizer.solve(iterations, lamda_init, lamda_min, huber_radius);
    }
    LOG(INFO) << "tracking use time : " << tictoc::toc() / 1000.f << "ms";

    pose_in_out = optimizer.getT();
    affine_in_out = optimizer.getAffineLight();

    to_track_frame->set_state(pose_in_out, affine_in_out);

    return true;
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