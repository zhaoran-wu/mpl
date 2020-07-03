#include "tracker.h"
#include "../util/tictoc.h"

namespace mpl {

void Tracker::set_tracking_ref(
    const Frame::ptr ref_frame,
    const PointCloudPyramid::ptr point_cloud_pyramid) {
    assert(point_cloud_pyramid != nullptr);
    this->point_cloud_pyramid_ = point_cloud_pyramid;
    curr_ref_frame = ref_frame;

    T_last_lastKF = Sophus::SE3f(Eigen::Quaternionf::Identity(),
                                 Eigen::Vector3f(0, 0, -1.0f));
    aff_last_lastKF = AffineLight(0, 0);
}

bool Tracker::tracking(Frame::ptr to_track_frame) {
    // select best movement prediction

    to_track_frame->set_ref_frame(curr_ref_frame);
    generate_movement_predictions();
    Sophus::SE3f init_pose;
    AffineLight init_aff_light = aff_last_lastKF;
    float min_energy = std::numeric_limits<float>::max();
    int idx = 0;
    int lvls = point_cloud_pyramid_->lvls();
    int iterations = 10;
    float lamda_init = 1e-2;
    int huber_radius = 100;
    float lamda_min = 1e-5;

    for (int i = 0; i < 10; ++i) {
        init_pose = movement_prediction[i];
        optimizer.init(init_pose, init_aff_light, point_cloud_pyramid_,
                       to_track_frame);
        optimizer.set_lvl(lvls - 1);
        float energy =
            optimizer.solve(iterations, lamda_init, lamda_min, huber_radius);
        std::cout << " idx :" << i << "energy :" << energy << '\n';
        if (energy < min_energy) {
            idx = i;
            min_energy = energy;
        }
    }

    std::cout << "best movement idx : " << idx << '\n';
    init_pose = movement_prediction[idx];
    // tracking with best prediction

    optimizer.init(init_pose, init_aff_light, point_cloud_pyramid_,
                   to_track_frame);

    tictoc::tic();
    for (int lvl = lvls - 1; lvl >= 0; --lvl) {
        optimizer.set_lvl(lvl);

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

    Sophus::SE3f T_curr_lastKF = optimizer.getT();
    AffineLight aff_curr_lastKF = optimizer.getAffineLight();

    to_track_frame->set_state(T_curr_lastKF, aff_curr_lastKF);

    T_last_lastKF = T_curr_lastKF;
    aff_last_lastKF = aff_last_lastKF;

    std::cout << "last translation :" << T_curr_lastKF.translation().transpose()
              << '\n';
    std::cout << "last X :" << T_curr_lastKF.angleX()
              << " last Y :" << T_curr_lastKF.angleY()
              << " last Z : " << T_curr_lastKF.angleZ() << '\n';

    return true;
}

void Tracker::generate_movement_predictions() {
    // 1x movement
    movement_prediction[0] = T_last_lastKF;

    auto se3_priori = T_last_lastKF.log();
    // 0.5x movement
    auto half_movement = 0.5f * se3_priori;
    movement_prediction[1] = Sophus::SE3f::exp(half_movement);
    // 2x movement
    movement_prediction[2] = Sophus::SE3f::exp(2 * se3_priori);
    // 2* (movement + turn left , movement + turn right)
    Sophus::SE3f turn_right_small = Sophus::SE3f::rotY(0.01f);
    Sophus::SE3f turn_right_large = Sophus::SE3f::rotY(0.02f);
    Sophus::SE3f turn_left_small = Sophus::SE3f::rotY(-0.01f);
    Sophus::SE3f turn_left_large = Sophus::SE3f::rotY(-0.02f);

    movement_prediction[3] = turn_right_small * Sophus::SE3f::exp(se3_priori);
    movement_prediction[4] = turn_right_large * Sophus::SE3f::exp(se3_priori);
    movement_prediction[5] = turn_left_small * Sophus::SE3f::exp(se3_priori);
    movement_prediction[6] = turn_left_large * Sophus::SE3f::exp(se3_priori);
    // no movement
    movement_prediction[7] =
        Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero());
    // forward movement
    //* notice the movement is T_new_ref, so z negative here
    movement_prediction[8] = Sophus::SE3f::transZ(-0.3f);
    movement_prediction[9] = Sophus::SE3f::transZ(-1.7f);
}

}  // namespace mpl