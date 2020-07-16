#include "tracker.h"
#include "../util/tictoc.h"

namespace mpl {

void Tracker::set_tracking_ref(const Frame::ptr ref_frame, const PointCloudPyramid::ptr point_cloud_pyramid) {
    assert(point_cloud_pyramid != nullptr);
    this->point_cloud_pyramid_ = point_cloud_pyramid;
    curr_ref_frame = ref_frame;
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
    int iterations = 20;
    float lamda_init = 1;
    int huber_radius = 100;
    float lamda_min = 1e-7;
    //
    for (int i = 0; i < 22; ++i) {
        init_pose = movement_prediction[i] * T_last_lastKF;
        optimizer.init(init_pose, init_aff_light, point_cloud_pyramid_, to_track_frame);
        optimizer.set_lvl(lvls - 1);
        float energy = optimizer.solve(iterations, lamda_init, lamda_min, huber_radius);
        std::cout << " idx :" << i << "energy per pixel :" << energy / point_cloud_pyramid_->operator[](lvls - 1).size()
                  << '\n';
        if (energy < min_energy) {
            idx = i;
            min_energy = energy;
        }
    }
    // detect if tracking failed
    if (min_energy / point_cloud_pyramid_->operator[](lvls - 1).size() > 680) {
        ++this->failaure_cnt;
        return false;
    } else {
        failaure_cnt = 0;
    }

    std::cout << "best movement idx : " << idx << '\n';
    init_pose = movement_prediction[idx] * T_last_lastKF;
    // tracking with best prediction

    optimizer.init(init_pose, init_aff_light, point_cloud_pyramid_, to_track_frame);

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
        min_energy = optimizer.solve(iterations, lamda_init, lamda_min, huber_radius);
    }
    LOG(INFO) << "tracking use time : " << tictoc::toc() / 1000.f << "ms";
    std::cout << "final energy per pixel :" << min_energy / point_cloud_pyramid_->operator[](0).size() << '\n';
    if (min_energy / point_cloud_pyramid_->operator[](0).size() > 450) {
        ++this->failaure_cnt;
        return false;
    } else {
        failaure_cnt = 0;
    }
    per_pixel_energy = min_energy / point_cloud_pyramid_->operator[](0).size();
    Sophus::SE3f T_curr_lastKF = optimizer.getT();
    AffineLight aff_curr_lastKF = optimizer.getAffineLight();

    to_track_frame->set_tracking_result(T_curr_lastKF, aff_curr_lastKF);

    T_last_lastKF = T_curr_lastKF;
    aff_last_lastKF = aff_curr_lastKF;

    over_last_frame = last_frame;
    last_frame = to_track_frame;

    return true;
}

void Tracker::generate_movement_predictions() {
    Sophus::SE3f T_last_overlast;
    AffineLight aff_last_overlast;

    if (last_frame == nullptr || over_last_frame == nullptr) {
        T_last_overlast = Sophus::SE3f::transZ(-0.7f);

        aff_last_overlast = AffineLight(0, 0);
    } else {
        T_last_overlast = get_src_to_dst_transform(over_last_frame, last_frame);
        aff_last_overlast = get_src_to_dst_aff_light(over_last_frame, last_frame);
    }

    std::cout << "last translation :" << T_last_overlast.translation().transpose() << '\n';
    std::cout << "last rotation X :" << T_last_overlast.angleX() << " last rotation Y :" << T_last_overlast.angleY()
              << " last roration Z : " << T_last_overlast.angleZ() << "aff a :" << aff_last_overlast.alpha()
              << "aff b:  " << aff_last_overlast.beta() << '\n';

    // 1x movement // assume T_last_overlast = T_curr_last
    movement_prediction[0] = T_last_overlast;

    const Sophus::Vector6f se3_priori = T_last_overlast.log() * (failaure_cnt + 1);
    // X x movement
    movement_prediction[1] = Sophus::SE3f::exp(0.5 * se3_priori);
    movement_prediction[2] = Sophus::SE3f::exp(0.75 * se3_priori);
    movement_prediction[3] = Sophus::SE3f::exp(1.25 * se3_priori);
    movement_prediction[4] = Sophus::SE3f::exp(1.5 * se3_priori);
    movement_prediction[5] = Sophus::SE3f::exp(1.75 * se3_priori);
    // 2* (movement + turn left , movement + turn right)
    Sophus::SE3f turn_right_small = Sophus::SE3f::rotY(-0.02f);
    Sophus::SE3f turn_right_mid = Sophus::SE3f::rotY(-0.04f);
    Sophus::SE3f turn_right_large = Sophus::SE3f::rotY(-0.06f);
    Sophus::SE3f turn_left_small = Sophus::SE3f::rotY(0.02f);
    Sophus::SE3f turn_left_mid = Sophus::SE3f::rotY(0.04f);
    Sophus::SE3f turn_left_large = Sophus::SE3f::rotY(0.06f);

    movement_prediction[6] = Sophus::SE3f::exp(se3_priori) * turn_right_small;
    movement_prediction[7] = Sophus::SE3f::exp(se3_priori) * turn_right_mid;
    movement_prediction[8] = Sophus::SE3f::exp(se3_priori) * turn_right_large;
    movement_prediction[9] = Sophus::SE3f::exp(se3_priori) * turn_left_small;
    movement_prediction[10] = Sophus::SE3f::exp(se3_priori) * turn_left_mid;
    movement_prediction[11] = Sophus::SE3f::exp(se3_priori) * turn_left_large;
    // no movement
    movement_prediction[12] = Sophus::SE3f::trans(0, 0, 0);

    // forward movement
    //* notice the movement is T_new_ref, so z negative here
    // experiment
    movement_prediction[13] = Sophus::SE3f::rotY(-0.03f) * Sophus::SE3f::exp(0.5 * se3_priori);
    movement_prediction[14] = Sophus::SE3f::rotY(0.03f) * Sophus::SE3f::exp(0.5 * se3_priori);
    movement_prediction[15] = Sophus::SE3f::rotY(-0.01f) * Sophus::SE3f::exp(0.5 * se3_priori);
    movement_prediction[16] = Sophus::SE3f::rotY(0.01f) * Sophus::SE3f::exp(0.5 * se3_priori);

    movement_prediction[17] = Sophus::SE3f::transZ(-0.32) * Sophus::SE3f::rotY(-0.06f);
    movement_prediction[18] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(-0.08);
    movement_prediction[19] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(-0.1f);
    movement_prediction[20] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(-0.12f);
}

}  // namespace mpl