#include "tracker.h"
#include "../util/tictoc.h"
#include "debug.h"
#include <opencv2/highgui.hpp>

namespace mpl {

void Tracker::set_tracking_ref(const Frame::ptr ref_frame, const PointCloudPyramid::ptr point_cloud_pyramid) {
    assert(point_cloud_pyramid != nullptr);
    this->point_cloud_pyramid_ = point_cloud_pyramid;
    curr_ref_frame = ref_frame;
}

bool Tracker::tracking(Frame::ptr to_track_frame) {
    auto& config = Config::getInstance();
    // select best movement prediction

    to_track_frame->set_ref_frame(curr_ref_frame);
    generate_movement_predictions();

    Sophus::SE3f init_pose;
    AffineLight init_aff_light = aff_last_map;

    float min_energy = std::numeric_limits<float>::max();
    int idx = 0;
    int lvls = point_cloud_pyramid_->lvls();
    int iterations = config.max_iteration_each_lvl[lvls - 1];
    float lamda_init = config.lamda_init_each_lvl[lvls - 1];
    int huber_radius = config.huber_residual_each_lvl[lvls - 1];
    float lamda_min = config.lamda_min_eahc_lvl[lvls - 1];
    //
    tictoc::tic();
    for (int i = 0; i < 7; ++i) {
        init_pose = movement_prediction[i] * T_last_lastKF;
        optimizer.init(init_pose, init_aff_light, point_cloud_pyramid_, to_track_frame);
        optimizer.set_lvl(lvls - 1);
        float energy = optimizer.solve(iterations, lamda_init, lamda_min, huber_radius, true);
        // std::cout << " idx :" << i << "energy per pixel :" << energy << '\n';
        if (energy < min_energy && energy > 1e-10) {
            idx = i;
            min_energy = energy;
            if (energy < 10.0f) break;
        }
    }
    // detect if tracking failed
    float thresh1 = (to_track_frame->get_id() < 10) ? 30 : 20;

    if (min_energy > thresh1) {
        std::cout << "@@@@@@@@@@@@tracking failed : no good prediction"
                  << "    min per pixel energy:   " << min_energy << '\n';
        ++this->failaure_cnt;

        if (min_energy > 1e4) {
            std::exit(EXIT_FAILURE);
        }
        return false;
    } else {
        failaure_cnt = 0;
    }

    std::cout << "best movement idx : " << idx << '\n';
    init_pose = movement_prediction[idx] * T_last_lastKF;
    // tracking with best prediction

    optimizer.init(init_pose, init_aff_light, point_cloud_pyramid_, to_track_frame);

    std::vector<Sophus::SE3f> T_curr_lastKF_pyramid;
    std::vector<float> energy_vec;

    for (int lvl = lvls - 1; lvl >= 0; --lvl) {
        optimizer.set_lvl(lvl);

        if (lvl > 4) {
            iterations = 8;
            lamda_init = 1e-2;
            huber_radius = 35;
            lamda_min = 1e-5;
        } else {
            iterations = config.max_iteration_each_lvl[lvl];
            lamda_init = config.lamda_init_each_lvl[lvl];
            lamda_min = config.lamda_min_eahc_lvl[lvl];
            huber_radius = config.huber_residual_each_lvl[lvl];
        }
        min_energy = optimizer.solve(iterations, lamda_init, lamda_min, huber_radius, false);
        T_curr_lastKF_pyramid.push_back(optimizer.getT());
        energy_vec.push_back(min_energy);
    }

    float time_cost = tictoc::toc();
    debug::execute_mem_according_to_config(
        config.DEBUG_COARSE_TO_FINE_TRACKING, config.debug_coarse_to_fine_tracking_mutex, &Tracker::draw_result, this,
        point_cloud_pyramid_, T_curr_lastKF_pyramid, to_track_frame, time_cost, energy_vec);

    Sophus::SE3f T_curr_lastKF = optimizer.getT();
    AffineLight aff_curr_map = optimizer.getAffineLight();

    float thresh = (to_track_frame->get_id() < 10) ? 30 : 20;
    if (min_energy > thresh) {
        std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@ tracking failed :"
                  << "per pixel energy :" << min_energy << "alpha : " << aff_curr_map.alpha()
                  << " beta : " << aff_curr_map.beta() << "@@@@@@@" << '\n';
        ++this->failaure_cnt;
        return false;
    } else {
        failaure_cnt = 0;
    }

    per_pixel_energy = min_energy;
    to_track_frame->set_tracking_result(T_curr_lastKF, aff_curr_map);

    T_last_lastKF = T_curr_lastKF;
    aff_last_map = aff_curr_map;

    over_last_frame = last_frame;
    last_frame = to_track_frame;

    return true;
}

void Tracker::generate_movement_predictions() {
    Sophus::SE3f T_last_overlast;

    if (last_frame == nullptr || over_last_frame == nullptr) {
        T_last_overlast = Sophus::SE3f::transZ(-0.7f);

    } else {
        T_last_overlast = get_src_to_dst_transform(over_last_frame, last_frame);
    }

    std::cout << "last translation :" << T_last_overlast.translation().transpose() << '\n';
    std::cout << "last rotation X :" << T_last_overlast.angleX() << " last rotation Y :" << T_last_overlast.angleY()
              << " last roration Z : " << T_last_overlast.angleZ() << "aff alpha:" << aff_last_map.alpha()
              << "aff beta:  " << aff_last_map.beta() << '\n';

    // 1x movement // assume T_last_overlast = T_curr_last
    movement_prediction[0] = T_last_overlast;

    const Sophus::Vector6f se3_priori = T_last_overlast.log() * (failaure_cnt + 1);
    // X x movement
    movement_prediction[1] = Sophus::SE3f::trans(0, 0, 0);
    movement_prediction[2] = Sophus::SE3f::exp(0.5 * se3_priori);
    movement_prediction[3] = Sophus::SE3f::exp(0.75 * se3_priori);
    movement_prediction[4] = Sophus::SE3f::exp(1.25 * se3_priori);
    movement_prediction[5] = Sophus::SE3f::exp(1.5 * se3_priori);
    movement_prediction[6] = Sophus::SE3f::exp(1.75 * se3_priori);
    // small delta
    // movement_prediction[7] = Sophus::SE3f::trans(0.0f, -0.0f, -0.8f) * Sophus::SE3f::rotX(0.006f);
    // movement_prediction[8] = Sophus::SE3f::trans(0.0f, -0.0f, -0.8f) * Sophus::SE3f::rotX(-0.006f);
    // movement_prediction[9] = Sophus::SE3f::trans(0.0f, -0.0f, -0.8f) * Sophus::SE3f::rotZ(0.003f);
    // movement_prediction[10] = Sophus::SE3f::trans(0.0f, -0.0f, -0.8f) * Sophus::SE3f::rotZ(-0.003f);
    //// small z
    // movement_prediction[11] = Sophus::SE3f::transZ(-0.2f);
    // movement_prediction[12] = Sophus::SE3f::transZ(-0.4f);
    // 2* (movement + turn left , movement + turn right)
    // Sophus::SE3f turn_right_small = Sophus::SE3f::rotY(-0.02f);
    // Sophus::SE3f turn_right_mid = Sophus::SE3f::rotY(-0.04f);
    // Sophus::SE3f turn_right_large = Sophus::SE3f::rotY(-0.06f);
    // Sophus::SE3f turn_left_small = Sophus::SE3f::rotY(0.02f);
    // Sophus::SE3f turn_left_mid = Sophus::SE3f::rotY(0.04f);
    // Sophus::SE3f turn_left_large = Sophus::SE3f::rotY(0.06f);
    //
    //    movement_prediction[13] = Sophus::SE3f::exp(se3_priori) * turn_right_small;
    //    movement_prediction[14] = Sophus::SE3f::exp(se3_priori) * turn_right_mid;
    //    movement_prediction[15] = Sophus::SE3f::exp(se3_priori) * turn_right_large;
    //    movement_prediction[16] = Sophus::SE3f::exp(se3_priori) * turn_left_small;
    //    movement_prediction[17] = Sophus::SE3f::exp(se3_priori) * turn_left_mid;
    //    movement_prediction[18] = Sophus::SE3f::exp(se3_priori) * turn_left_large;
    // no movement

    // movement_prediction[17] = Sophus::SE3f::transZ(-0.32) * Sophus::SE3f::rotY(-0.06f);
    // movement_prediction[18] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(-0.08);
    // movement_prediction[19] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(-0.1f);
    // movement_prediction[20] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(-0.12f);
    // movement_prediction[21] = Sophus::SE3f::transZ(-0.32) * Sophus::SE3f::rotY(+0.06f);
    // movement_prediction[22] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(+0.08);
    // movement_prediction[23] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(+0.1f);
    // movement_prediction[24] = Sophus::SE3f::transZ(-0.4) * Sophus::SE3f::rotY(+0.12f);
}

void Tracker::draw_result(PointCloudPyramid::ptr pcp, const std::vector<Sophus::SE3f>& T_vec, Frame::ptr frame,
                          const float time_cost, const std::vector<float> energy_vec) const {
    const int interval = 15;
    cv::Size back_ground_size(frame->width() + 2 * interval, frame->height() * 1.5 + 3 * interval);

    cv::Mat result = cv::Mat::zeros(back_ground_size, CV_8UC3);

    int l = interval, u = interval;

    for (int lvl = 0; lvl < pcp->lvls(); ++lvl) {
        cv::Size size_im(frame->width(lvl), frame->height(lvl));

        cv::Mat im(size_im, CV_8UC1);
        im.data = frame->get_image_pyramid()->data(lvl).get();

        cv::Rect roi(l, u, frame->width(lvl), frame->height(lvl));

        cv::rectangle(result, roi, cv::Scalar(255, 0, 0), 2);

        cv::Mat im_to_draw;
        cv::cvtColor(im, im_to_draw, cv::COLOR_GRAY2BGR);
        // draw all candidate on image(with tracking energy)
        if (lvl != 0) {
            for (const auto& point : pcp->operator[](lvl)) {
                Eigen::Vector2f p =
                    frame->project(T_vec[T_vec.size() - lvl - 1] * point.position, lvl);  // todo avoid reprojection
                int thickness = (lvl == 0) ? 2 : 1;
                cv::Scalar color;
                if (lvl != 0) {
                    color = cv::Scalar(0, 255, 0);
                    cv::circle(im_to_draw, cv::Point(p(0), p(1)), 1, color, thickness);
                }
            }
        } else {
            // compute statistic
            int point_size = pcp->operator[](0).size();
            std::vector<float> energy_vec;
            energy_vec.reserve(point_size);

            for (const auto& point : pcp->operator[](0)) {
                if (!point.vis_data.visible_for_newst_frame) continue;
                energy_vec.push_back(point.vis_data.last_tracking_energy);
            }

            const float lo_percent = 0.05;
            const float hi_percent = 0.95;
            float lo, hi;
            std::sort(energy_vec.begin(), energy_vec.end());
            lo = energy_vec[lo_percent * energy_vec.size()];
            hi = energy_vec[hi_percent * energy_vec.size()];

            cv::Mat energy_map = cv::Mat(im_to_draw.size(), CV_8UC1, cv::Scalar(0));

            for (auto& point : pcp->operator[](0)) {
                if (!point.vis_data.visible_for_newst_frame) continue;
                Eigen::Vector2f hit_pixel = point.vis_data.hit_pixel_in_newst_frame;

                float energy = point.vis_data.last_tracking_energy;

                energy = std::min(std::max(energy, lo), hi);

                uchar wrapped_energy = 255 * std::pow((energy - lo) / (hi - lo), 0.7f);

                cv::circle(energy_map, cv::Point2f(hit_pixel(0), hit_pixel(1)), 1, cv::Scalar(wrapped_energy), 2);
            }
            cv::Mat mask = energy_map.clone();
            cv::applyColorMap(energy_map, energy_map, cv::COLORMAP_JET);
            energy_map.copyTo(im_to_draw, mask);
        }

        // copy child to father
        im_to_draw.copyTo(result(roi));

        // change child img location
        if (lvl & 0x1) {
            l += frame->width(lvl) + interval;
        } else {
            u += frame->height(lvl) + interval;
        }
        LOG(INFO) << '\n'
                  << "lvl : " << lvl << " point cloud size : " << point_cloud_pyramid_->operator[](lvl).size()
                  << "  final energy per pixel :" << energy_vec[energy_vec.size() - lvl - 1];
    }

    LOG(INFO) << "tracking use time : " << time_cost / 1000.f << "ms";

    cv::imshow("coarse to fine tracking", result);
    cv::waitKey(0);
}

}  // namespace mpl