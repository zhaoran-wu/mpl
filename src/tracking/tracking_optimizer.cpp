#include "tracking_optimizer.h"
#include "candidate_manager.h"
#include <cmath>
#include <glog/logging.h>

namespace mpl {
void TrackingOptimizer::init(const Sophus::SE3f init_pose, const AffineLight init_rel_affL,
                             PointCloudPyramid::ptr ref_point_cloud_prymid, Frame::ptr to_track_frame) {
    this->pose = init_pose.cast<double>();
    this->affine_light = init_rel_affL;

    assert(ref_point_cloud_prymid != nullptr);
    assert(to_track_frame != nullptr);

    this->to_track_frame = to_track_frame;
    this->point_cloud_pyramid = ref_point_cloud_prymid;

    cam = &CamData::getInstance();
    config = &Config::getInstance();
}

float TrackingOptimizer::solve(const int iterations, const float lamda_init, const float lamda_min,
                               const int huber_radius_, const bool remove_outlier_) {
    // build Hx = b problem
    this->huber_radius = huber_radius_;
    lamda = lamda_init;
    lamda_failed_penalize_factor = config->OPTIMIZATION_LAMDA_FAILED_PENALIZE;
    sum_weighted_squared_residual = 0.0f;

    build_problem();

    int iteration_cnt = 0;
    bool is_converge = false;
    while (!is_converge && iteration_cnt++ < iterations) {
        // LOG(INFO) << " curr lvl : " << curr_lvl << " iterations begin : " << iteration_cnt << ", lamda : " << lamda
        //          << "------------------------";

        auto H_damped = get_damped_hessian();
        Vec8 delta_x = H_damped.ldlt().solve(b);

        for (int i = 0; i < delta_x.rows(); ++i) {
            if (std::isinf(delta_x(i)) || std::isnan(delta_x(i))) {
                LOG(INFO) << "delta_x has inf or nan";
                return sum_weighted_squared_residual;
            }
        }

        scaling_delta_x(delta_x);

        // LOG(INFO) << "delta_x : " << delta_x.transpose();

        Sophus::SE3d pose_old(pose);
        AffineLight affine_light_old(affine_light);

        update(delta_x);
        // evaluate result
        float sum_at_new_states = calc_sum_weighted_squared_residual(false);
        // affine light should not change to much
        bool is_accept = (sum_at_new_states < sum_weighted_squared_residual);  //&& std::abs(delta_x(6)) < 0.1f);

        if (is_accept) {
            // LOG(INFO) << "step accepted @@@@@@@, curr sum residual : " << sum_at_new_states;
            // LOG(INFO) << "curr gradient" << b.norm();
            if (delta_x.norm() < config->OPTIMIZATION_STEP_MIN) {
                is_converge = true;
                continue;
            }
            lamda *= config->OPTIMIZATION_LAMDA_SUCCESS_PENALIZE;
            lamda = std::max(lamda, lamda_min);
            lamda_failed_penalize_factor = config->OPTIMIZATION_LAMDA_FAILED_PENALIZE;
            // evaluate J and r according to new parameters
            build_problem();
        } else {
            // LOG(INFO) << "step rejected !!!!!!, curr residual : " << sum_at_new_states
            //          << "  old residual : " << sum_weighted_squared_residual;
            // LOG(INFO) << "curr gradient" << b.norm();
            // if not accept we roll back our state
            pose = pose_old;
            affine_light = affine_light_old;

            lamda *= lamda_failed_penalize_factor;
            lamda_failed_penalize_factor *= 2.5;

            is_converge = (lamda > config->OPTIMIZATION_LAMDA_MAX);
        }
        // LOG(INFO) << " iteration end  -----------------------------------";
    }

    // assign voxel final energy and hit position
    if (remove_outlier_) remove_outlier();
    assign_result_for_visualization();

    return calc_sum_weighted_squared_residual(true);
}

inline void TrackingOptimizer::update(const Vec8 delta_x) {
    Eigen::Map<const Sophus::Vector6d> delta_se3(delta_x.data());
    pose = Sophus::SE3d::exp(delta_se3) * pose;
    affine_light.update(delta_x(6), delta_x(7));
}

void TrackingOptimizer::assign_result_for_visualization() {
    if (curr_lvl != 0) return;

    for (auto& voxel : (*point_cloud_pyramid)[0]) {
        // check if is out of image
        Eigen::Vector3f P = map(pose, voxel.position);  // point in curr frame
        Eigen::Vector2f hit_pixel = to_track_frame->project(P);

        if (!to_track_frame->is_in_image(hit_pixel)) {
            voxel.vis_data.visible_for_newst_frame = false;
            continue;
        } else {
            voxel.vis_data.hit_pixel_in_newst_frame = hit_pixel;
            voxel.vis_data.depth_in_newst_frame = P(2);
        }
        // check if outlier
        if (voxel.can->status == CandidateStatus::OUTLIER) {
            voxel.vis_data.is_outlier = true;
        }

        // set tracking energy
        float intensity_on_image = to_track_frame->at(hit_pixel);
        float r = calc_residual(intensity_on_image, voxel.intensity);

        voxel.vis_data.last_tracking_energy = r * r;  // without any weight
    }
}

void TrackingOptimizer::remove_outlier() {
    for (auto& voxel : (*point_cloud_pyramid)[curr_lvl]) {
        Eigen::Vector3f P = map(pose, voxel.position);  // point in curr frame
        Eigen::Vector2f hit_pixel = to_track_frame->project(P, curr_lvl);
        // out of image
        if (!to_track_frame->is_in_image(hit_pixel, curr_lvl)) {
            // outlier will not join backend optimization,but oob will;
            voxel.can->status = (voxel.can->good_track_cnt > 0) ? CandidateStatus::OOB : CandidateStatus::OUTLIER;
            continue;
        }

        float intensity_on_image = to_track_frame->at(hit_pixel, curr_lvl);
        float r = calc_residual(intensity_on_image, voxel.intensity);

        // remove outlier according to tracking result
        float mag2 = to_track_frame->mag_squared(hit_pixel, curr_lvl);
        if ((r * r > 5 && r * r * std::sqrt(mag2) > 1000) || (mag2 == 0)) {
            // std::cout << "last tracking energy " << r * r << " mag"
            //          << std::sqrt(to_track_frame->mag_squared(hit_pixel, curr_lvl))
            //          << "r*r*mag = " << r * r * std::sqrt(to_track_frame->mag_squared(hit_pixel, curr_lvl)) << '\n';
            voxel.can->bad_track_cnt++;
            if (voxel.can->good_track_cnt == 0) {
                voxel.can->status = CandidateStatus::OUTLIER;

            } else {
                voxel.can->status = CandidateStatus::OOB;
            }
        } else {
            voxel.can->good_track_cnt++;
        }
    }
}

inline float TrackingOptimizer::calc_huber_weight(const float residual) const {
    return (std::abs(residual) < huber_radius) ? 1.0f : huber_radius / std::abs(residual);
}

inline float TrackingOptimizer::calc_huber_weigted_redidual(const float huber_weight, const float residual) const {
    return huber_weight * (2.0f - huber_weight) * residual * residual;
}

// todo only update b
void TrackingOptimizer::build_problem() {
    H.setZero();
    b.setZero();
    sum_weighted_squared_residual = 0.0f;
    int num_outlier = 0;
    int cnt = 0;
    for (auto& voxel : (*point_cloud_pyramid)[curr_lvl]) {
        if (voxel.can->status != CandidateStatus::ACTIVE) continue;

        const Eigen::Vector3f P = map(pose, voxel.position);  // point in curr frame
        const Eigen::Vector2f hit_pixel = to_track_frame->project(P, curr_lvl);

        if (!to_track_frame->is_in_image(hit_pixel, curr_lvl)) {
            continue;
        }

        const float intensity_on_image = to_track_frame->at(hit_pixel, curr_lvl);

        // compute residual
        r_tmp = calc_residual(intensity_on_image, voxel.intensity);
        const float huber_weight = calc_huber_weight(r_tmp);
        if (std::abs(r_tmp) > huber_radius) {
            ++num_outlier;
        }

        // compute jacobian
        const float inv_z = (1.0f / P(2));
        const float inv_zz = inv_z * inv_z;
        const float dx = to_track_frame->dx(hit_pixel, curr_lvl);
        const float dy = to_track_frame->dy(hit_pixel, curr_lvl);
        const float mag2 = to_track_frame->mag_squared(hit_pixel, curr_lvl);

        const float heuristic_const = 2500.0f;
        const float gradient_weight = sqrt(heuristic_const / (heuristic_const + mag2));

        // if (std::abs(dx + dy) < 1e-2) {
        //    // std::cout << "dx + dy: " << dx + dy << std::endl;
        //    continue;  // igonore point with very weak gradient,that can not contribute to the result
        //}

        const float fx_dx = cam->fx[curr_lvl] * dx;
        const float fy_dy = cam->fy[curr_lvl] * dy;
        const float p1_inv_zz = P(1) * inv_zz;

        // dIdp
        // todo overloader image_pyramid and do subprecision

        J_tmp(0) = fx_dx * inv_z;
        J_tmp(1) = fy_dy * inv_z;
        J_tmp(2) = -(fx_dx * P(0) + fy_dy * P(1)) * inv_zz;

        J_tmp(3) = -(fx_dx * P(0) * p1_inv_zz + fy_dy * (1.0f + P(1) * p1_inv_zz));
        J_tmp(4) = fx_dx * (1.0f + P(0) * P(0) * inv_zz) + fy_dy * P(0) * p1_inv_zz;
        J_tmp(5) = -fx_dx * P(1) * inv_z + fy_dy * P(0) * inv_z;

        J_tmp(6) = -voxel.intensity * exp(affine_light.alpha());
        J_tmp(7) = -1.0;

        float total_weight = gradient_weight * voxel.aligenment_weight;

        J_tmp *= total_weight;

        sum_weighted_squared_residual += std::pow(total_weight, 2) * calc_huber_weigted_redidual(huber_weight, r_tmp);
        cnt++;
        // LOG(INFO) << "J :" << '\n' << J_tmp;
        // LOG(INFO) << "r :" << sum_weighted_squared_residual;

        accumulate_H_b(huber_weight);
    }
    // std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@ finial used pcp cnt !!!!!!!!!!(in build problem): " << cnt
    //          << "average energy : " << std::sqrt(sum_weighted_squared_residual / cnt) << '\n';
    // std::cerr << " huber radius :" << huber_radius << " num outlier :" << num_outlier << '\n';
    // LOG(INFO) << " H : " << '\n' << H;
    scaling_H_b();

    // LOG(INFO) << "!!!!new  H : " << '\n' << H;
}

void TrackingOptimizer::accumulate_H_b(const float robust_weight) {
    H.noalias() += robust_weight * J_tmp.transpose() * J_tmp;
    b += -robust_weight * J_tmp.transpose() * r_tmp;
}

Mat88 TrackingOptimizer::get_damped_hessian() {
    Mat88 H_damped(H);
    H_damped.diagonal() *= (1.0 + lamda);  //? + 1.0f will crash
    return H_damped;
}
/**
 * @brief  [A B C]' * [a,b,c] = [ Aa, Ab ,Ac]
 *                              [Ba, Bb ,Bc]
 *                              [Ca, Cb ,Cc]
 */
void TrackingOptimizer::scaling_H_b() {
    H.block<8, 3>(0, 0) *= config->OPTIMIZATION_TRANS_SCALE;
    H.block<8, 3>(0, 3) *= config->OPTIMIZATION_ROTATION_SCALE;
    H.block<8, 1>(0, 6) *= config->OPTIMIZATION_ALPHA_SCALE;
    H.block<8, 1>(0, 7) *= config->OPTIMIZATION_BETA_SCALE;

    H.block<3, 8>(0, 0) *= config->OPTIMIZATION_TRANS_SCALE;
    H.block<3, 8>(3, 0) *= config->OPTIMIZATION_ROTATION_SCALE;
    H.block<1, 8>(6, 0) *= config->OPTIMIZATION_ALPHA_SCALE;
    H.block<1, 8>(7, 0) *= config->OPTIMIZATION_BETA_SCALE;

    b.segment<3>(0) *= config->OPTIMIZATION_TRANS_SCALE;
    b.segment<3>(3) *= config->OPTIMIZATION_ROTATION_SCALE;
    b(6) *= config->OPTIMIZATION_ALPHA_SCALE;
    b(7) *= config->OPTIMIZATION_BETA_SCALE;
}
void TrackingOptimizer::scaling_delta_x(Vec8& delta_x) {
    delta_x.segment<3>(0) *= config->OPTIMIZATION_TRANS_SCALE;
    delta_x.segment<3>(3) *= config->OPTIMIZATION_ROTATION_SCALE;
    delta_x(6) *= config->OPTIMIZATION_ALPHA_SCALE;
    delta_x(7) *= config->OPTIMIZATION_BETA_SCALE;
}

AffineLight TrackingOptimizer::getAffineLight() const {
    return affine_light;
}

Sophus::SE3f TrackingOptimizer::getT() const {
    return pose.cast<float>();
}

void TrackingOptimizer::set_lvl(const int lvl) {
    curr_lvl = lvl;
}

inline Eigen::Vector3f TrackingOptimizer::map(Sophus::SE3d pose, Eigen::Vector3f point) const {
    return pose.cast<float>().rotationMatrix() * point + pose.cast<float>().translation();
}

float TrackingOptimizer::calc_sum_weighted_squared_residual(bool return_average_residual) const {
    float sum = 0.f;
    int cnt = 0;
    for (auto& voxel : (*point_cloud_pyramid)[curr_lvl]) {
        if (voxel.can->status != CandidateStatus::ACTIVE) continue;
        Eigen::Vector3f P = map(pose, voxel.position);  // point in curr frame
        Eigen::Vector2f hit_pixel = to_track_frame->project(P, curr_lvl);
        if (!to_track_frame->is_in_image(hit_pixel, curr_lvl)) {
            continue;
        }
        float intensity_on_image = to_track_frame->at(hit_pixel, curr_lvl);
        float r = calc_residual(intensity_on_image, voxel.intensity);
        cnt++;
        const float mag2 = to_track_frame->mag_squared(hit_pixel, curr_lvl);

        const float heuristic_const = 2500.0f;
        const float gradient_weight = sqrt(heuristic_const / (heuristic_const + mag2));

        sum += std::pow(voxel.aligenment_weight * gradient_weight, 2) *
               calc_huber_weigted_redidual(calc_huber_weight(r), r);
    }

    // std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@ finial used pcp cnt :   " << cnt
    //          << "   average energy :" << std::sqrt(sum / cnt)
    //          << " total pcp size : " << (*point_cloud_pyramid)[curr_lvl].size() << '\n';

    float result = sum;
    if (return_average_residual) {
        result /= cnt;
        result = std::sqrt(result);
    }

    return result;
}

inline float TrackingOptimizer::calc_residual(const float curr_intensity, const float point_cloud_intensity) const {
    return calc_light_diff(point_cloud_intensity, curr_intensity, affine_light);
}

}  // namespace mpl
