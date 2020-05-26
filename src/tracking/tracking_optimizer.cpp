#include "tracking_optimizer.h"
#include <cmath>
#include <glog/logging.h>

namespace mpl {
void TrackingOptimizer::init(const Sophus::SE3f init_pose,
                             const AffineLight init_rel_affL,
                             PointCloudPyramid::ptr ref_point_cloud_prymid,
                             Frame::ptr to_track_frame) {
    this->pose = init_pose.cast<double>();
    this->affine_light = init_rel_affL;

    assert(ref_point_cloud_prymid != nullptr);
    assert(to_track_frame != nullptr);
    this->to_track_frame = to_track_frame;
    this->point_cloud_pyramid = ref_point_cloud_prymid;
    image_pyramid = to_track_frame->getImagePyramid();

    cam = &CamData::getInstance();
    config = &Config::getInstance();
}

float TrackingOptimizer::solve(const int iterations, const float lamda_init,
                               const float lamda_min, const int huber_radius_) {
    // build Hx = b problem
    this->huber_radius = huber_radius_;
    lamda = lamda_init;
    lamda_failed_penalize_factor = config->OPTIMIZATION_LAMDA_FAILED_PENALIZE;
    sum_weighted_squared_residual = 0;
    build_problem();

    int iteration_cnt = 0;
    bool is_converge = false;
    while (!is_converge && iteration_cnt++ < iterations) {
        // LOG(INFO) << " curr lvl : " << curr_lvl
        //           << " iterations begin : " << iteration_cnt
        //           << ", lamda : " << lamda << "------------------------";

        auto H_tmp = get_damped_hessian();
        Vec8 delta_x = H_tmp.ldlt().solve(
            b);  // H_tmp if for sure spd->llt,? faster than ldlt

        for (int i = 0; i < delta_x.rows(); ++i) {
            if (std::isinf(delta_x(i)) || std::isnan(delta_x(i))) {
                LOG(INFO) << "delta_x has inf or nan";
                return sum_weighted_squared_residual;
            }
        }

        scaling_delta_x(delta_x);

        // LOG(INFO) << "delta_x : " << delta_x.transpose();

        Sophus::SE3d old_pose(pose);
        AffineLight old_affine_light(affine_light);

        update(delta_x);
        // evaluate result
        float new_sum_residual = calc_sum_weighted_squared_residual();
        bool is_accept = (new_sum_residual < sum_weighted_squared_residual);

        if (is_accept) {
            // LOG(INFO) << "step accepted @@@@@@@, curr sum residual : "
            //           << new_sum_residual;
            // LOG(INFO) << "curr gradient" << b.norm();
            if (delta_x.norm() < 1e-5) {
                is_converge = true;
                continue;
            }
            lamda *= config->OPTIMIZATION_LAMDA_SUCCESS_PENALIZE;
            lamda = std::max(lamda, lamda_min);
            lamda_failed_penalize_factor =
                config->OPTIMIZATION_LAMDA_FAILED_PENALIZE;
            // evaluate J and r according to new parameters
            build_problem();
        } else {
            // LOG(INFO) << "step rejected !!!!!!, curr residual : "
            //          << sum_weighted_squared_residual;
            // LOG(INFO) << "curr gradient" << b.norm();
            // if not accept we roll back our state
            pose = old_pose;
            affine_light = old_affine_light;

            lamda *= lamda_failed_penalize_factor;
            lamda_failed_penalize_factor *= 2.5;

            if (lamda > config->OPTIMIZATION_LAMDA_MAX) {
                is_converge = true;
            }
        }
        // LOG(INFO) << " iteration end  -----------------------------------";
    }
    return sum_weighted_squared_residual;
}

inline void TrackingOptimizer::update(const Vec8 delta_x) {
    Eigen::Map<const Sophus::Vector6d> delta_se3(delta_x.data());
    pose = Sophus::SE3d::exp(delta_se3) * pose;
    affine_light.update(delta_x(6), delta_x(7));
}

inline float TrackingOptimizer::calc_huber_weight(const float residual) const {
    return (std::abs(residual) < huber_radius)
               ? 1
               : huber_radius / std::abs(residual);
}

inline float TrackingOptimizer::calc_huber_weigted_redidual(
    const float huber_weight, const float residual) const {
    return huber_weight * (2 - huber_weight) * residual * residual;
}

// todo only update b
void TrackingOptimizer::build_problem() {
    H.setZero();
    b.setZero();
    sum_weighted_squared_residual = 0.0f;
    int num_outlier = 0;
    for (auto& voxel : (*point_cloud_pyramid)[curr_lvl]) {
        Eigen::Vector3f P = map(pose, voxel.position);  // point in curr frame
        Eigen::Vector2f hit_pixel = to_track_frame->project(P, curr_lvl);
        // todo in frame function
        if (!image_pyramid->is_in_image(curr_lvl, hit_pixel(0), hit_pixel(1))) {
            continue;
        }
        float intensity_on_image =
            (*image_pyramid)(curr_lvl, hit_pixel(0), hit_pixel(1));
        // compute residual
        r_tmp = calc_residual(intensity_on_image, voxel.intensity);
        const float huber_weight = calc_huber_weight(r_tmp);
        if (std::abs(r_tmp) > huber_radius) {
            ++num_outlier;
            if (num_outlier / (float)config->PIXEL_SELECTION_NUM > 0.4) {
                huber_radius += 20;
                return build_problem();
            }
        }
        sum_weighted_squared_residual +=
            calc_huber_weigted_redidual(huber_weight, r_tmp);

        // compute jacobian
        float inv_z = (1.0f / P(2));
        float inv_zz = inv_z * inv_z;
        float dx = image_pyramid->dx(curr_lvl, hit_pixel(0), hit_pixel(1));
        float dy = image_pyramid->dy(curr_lvl, hit_pixel(0), hit_pixel(1));

        float fx_dx = cam->fx[curr_lvl] * dx;
        float fy_dy = cam->fy[curr_lvl] * dy;
        float p1_inv_zz = P(1) * inv_zz;

        // dIdp
        // todo overloader image_pyramid and do subprecision

        J_tmp(0) = fx_dx * inv_z;
        J_tmp(1) = fy_dy * inv_z;
        J_tmp(2) = -(fx_dx * P(0) + fy_dy * P(1)) * inv_zz;

        J_tmp(3) =
            -(fx_dx * P(0) * p1_inv_zz + fy_dy * (1.0f + P(1) * p1_inv_zz));
        J_tmp(4) =
            fx_dx * (1.0f + P(0) * P(0) * inv_zz) + fy_dy * P(0) * p1_inv_zz;
        J_tmp(5) = -fx_dx * P(1) * inv_z + fy_dy * P(0) * inv_z;

        J_tmp(6) = -voxel.intensity * exp(affine_light.alpha());
        J_tmp(7) = -1;
        accumulate_H_b(huber_weight);
    }
    // std::cerr << " huber radius :" << huber_radius
    //          << " num outlier :" << num_outlier << '\n';

    // LOG(INFO) << " H : " << '\n' << H;
    scaling_H_b();

    // LOG(INFO) << "!!!!new  H : " << '\n' << H;
}

void TrackingOptimizer::accumulate_H_b(const float robust_weight) {
    H.noalias() += robust_weight * J_tmp.transpose() * J_tmp;
    b += -robust_weight * J_tmp.transpose() * r_tmp;
}

Mat88 TrackingOptimizer::get_damped_hessian() {
    Mat88 H_tmp(H);
    H_tmp.diagonal() *= (1.0 + lamda);
    return H_tmp;
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

inline Eigen::Vector3f TrackingOptimizer::map(Sophus::SE3d pose,
                                              Eigen::Vector3f point) const {
    return pose.cast<float>().rotationMatrix() * point +
           pose.cast<float>().translation();
}

float TrackingOptimizer::calc_sum_weighted_squared_residual() const {
    float sum = 0.f;
    for (auto& voxel : (*point_cloud_pyramid)[curr_lvl]) {
        Eigen::Vector3f P = map(pose, voxel.position);  // point in curr frame
        Eigen::Vector2f hit_pixel = to_track_frame->project(P, curr_lvl);
        if (!image_pyramid->is_in_image(curr_lvl, hit_pixel(0), hit_pixel(1))) {
            continue;
        }
        float intensity_on_image =
            (*image_pyramid)(curr_lvl, hit_pixel(0), hit_pixel(1));
        float r = calc_residual(intensity_on_image, voxel.intensity);
        sum += calc_huber_weigted_redidual(calc_huber_weight(r), r);
    }
    return sum;
}

inline float TrackingOptimizer::calc_residual(
    const float curr_intensity, const float point_cloud_intensity) const {
    return curr_intensity - exp(affine_light.alpha()) * point_cloud_intensity -
           affine_light.beta();
}

}  // namespace mpl
