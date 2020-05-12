#include "tracking_optimizer.h"
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
// todo move to tracking for fine controll
float TrackingOptimizer::solve(const int iterations) {
    // build Hx = b problem
    build_problem();
    // set init lamda value = tao* max{H(i,i)};
    float scale_ratio = 10e-6;
    float h_max_diag = max_diag(H);
    lamda = scale_ratio * h_max_diag;
    lamda = 5;
    int iteration_cnt = 0;
    bool is_converge = false;
    int successive_failed_times = 0;
    while (!is_converge && iteration_cnt++ < iterations) {
        // solve (H + lamda*I)*delta_x = b
        LOG(INFO) << " iterations begin : " << iteration_cnt
                  << ", lamda : " << lamda << "------------------------";
        auto H_tmp = H;
        H_tmp *= (1 + lamda) * Mat88::Identity();
        Vec8 delta_x = H_tmp.ldlt().solve(b);

        // we evaluate model decent before scaling
        // we salce our delta_x,because we sclae out jacobian in build_problem()
        scaling_delta_x(delta_x);
        LOG(INFO) << "delta_x : " << delta_x.transpose();
        // todo add new converge condition
        // * residual dosen't change
        // * gradient to small

        // pre-update the state for evaluation
        Sophus::SE3d old_pose(pose);
        AffineLight old_affine_light(affine_light);
        update(delta_x);

        // evaluate result
        float model_decent =
            1e-6 + 0.5f * delta_x.transpose() * (lamda * delta_x + b);
        float new_sum_residual = evaluate_sum_residual();
        float totoal_residual_decent = sum_residual - new_sum_residual;
        bool is_accept = (new_sum_residual < sum_residual) ? true : false;

        if (is_accept) {
            successive_failed_times = 0;
            LOG(INFO) << "step accepted @@@@@@@, curr sum residual : "
                      << new_sum_residual;
            lamda *= 0.5f;
            if (lamda < 0.00001) {
                lamda = 0.00001;
            }
            //! std::max(0.333, 1 - std::pow((2 * rho - 1), 3));
            lamda_failed_factor = 2;
            LOG(INFO) << "curr gradient" << b.norm();
            // evaluate J and r according to new parameters
            build_problem();
        } else {
            LOG(INFO) << "step rejected !!!!!!, curr sum residual: "
                      << sum_residual;
            LOG(INFO) << "curr gradient" << b.norm();
            // if not accept we roll back our state
            pose = old_pose;
            affine_light = old_affine_light;
            lamda *= lamda_failed_factor;
            if (lamda > 1e10) {
                is_converge = true;
            }
            lamda_failed_factor *= 2;
            if (++successive_failed_times > 18) {
                is_converge = true;
            }
        }
        LOG(INFO) << " iteration end -----------------------------------";
    }
    return sum_residual;
}

inline void TrackingOptimizer::update(const Vec8 delta_x) {
    Eigen::Map<const Sophus::Vector6d> delta_se3(delta_x.data());
    pose = Sophus::SE3d::exp(delta_se3) * pose;
    affine_light.update(delta_x(6), delta_x(7));
}

void TrackingOptimizer::build_problem() {
    H.setZero();
    b.setZero();
    sum_residual = 0.0f;
    for (auto& voxel : (*point_cloud_pyramid)[0]) {
        Eigen::Vector3f P = map(pose, voxel.position);  // point in curr frame
        Eigen::Vector2f hit_pixel = to_track_frame->project(P);
        // todo in frame function
        if (!image_pyramid->is_in_image(0, hit_pixel(0), hit_pixel(1))) {
            continue;
        }
        float intensity_on_image =
            (*image_pyramid)(0, hit_pixel(0), hit_pixel(1));
        // compute residual
        r_tmp = calc_residual(intensity_on_image, voxel.intensity);
        sum_residual += std::pow(r_tmp, 2);
        // compute jacobian
        float inv_z = (1.0f / P(2));
        float inv_zz = inv_z * inv_z;
        float dx = image_pyramid->dx(0, hit_pixel(0), hit_pixel(1));
        float dy = image_pyramid->dy(0, hit_pixel(0), hit_pixel(1));

        float fx_dx = cam->fx[0] * dx;
        float fy_dy = cam->fy[0] * dy;
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

        accumulate_H_b();
    }
    b_for_evaluation = b;
    scaling_H_b();

    LOG(INFO) << "HESSIAN : " << '\n' << H;
}

void TrackingOptimizer::accumulate_H_b() {
    H += J_tmp.transpose() * J_tmp;
    b += -J_tmp.transpose() * r_tmp;
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

inline Eigen::Vector3f TrackingOptimizer::map(Sophus::SE3d pose,
                                              Eigen::Vector3f point) const {
    return pose.cast<float>().rotationMatrix() * point +
           pose.cast<float>().translation();
}

inline float TrackingOptimizer::max_diag(const Mat88& H) const {
    float max = 0;
    for (int i = 0; i < H.rows(); ++i) {
        if (H(i, i) > max) {
            max = H(i, i);
        }
    }
    return max;
}

float TrackingOptimizer::evaluate_sum_residual() const {
    float sum = 0.f;
    for (auto& voxel : (*point_cloud_pyramid)[0]) {
        Eigen::Vector3f P = map(pose, voxel.position);  // point in curr frame
        Eigen::Vector2f hit_pixel = to_track_frame->project(P);
        if (!image_pyramid->is_in_image(0, hit_pixel(0), hit_pixel(1))) {
            continue;
        }
        float intensity_on_image =
            (*image_pyramid)(0, hit_pixel(0), hit_pixel(1));
        sum += std::pow(calc_residual(intensity_on_image, voxel.intensity), 2);
    }
    return sum;
}

inline float TrackingOptimizer::calc_residual(
    const float curr_intensity, const float point_cloud_intensity) const {
    return curr_intensity - exp(affine_light.alpha()) * point_cloud_intensity -
           affine_light.beta();
}

}  // namespace mpl
