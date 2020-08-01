#pragma once
#include "affine_light.h"
#include "frame.h"
#include "point_cloud_pyramid.h"
#include "sophus/se3.hpp"

namespace mpl {
typedef Eigen::Matrix<double, 8, 8> Mat88;
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef Eigen::Matrix<double, 1, 8> RowVec8;

class TrackingOptimizer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TrackingOptimizer() = default;
    void init(const Sophus::SE3f init_pose, const AffineLight init_rel_affL,
              PointCloudPyramid::ptr ref_point_cloud_prymid, Frame::ptr to_track_frame);

    float solve(const int iterations, const float lamda_init, const float lamda_min, const int l1_turacation,
                const bool remove_outlier);
    Sophus::SE3f getT() const;
    AffineLight getAffineLight() const;
    void set_lvl(const int lvl);

   private:
    // accumulate to H and b
    void update(const Vec8 delta_x);
    void build_problem();
    void remove_outlier(const bool remove_outlier);
    void accumulate_H_b(float roboust_weight);
    Mat88 get_damped_hessian();
    void scaling_H_b();
    void scaling_delta_x(Vec8& delta_x);
    float calc_residual(const float curr_intensity, const float point_cloud_intensity) const;
    float calc_sum_weighted_squared_residual(bool return_average_residual) const;

    void assign_result_for_visualization();
    void remove_outlier();

    float calc_huber_weight(const float residual) const;
    float calc_huber_weigted_redidual(const float huber_weight, const float residual) const;
    Eigen::Vector3f map(Sophus::SE3d pose,
                        Eigen::Vector3f point) const;  // map point to curr coordinate system

    int curr_lvl;

    Sophus::SE3d pose;         // T_curr_ref   pose from reference to curr frame
    AffineLight affine_light;  // Aff_curr_map, aff light map to curr frame
    PointCloudPyramid::ptr point_cloud_pyramid;
    Frame::ptr to_track_frame;

    CamData* cam;
    Config* config;

    Mat88 H = Mat88::Zero();  // H = J.trans()*J
    Vec8 b = Vec8::Zero();    // b = - J.trans()*r_vec

    float sum_weighted_squared_residual = 0.0f;
    double r_tmp;                     // r for every measurement
    RowVec8 J_tmp = RowVec8::Zero();  // J for every measurement

    float lamda;
    float huber_radius;
    float lamda_failed_penalize_factor = 2.0f;
};

}  // namespace mpl