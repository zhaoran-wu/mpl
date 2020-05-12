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
    TrackingOptimizer() = default;
    void init(const Sophus::SE3f init_pose, const AffineLight init_rel_affL,
              PointCloudPyramid::ptr ref_point_cloud_prymid,
              Frame::ptr to_track_frame);

    float solve(const int iterations);
    Sophus::SE3f getT() const;
    AffineLight getAffineLight() const;

   private:
    // accumulate to H and b
    void update(const Vec8 delta_x);
    void roll_back(const Vec8 delta_x);
    void build_problem();
    void update_H_b();
    void scaling_H_b();
    void scaling_delta_x(Vec8& delta_x);
    float max_diag(const Mat88& H) const;
    float calc_residual(const float curr_intensity,
                        const float point_cloud_intensity) const;
    float evaluate_sum_residual() const;

    Eigen::Vector3f map(
        Sophus::SE3d pose,
        Eigen::Vector3f point) const;  // map point to curr coordinate system

    Sophus::SE3d pose;  // pose from reference to curr frame
    AffineLight affine_light;
    PointCloudPyramid::ptr point_cloud_pyramid;
    Frame::ptr to_track_frame;
    ImagePyramid::ptr image_pyramid;

    CamData* cam;
    Config* config;

    Mat88 H;  // H = J.trans()*J
    Vec8 b;   // b = - J.trans()*r_vec

    double sum_residual;
    double r_tmp;   // r for every measurement
    RowVec8 J_tmp;  // J for every measurement

    float lamda;
    float v = 1e-6;
};

}  // namespace mpl