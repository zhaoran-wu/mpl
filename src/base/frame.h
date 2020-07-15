#pragma once
#include "affine_light.h"
#include "ceres/FrameParameterBlock.h"
#include "image_pyramid.h"
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
namespace mpl {
class Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Frame> ptr;
    Frame(const uchar* const row_image_data);
    static Frame::ptr create(cv::Mat image);
    ImagePyramid::ptr getImagePyramid();
    // T_w_c
    void set_pose(const Sophus::SE3f& pose);
    // Aff_w_c
    void set_aff_light(const int alpha, const int beta);
    void set_state(const Sophus::SE3f& T_curr_lastKF,
                   const AffineLight& aff_light_curr_lastKF);
    Sophus::SE3f get_T_curr_lastKF();
    AffineLight get_aff_curr_lastKF();

    void set_ref_frame(const Frame::ptr& frame);

    // return T_w_c
    template <typename T>
    T get_pose() const;
    // return global aff light
    AffineLight get_aff_light() const;

    // unproject a point in to 3D refer in current frame
    // coordinate system
    Eigen::Vector3f unproject(const Eigen::Vector2i& pixel, const float depth,
                              const int lvl = 0) const;
    // poject a point in current frame coordinate system to image at lvl
    Eigen::Vector2f project(const Eigen::Vector3f& point, int lvl = 0) const;

    bool is_in_image(const int lvl, const float u, const float v) const;

    // get the frame, which used to tracking curr frame;
    Frame::ptr get_ref_frame();

    // get ceres param block
    const std::unique_ptr<FrameParameterBlock>& get_frame_block() const;

    int get_id();

    void merge_optimization_result();

   private:
    ImagePyramid::ptr pyramid;
    // relative info
    Frame::ptr last_KF_ref;
    Sophus::SE3f T_curr_lastKF =
        Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 0, 0));
    AffineLight aff_light_curr_lastKF = AffineLight(0, 0);
    // global info
    Sophus::SE3f pose = Sophus::SE3f::transZ(0.0f);  // T_c_w;
    AffineLight affine_light = AffineLight(0, 0);
    CamData* cam;
    int id;

    static int id_cnt;
    // ceres optimization params
    std::unique_ptr<FrameParameterBlock> frame_block;
};

inline int Frame::get_id() {
    return id;
}
inline Eigen::Vector3f Frame::unproject(const Eigen::Vector2i& pixel,
                                        const float depth,
                                        const int lvl) const {
    return depth * cam->K_inv[lvl] *
           (pixel.cast<float>() + Eigen::Vector2f(0.5f, 0.5f)).homogeneous();
}

inline Eigen::Vector2f Frame::project(const Eigen::Vector3f& point,
                                      const int lvl) const {
    return (cam->K[lvl] * point).hnormalized();
}

inline void Frame::set_pose(const Sophus::SE3f& pose) {
    this->pose = pose;
}
inline void Frame::set_aff_light(const int alpha, const int beta) {
    this->affine_light = AffineLight(alpha, beta);
}

inline void Frame::set_state(
    // relative info
    const Sophus::SE3f& T_curr_lastKF,
    const AffineLight& aff_light_curr_lastKF) {
    this->T_curr_lastKF = T_curr_lastKF;
    this->aff_light_curr_lastKF = aff_light_curr_lastKF;

    // inital global info
    this->pose = this->last_KF_ref->pose * T_curr_lastKF.inverse();
    this->affine_light = AffineLight::calc_dst_global_aff(
        this->last_KF_ref->affine_light, aff_light_curr_lastKF);
}

// return T_w_c

template <typename T>
inline T Frame::get_pose() const {
    return T(pose.matrix());
}

// return global aff light
inline AffineLight Frame::get_aff_light() const {
    return affine_light;
}

inline void Frame::set_ref_frame(const Frame::ptr& frame) {
    this->last_KF_ref = frame;
}

inline bool Frame::is_in_image(const int lvl, const float u,
                               const float v) const {
    return pyramid->is_in_image(lvl, u, v);
}

inline Eigen::Vector2f unproject_trans_project(const float d_inv,
                                               const Eigen::Vector2i& src_pixel,
                                               const Frame::ptr src_frame,
                                               const Frame::ptr target_frame,

                                               int lvl = 0) {
    Eigen::Vector3f src_point =
        src_frame->unproject(src_pixel, 1.f / d_inv, lvl);
    Eigen::Isometry3f T_target_src =
        target_frame->get_pose<Eigen::Isometry3f>().inverse() *
        src_frame->get_pose<Eigen::Isometry3f>();
    return target_frame->project(T_target_src * src_point, lvl);
}

Sophus::SE3f get_src_to_dst_transform(const Frame::ptr src,
                                      const Frame::ptr dst);

AffineLight get_src_to_dst_aff_light(const Frame::ptr src,
                                     const Frame::ptr dst);

inline const std::unique_ptr<FrameParameterBlock>& Frame::get_frame_block()
    const {
    return frame_block;
}

inline void Frame::merge_optimization_result() {
    // std::cout << "pose before : " << '\n' << this->pose.matrix() << '\n';
    this->pose = frame_block->getPose().inverse().cast<float>();
    this->affine_light = frame_block->getAffineLight();

    // std::cout << "pose after : " << '\n' << this->pose.matrix() << '\n';
}

}  // namespace mpl