#pragma once
#include "affine_light.h"
#include "ceres/FrameParameterBlock.h"
#include "image_pyramid.h"
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
namespace mpl {
/**
 * @brief frame hold it's pose, affine param, image pyramid
 *
 */
class Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   public:
    typedef std::shared_ptr<Frame> ptr;

    // construct with image data
    Frame(const uchar* const row_image_data);
    static Frame::ptr create(cv::Mat image);

    // set member variable
    void set_pose(const Sophus::SE3f& T_w_c);             // global T_w_c
    void set_aff_light(const int alpha, const int beta);  // global aff_w_c
    void set_tracking_result(const Sophus::SE3f& T_curr_refKF, const AffineLight& aff_light_curr_refKF);
    void set_ref_frame(const Frame::ptr& frame);

    // get member variable
    ImagePyramid::ptr get_image_pyramid() const;
    Sophus::SE3f get_pose() const;      // return T_w_c
    AffineLight get_aff_light() const;  // return global aff light
    Frame::ptr get_ref_frame();         // get the frame, which used to tracking curr frame;
    int get_id() const;
    const std::unique_ptr<FrameParameterBlock>& get_frame_block() const;  // get ceres param block

    // access image pyramid data
    // get pixel value at lvl
    template <typename T>
    float at(const T u, const T v, const int lvl = 0) const;
    template <typename T>
    float dy(const T u, const T v, const int lvl = 0) const;
    template <typename T>
    float dx(const T u, const T v, const int lvl = 0) const;
    template <typename T>
    float mag_squared(const T u, const T v, const int lvl = 0) const;
    template <typename T>
    float at(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl = 0) const;
    template <typename T>
    float dy(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl = 0) const;
    template <typename T>
    float dx(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl = 0) const;
    template <typename T>
    float mag_squared(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl = 0) const;

    // unproject a pixel(in pixel coordinate system) in to p3d in current frame coordinate system
    Eigen::Vector3f unproject(const Eigen::Vector2i& pixel, const float inv_d, const int lvl = 0) const;
    // poject a p3d in current frame coordinate system to pixel coordinate systems
    Eigen::Vector2f project(const Eigen::Vector3f& point, const int lvl = 0) const;
    // check if a projection is in the image at lvl
    bool is_in_image(const float u, const float v, const int lvl = 0) const;
    template <typename T>
    bool is_in_image(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl = 0) const;

    // interface to access image pyramid value

    // interface for ceres callback
    void merge_optimization_result();

   private:
    ImagePyramid::ptr pyramid = nullptr;
    Frame::ptr refKF = nullptr;

    // global info
    Sophus::SE3f T_w_c = Sophus::SE3f::transZ(0.0f);  // T_w_c;
    AffineLight affine_light = AffineLight(0, 0);     // aff_w_c
    int id;

    // ceres optimization params
    std::unique_ptr<FrameParameterBlock> frame_block;

    CamData* cam;
    static int id_cnt;
};

inline int Frame::get_id() const {
    return id;
}

inline Eigen::Vector3f Frame::unproject(const Eigen::Vector2i& pixel, const float inv_d, const int lvl) const {
    float z = 1.0f / inv_d;
    return Eigen::Vector3f((pixel(0) - cam->cx[lvl]) * z / cam->fx[lvl], (pixel(1) - cam->cy[lvl]) * z / cam->fy[lvl],
                           z);
}

inline Eigen::Vector2f Frame::project(const Eigen::Vector3f& point, const int lvl) const {
    return Eigen::Vector2f(cam->fx[lvl] * (point(0) / point(2)) + cam->cx[lvl],
                           cam->fy[lvl] * (point(1) / point(2)) + cam->cy[lvl]);
}

inline void Frame::set_pose(const Sophus::SE3f& T_w_c) {
    this->T_w_c = T_w_c;
}
inline void Frame::set_aff_light(const int alpha, const int beta) {
    this->affine_light = AffineLight(alpha, beta);
}

inline void Frame::set_tracking_result(const Sophus::SE3f& T_curr_refKF, const AffineLight& aff_light_curr_refKF) {
    // inital global info
    this->T_w_c = this->refKF->T_w_c * T_curr_refKF.inverse();
    this->affine_light = calc_dst_global_aff(this->refKF->affine_light, aff_light_curr_refKF);
}

inline Sophus::SE3f Frame::get_pose() const {
    return T_w_c;
}

inline AffineLight Frame::get_aff_light() const {
    return affine_light;
}

inline ImagePyramid::ptr Frame::get_image_pyramid() const {
    return this->pyramid;
}

inline void Frame::set_ref_frame(const Frame::ptr& frame) {
    this->refKF = frame;
}

inline const std::unique_ptr<FrameParameterBlock>& Frame::get_frame_block() const {
    return frame_block;
}

inline void Frame::merge_optimization_result() {
    this->T_w_c = frame_block->getPose().cast<float>();
    this->affine_light = frame_block->getAffineLight();
}

inline Frame::ptr Frame::create(cv::Mat image) {
    return std::make_shared<Frame>(image.data);
}

inline Frame::ptr Frame::get_ref_frame() {
    return refKF;
}

template <typename T>
inline float Frame::at(const T u, const T v, const int lvl) const {
    return (float)this->pyramid->at(u, v, lvl);
}
template <typename T>
inline float Frame::dy(const T u, const T v, const int lvl) const {
    return this->pyramid->dy(u, v, lvl);
}
template <typename T>
inline float Frame::dx(const T u, const T v, const int lvl) const {
    return this->pyramid->dx(u, v, lvl);
}
template <typename T>
inline float Frame::mag_squared(const T u, const T v, const int lvl) const {
    return this->pyramid->mag_squared(u, v, lvl);
}

inline bool Frame::is_in_image(const float u, const float v, const int lvl) const {
    return this->pyramid->is_in_image(u, v, lvl);
}

template <typename T>
inline float Frame::at(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl) const {
    return this->pyramid->at(pixel(0), pixel(1), lvl);
}
template <typename T>
inline float Frame::dy(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl) const {
    return this->pyramid->dy(pixel(0), pixel(1), lvl);
}
template <typename T>
inline float Frame::dx(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl) const {
    return this->pyramid->dx(pixel(0), pixel(1), lvl);
}
template <typename T>
inline float Frame::mag_squared(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl) const {
    return this->pyramid->mag_squared(pixel(0), pixel(1), lvl);
}

template <typename T>
inline bool Frame::is_in_image(const Eigen::Matrix<T, 2, 1>& pixel, const int lvl) const {
    return this->pyramid->is_in_image((float)pixel(0), (float)pixel(1), lvl);
}
// help function
inline Sophus::SE3f get_src_to_dst_transform(const Frame::ptr src, const Frame::ptr dst) {
    return dst->get_pose().inverse() * src->get_pose();
}

inline AffineLight get_src_to_dst_aff_light(const Frame::ptr src, const Frame::ptr dst) {
    return calc_aff_map_src_to_dst(src->get_aff_light(), dst->get_aff_light());
}

}  // namespace mpl