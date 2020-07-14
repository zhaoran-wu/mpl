#include "frame.h"

namespace mpl {

int Frame::id_cnt = 0;

Frame::Frame(const uchar* const row_image_data)
    : pyramid(new ImagePyramid(row_image_data)),
      pose(Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero()),
      affine_light(0, 0),
      id(++id_cnt) {
    cam = &CamData::getInstance();
    frame_block = std::make_unique<FrameParameterBlock>();
    assert(cam != nullptr);
}

ImagePyramid::ptr Frame::getImagePyramid() {
    return this->pyramid;
}

Frame::ptr Frame::create(cv::Mat image) {
    return std::make_shared<Frame>(image.data);
}

Frame::ptr Frame::get_ref_frame() {
    return last_KF_ref;
}

Sophus::SE3f Frame::get_T_curr_lastKF() {
    return T_curr_lastKF;
}
AffineLight Frame::get_aff_curr_lastKF() {
    return aff_light_curr_lastKF;
}

Sophus::SE3f get_src_to_dst_transform(const Frame::ptr src,
                                      const Frame::ptr dst) {
    return dst->get_pose<Sophus::SE3f>().inverse() *
           src->get_pose<Sophus::SE3f>();
}

AffineLight get_src_to_dst_aff_light(const Frame::ptr src,
                                     const Frame::ptr dst) {
    return AffineLight::calc_aff_map_src_to_dst(src->get_aff_light(),
                                                dst->get_aff_light());
}

}  // namespace mpl
   // namespace mpl