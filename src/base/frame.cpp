#include "frame.h"

namespace mpl {
Frame::Frame(const uchar* const row_image_data)
    : pyramid(new ImagePyramid(row_image_data)),
      pose(Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero()),
      affine_light(0, 0) {
    cam = &CamData::getInstance();
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

}  // namespace mpl
   // namespace mpl