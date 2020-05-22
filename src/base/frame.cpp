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

}  // namespace mpl
   // namespace mpl