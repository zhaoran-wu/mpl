#include "frame.h"

namespace mpl {
Frame::Frame(const uchar* const row_image_data)
    : pyramid(new ImagePyramid(row_image_data)) {
    cam = &CamData::getInstance();
}

ImagePyramid::ptr Frame::getImagePyramid() {
    return this->pyramid;
}

}  // namespace mpl
   // namespace mpl