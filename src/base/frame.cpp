#include "frame.h"

namespace mpl {

int Frame::id_cnt = 0;

Frame::Frame(const uchar* const row_image_data)
    : pyramid(new ImagePyramid(row_image_data)), affine_light(0, 0), id(++id_cnt) {
    cam = &CamData::getInstance();
    frame_block = std::make_unique<FrameParameterBlock>();
}

}  // namespace mpl