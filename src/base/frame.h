#pragma once
#include "image_pyramid.h"
#include <Eigen/Geometry>
namespace mpl {
class Frame {
   public:
    typedef std::shared_ptr<Frame> ptr;
    Frame(const uchar* const row_image_data);
    ImagePyramid::ptr getImagePyramid();

    // unproject a point in image lvl to 3D refer to current frame coordinate
    // system
    Eigen::Vector3f unproject(const Eigen::Vector2i& pixel, const ushort depth,
                              const int lvl = 0) const;
    // poject a point in current frame coordinate system to image at lvl
    Eigen::Vector2f project(const Eigen::Vector3f point, int lvl = 0) const;

   private:
    ImagePyramid::ptr pyramid;
    CamData* cam;
};

inline Eigen::Vector3f Frame::unproject(const Eigen::Vector2i& pixel,
                                        const ushort depth,
                                        const int lvl) const {
    return depth * cam->K_inv[lvl] * pixel.cast<float>().homogeneous();
}

inline Eigen::Vector2f Frame::project(const Eigen::Vector3f point,
                                      const int lvl) const {
    return (cam->K[lvl] * point).hnormalized();
}

}  // namespace mpl