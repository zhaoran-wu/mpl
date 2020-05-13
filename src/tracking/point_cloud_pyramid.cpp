#include "point_cloud_pyramid.h"
#include "config.h"

namespace mpl {
PointCloudPyramid::PointCloudPyramid() {
    auto& config = Config::getInstance();
    pyramid_.resize(config.PYRAMID_LVLS);
    // avoid reallocation each time
}

PointCloud& PointCloudPyramid::operator[](const int lvl) {
    return pyramid_[lvl];
}
int PointCloudPyramid::lvls() {
    return pyramid_.size();
}
}  // namespace mpl
