#include "point_cloud_pyramid.h"
#include "config.h"

namespace mpl {
PointCloudPyramid::PointCloudPyramid() {
    auto& config = Config::getInstance();
    pyramid_.resize(config.PYRAMID_LVLS);
    // avoid reallocation each time
}
PointCloudPyramid::PointCloudPyramid(cv::Mat rendered_depth,
                                     const std::vector<std::unique_ptr<Candidate>>& semi_dense_depth) {
    auto& config = Config::getInstance();
    pyramid_.resize(config.PYRAMID_LVLS);
}

PointCloud& PointCloudPyramid::operator[](const int lvl) {
    return pyramid_[lvl];
}
int PointCloudPyramid::lvls() {
    return pyramid_.size();
}

PointCloudPyramid::ptr PointCloudPyramid::create(cv::Mat rendered_depth,
                                                 const std::vector<std::unique_ptr<Candidate>>& semi_dense_depth) {
    return std::make_shared<PointCloudPyramid>(rendered_depth, semi_dense_depth);
}
}  // namespace mpl
