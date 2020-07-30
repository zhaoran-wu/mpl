#pragma once
#include <Eigen/Core>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace mpl {

class Candidate;
struct Voxel {
    Voxel() = default;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Voxel(Candidate* can, const Eigen::Vector3f& position, const float intensity, const float weight = 1)
        : position(position), intensity(intensity), weight(weight), can(can) {
    }

    Eigen::Vector3f position;
    float intensity;
    bool visible_for_newst_frame;
    Eigen::Vector2f hit_pixel_in_newst_frame;
    float depth_in_newst_frame;
    float last_tracking_energy;  // final energy in last tracking
    float weight;
    bool is_outlier = false;
    Candidate* can;  // correspond candidate;
};

typedef std::vector<Voxel> PointCloud;
class PointCloudPyramid {
   public:
    typedef std::shared_ptr<PointCloudPyramid> ptr;

    PointCloudPyramid();
    PointCloudPyramid(cv::Mat rendered_depth, const std::vector<Candidate>& semi_dense_depth);
    static ptr create(cv::Mat rendered_depth, const std::vector<Candidate>& semi_dense_depth);

    int lvls();
    PointCloud& operator[](const int lvl);

    //    PointCloudReference(std::vector<Candidates>& candidates);

   private:
    std::vector<PointCloud> pyramid_;
};
}  // namespace mpl