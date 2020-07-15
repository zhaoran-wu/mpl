#pragma once
#include <Eigen/Core>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace mpl {

struct Voxel {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Voxel(const Eigen::Vector3f& position, const float intensity) {
        this->position = position;
        this->intensity = intensity;
    }

    Eigen::Vector3f position;
    float intensity;
};

typedef std::vector<Voxel> PointCloud;
class Candidate;
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