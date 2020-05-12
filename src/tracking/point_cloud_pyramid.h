#pragma once
#include <Eigen/Core>
#include <memory>
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
class PointCloudPyramid {
   public:
    typedef std::shared_ptr<PointCloudPyramid> ptr;
    PointCloudPyramid();
    PointCloud& operator[](const int lvl);

    //    PointCloudReference(std::vector<Candidates>& candidates);

   private:
    std::vector<PointCloud> pyramid_;
};
}  // namespace mpl