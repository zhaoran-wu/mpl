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
    Voxel(Candidate* can, const Eigen::Vector3f& position, const float intensity, const float alignment_weight = 1.0f)
        : position(position), intensity(intensity), aligenment_weight(alignment_weight), can(can) {
    }

    Eigen::Vector3f position;
    float intensity;
    float aligenment_weight;
    Candidate* can;  // correspond candidate ptr;

    // data used only for visualization
    struct VisualizationData {
        bool is_outlier = false;
        float last_tracking_energy = 0.0f;  // final energy in last tracking
        Eigen::Vector2f hit_pixel_in_newst_frame = Eigen::Vector2f(-1.0f, -1.0f);
        float depth_in_newst_frame = 0.0f;
        bool visible_for_newst_frame = true;  // decide after tracking
    } vis_data;
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