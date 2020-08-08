#pragma once
#include <sophus/se3.hpp>
#include <string>
#include <vector>
namespace mpl {

class KittiReader {
   public:
    KittiReader() = default;
    KittiReader(const std::string file);

    Sophus::SE3f get_pose_at_index(const int idx) const;  // get T_w_c
   private:
    std::vector<Eigen::Isometry3f> pose_vec;
};
}  // namespace mpl