#include "kitti.h"
#include <fstream>
#include <iostream>

namespace mpl {

KittiReader::KittiReader(const std::string file) {
    std::ifstream ifs;
    ifs.open(file);
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);

        Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
        float num;
        int id = 0;
        while (ss >> num) {
            int col = id % 4;
            int row = id / 4;

            T.matrix()(row, col) = num;
            id++;
        }

        this->pose_vec.push_back(std::move(T));
    }
    std::cout << "loading kitti pose data finish"
              << "\n";
}

Sophus::SE3f KittiReader::get_pose_at_index(const int idx) const {
    return static_cast<Sophus::SE3f>(pose_vec[idx].matrix());
}

}  // namespace mpl